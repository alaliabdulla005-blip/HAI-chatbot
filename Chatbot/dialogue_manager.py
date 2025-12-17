"""
dialogue_manager.py

Coordinates:
- intent detection (SklearnIntentClassifier)
- FAQ search (FAQSearchEngine)
- simple conversation state:
  * user name / bot name
  * cart contents
  * last intent, name-prompt state

No LLMs, no AutoML – just rule-based routing + sklearn models trained
on your own data (intents.csv, faqs.csv, menu.csv).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import random
import re
from collections import Counter

from .intents import SklearnIntentClassifier, IntentLabel
from .state_models import ChatState, CartLine, FlowContext, UndoEntry
from .faq_search import FAQSearchEngine, FAQMatch
from .domain_models import MenuItem, FAQEntry, MenuOption  # adjust imports if needed
from . import guardrails
from .logging_utils import append_success_pair


@dataclass
class ChatbotConfig:
    """Tunables for thresholds etc."""
    intent_confidence_threshold: float = 0.10
    faq_min_score: float = 0.25
    faq_top_k: int = 3

    # Guardrail thresholds / decay
    max_anger_turns: int = 3
    max_abuse_turns: int = 3
    anger_decay_per_turn: int = 1      # how much to decrement anger_count on neutral turns
    abuse_decay_per_turn: int = 1      # how much to decrement abuse_count on neutral turns


# Fallback messages when the bot don't understand
FALLBACK_MESSAGES = [
    (
        "I’m not completely sure what you mean. "
        "I can help with café questions like drinks, allergens, calories, vegan/halal options, "
        "and your order."
    ),
    (
        "Sorry, I didn’t quite get that. "
        "Try asking about drinks, allergens, calories, or managing your cart and checkout."
    ),
    (
        "I might be missing something – I’m best at questions about our menu, "
        "allergens (including vegan/halal/gluten-free options), calories, and your order."
    ),
    (
        "That’s a bit outside what I can do. "
        "I’m here to help with café drinks, allergen info, calories, and your cart."
    ),
    (
        "I can’t really answer that one, but I can help you with drinks, allergens, "
        "calories, vegan/halal choices, and managing your order."
    ),
]

CALORIE_DISCLAIMER = (
    "These calorie values are approximate and based on typical Tall/Small servings. "
    "Larger sizes and customisations (different milks, syrups, toppings) will change the "
    "numbers, so always check the latest nutrition information in-store or on the brand’s website."
)


# ---------- Dialogue Manager ----------


class DialogueManager:
    """
    Core orchestrator for the chatbot.

    Responsibilities:
    - Maintain ConversationState (names, cart, last intent).
    - Call intent classifier and route to handlers.
    - Use FAQSearchEngine for “knowledge” answers (hours, allergens, etc.).
    - Enforce simple transactional rules (e.g. no checkout with empty cart).

    UI layers (CLI, Streamlit) should call `handle_message()` and display
    the returned text.
    """
    ORDER_WIZARD_FLOW = "order_wizard"


    GREET_KEYWORDS = {
        "hi","hello","hey","good morning","good afternoon",
        "good evening","hiya","yo","morning","afternoon",
    }

    # NEW: tokens we ignore when fuzzy-matching menu items,
    # so generic words like "milk" or "contains" don't cause bad matches.
    MATCH_STOPWORDS = {
        # generic function words
        "do", "does", "did", "is", "are", "was", "were",
        "the", "a", "an", "this", "that", "these", "those",
        "with", "without", "of", "for", "to", "in", "on", "at",
        "my", "your", "our", "their", "its", "it", "i", "you", "we", "they",
        "please", "pls", "can", "could", "would", "will", "shall", "should",
        "any", "some", "just", "and", "or", "plus",
        "what", "which", "how", "many", "much", "milk", "dairy", "lactose", "drink", "semi",
        "skimmed", "whole",

        # allergen / question words that shouldn't drive item matching
        "contains", "contain", "containing",
        "has", "have", "having",
        "allergen", "allergens", "allergy", "allergic",
        "safe", "suitable", "free",
        "info", "information", "list", "menu", "guide",
        "check", "show", "tell",
        "ingredients",

        # allergy / nutrition meta words
        "gluten",
        "calorie", "calories", "kcal",
        "sugar", "fat", "fats", "fatty",
        "vegan", "veg", "vegetarian", "veggie",

        # cart / container words (not part of item name)
        "cart", "basket", "order", "bag",

        # command / quantity words that shouldn't drive menu matching
        "add", "remove", "delete", "take", "put", "get", "buy", "drop",
        "cancel", "undo", "clear", "empty",
        "want", "like", "have",
        "one", "two", "three", "1", "2", "3",
    }

    ALLERGEN_CUES = {
        "allergen", "allergens", "allergy", "allergic",
        "contain", "contains", "may contain",
        "milk", "dairy", "lactose",
        "egg", "eggs", "nuts", "peanut", "gluten", "wheat", "soy", "soya", "sesame", "mustard",
    }

    CALORIE_CUES = {"calorie", "calories", "kcal", "cals"}

    def __init__(
        self,
        intent_classifier: SklearnIntentClassifier,
        faq_engine: FAQSearchEngine,
        menu_items: Sequence[MenuItem],
        menu_options: Optional[Sequence[MenuOption]] = None,
        config: Optional[ChatbotConfig] = None,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.faq_engine = faq_engine
        self.menu_items = list(menu_items)
        self._menu_vocab = self._build_menu_vocab(self.menu_items)
        self.menu_options = list(menu_options) if menu_options is not None else []
        self.options_by_id = {opt.option_id: opt for opt in self.menu_options}
        self._option_vocab = self._build_option_vocab(self.menu_options)
        self.config = config or ChatbotConfig()
        self.state = ChatState()
        self._sugar_free_notes: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_message(self, user_text: str) -> str:
        """
        Main entry point for each user turn.

        1. If we are waiting explicitly for a name → handle that directly.
        2. Otherwise, classify intent and route to a handler.
        3. Use FAQ search for knowledge-style intents.
        4. Enforce basic transaction rules (e.g. cannot checkout with empty cart).
        """
        raw = (user_text or "").strip()
        if not raw:
            return random.choice(FALLBACK_MESSAGES)

        # 1) Guardrails pass
        gr_result = guardrails.apply_guardrails(
            user_text=raw,
            state=self.state,
            max_anger_turns=self.config.max_anger_turns,
            max_abuse_turns=self.config.max_abuse_turns,
            anger_decay=self.config.anger_decay_per_turn,
            abuse_decay=self.config.abuse_decay_per_turn,
        )
        if gr_result is not None:
            return gr_result

        # 2) Pending “name” answers override normal intent detection
        if self._looks_like_undo(raw):
            return self._handle_undo()

        if self.state.awaiting_user_name:
            return self._handle_user_name_reply(raw)

        if self.state.awaiting_bot_name:
            return self._handle_bot_name_reply(raw)

        if self.state.awaiting_item_for_pending_options:
            return self._handle_pending_options_item(raw)

        if self.state.active_flow is not None:
            return self._route_active_flow(raw)

        lowered = raw.lower()

        # Start a guided order wizard on natural phrases
        if (
                "help me choose" in lowered
                or "guide my order" in lowered
                or "start order" in lowered
                or "recommend a drink" in lowered
                or "recommend something to drink" in lowered
                or "recommend food" in lowered
                or "recommend something to eat" in lowered
        ):
            # Pass the raw user text so we can inspect it in _start_order_wizard
            return self._start_order_wizard(raw)

        # 3) Direct question about the stored name
        if "what is my name" in lowered or "what's my name" in lowered or "whats my name" in lowered:
            return f"Right now, I know you as {self.state.user_name}."

        # Heuristic: user wants to change THEIR own name
        # e.g. "my name is Ali", "call me Ali"
        if re.search(r"\bmy name is\b", lowered) or re.search(r"\bcall me\b", lowered):
            return self._handle_change_user_name(raw)

        # Heuristic: user wants to change the BOT name
        # e.g. "I'll call you Jarvis", "change your name to Jarvis", "im calling you Steve"
        if any(
                phrase in lowered
                for phrase in [
                    "your name is",
                    "i'll call you",
                    "ill call you",
                    "calling you",
                    "change your name",
                    "rename you",
                    "rename yourself",
                    "rename bot",
                    "set bot name",
                    "bot rename",
                    "new name for you",
                ]
        ):
            return self._handle_change_bot_name(raw)

        if "what can you do" in lowered or "what do you do" in lowered:
            return self._handle_capabilities(raw)

        if re.search(r"\bput\b.+\b(basket|cart|order)\b", lowered):
            return self._handle_add_to_cart(raw)

        if lowered.startswith("who is ") or lowered.startswith("who's "):
            return self._handle_out_of_scope(raw)

        if lowered.startswith("who are ") or lowered.startswith("who're "):
            return self._handle_out_of_scope(raw)

        if lowered.startswith("do you know who "):
            return self._handle_out_of_scope(raw)

        for g in self.GREET_KEYWORDS:
            # Match the greeting as a whole word at the start, e.g. "yo" but not "you"
            pattern = r"^" + re.escape(g) + r"\b"
            if re.match(pattern, lowered):
                return self._handle_greet(raw)

        # options guard
        if self._looks_like_options_only(raw):
            self._push_undo("start customisations")
            self.state.pending_options_queue.append(raw)
            self.state.awaiting_item_for_pending_options = True
            return "Those sound like customisations (milk/syrups/shots). Which drink or food should I apply them to?"

        # If the user explicitly mentions calories, handle calories BEFORE allergen-cue overrides.
        # This fixes cases like: "calories in latte with oat milk ..."
        ORDERish = {"add", "remove", "delete", "checkout", "order", "buy", "put"}
        if any(cue in lowered for cue in self.CALORIE_CUES):
            if not any(f" {w} " in f" {lowered} " for w in ORDERish):
                return self._handle_calories(raw)

        # allergen cue override (avoid overriding order commands)
        if any(cue in lowered for cue in self.ALLERGEN_CUES):
            if not any(f" {w} " in f" {lowered} " for w in ORDERish):
                if self._find_menu_item_in_text(raw) or self._find_menu_item_in_text_relaxed(raw):
                    return self._handle_item_allergens(raw)

        if self._looks_like_show_menu_request(lowered):
            return self._handle_show_menu(raw)

        # 4) Intent detection
        intent_label, confidence = self.intent_classifier.predict_intent(raw)

        # NEW: override misclassified calorie math expressions
        # e.g. "Latte grande + vanilla × 2" should be ASK_CALORIES, not ADD_TO_CART
        if self._looks_like_calorie_expression(raw):
            intent_label = IntentLabel.ASK_CALORIES
            confidence = 1.0

            if self._find_menu_item_in_text_relaxed(raw) and len(raw.split()) <= 4:
                return self._handle_calories(raw)

        # Let guardrails down-weight obviously misclassified emotional intents
        intent_label = guardrails.filter_emotional_intent(intent_label, lowered)

        if intent_label is None or confidence < self.config.intent_confidence_threshold:
            # Ask guardrails for emotional signals and choose a suitable reply
            flags = guardrails.get_emotion_flags(lowered)

            if flags.get("insult"):
                return self._handle_user_insult(raw)
            if flags.get("anger"):
                return self._handle_user_angry(raw)
            if flags.get("complaint"):
                return self._handle_user_complaint(raw)

            # Otherwise: generic fallback
            return self._handle_fallback(raw)

        # 5) Dispatch to intent-specific handlers
        handler_map = {
            IntentLabel.GREET: self._handle_greet,
            IntentLabel.ASK_OPENING_HOURS: self._handle_opening_hours,
            IntentLabel.ASK_ALLERGENS_GENERAL: self._handle_allergens_general,
            IntentLabel.ASK_ITEM_ALLERGENS: self._handle_item_allergens,
            IntentLabel.ASK_VEGAN_OPTIONS: self._handle_vegan_options,
            IntentLabel.ASK_HALAL_OPTIONS: self._handle_halal_options,
            IntentLabel.ASK_GLUTEN_FREE: self._handle_gluten_free,
            IntentLabel.ASK_CALORIES: self._handle_calories,
            IntentLabel.ADD_TO_CART: self._handle_add_to_cart,
            IntentLabel.REMOVE_FROM_CART: self._handle_remove_from_cart,
            IntentLabel.SHOW_CART: self._handle_show_cart,
            IntentLabel.SHOW_MENU: self._handle_show_menu,
            IntentLabel.CHECKOUT: self._handle_checkout,
            IntentLabel.CHANGE_USER_NAME: self._handle_change_user_name,
            IntentLabel.CHANGE_BOT_NAME: self._handle_change_bot_name,
            IntentLabel.USER_THANKS: self._handle_user_thanks,
            IntentLabel.USER_COMPLAINT: self._handle_user_complaint,
            IntentLabel.USER_ANGRY: self._handle_user_angry,
            IntentLabel.USER_INSULT: self._handle_user_insult,
            IntentLabel.OUT_OF_SCOPE: self._handle_out_of_scope,
            IntentLabel.UNDO: self._handle_undo,
        }

        handler = handler_map.get(intent_label)

        if handler is None:
            # Unknown or not yet wired intent label
            return self._handle_fallback(raw)

        response = handler(raw)

        # Record the successfully handled intent for future turns / analytics
        self.state.last_intent = intent_label

        # TODO later:
        # - integrate guardrails (mood, abusive language, privacy)
        # - log successful <input, intent> pairs

        return response

    # ------------------------------------------------------------------
    # Handlers: name / greetings
    # ------------------------------------------------------------------

    def _handle_greet(self, _: str) -> str:
        return (
            f"Hi {self.state.user_name}! I’m {self.state.bot_name}. "
            "You can ask about allergens (including vegan/halal/gluten-free options), "
            "calories, start an order by adding or removing items, showing your cart, "
            "and checking out."
        )

    def _handle_capabilities(self, _: str) -> str:
        bot = self.state.bot_name

        lines: List[str] = [
            f"{bot}: I’m your café assistant. Quick guide:",
            "",
            "**Menu**",
            "- Type `show menu` — show the menu",
            "- Type `show drink menu` — drinks only",
            "- Type `show food menu` — food only",
            "- Type `starbucks menu` / `costa menu` — show a specific brand (if available)",
            "",
            "**Allergens & dietary**",
            "- Ask: `Does \"<item>\" contain milk/nuts/gluten/soy?`",
            "- Ask: `Is \"<item>\" vegan?` / `halal-friendly?` / `gluten-free?`",
            "- Ask: `What vegan/halal/gluten-free options do you have?`",
            "",
            "**Calories (approx.)**",
            "- Ask: `How many calories are in \"<item>\"?`",
            "- Ask: `Calories in a latte with oat milk + vanilla syrup`",
            "",
            "**Order / cart**",
            "- Type `add \"<item name>\"` — add to cart (you can include options like oat milk, extra shot, syrups)",
            "- Type `remove \"<item name>\"` — remove one from cart",
            "- Type `show cart` — view cart",
            "- Type `checkout` — place order",
            "",
            "**Customisations**",
            "- You can type options like `oat milk`, `vanilla syrup x 2`, `extra shot`",
            "- If you send options alone, I’ll ask which item to apply them to",
            "",
            "**Personalisation**",
            "- Type `my name is <name>` / `call me <name>`",
            "- Type `from now on call you <name>` (rename the bot)",
            "",
            "**Undo**",
            "- Type `undo` — revert the last action",
        ]
        return "\n".join(lines)

    def _handle_change_user_name(self, user_text: str) -> str:
        """
        Try to extract the USER's name from this message.
        If not clear, ask a follow-up and wait for the next turn.
        """
        name = self._extract_user_name(user_text)
        if name:
            self._push_undo("change user name")
            self.state.user_name = name
            self.state.awaiting_user_name = False
            return f"Got it, I’ll call you {name} from now on."

        self._push_undo("start user name change")
        self.state.awaiting_user_name = True
        return "Sure – what should I call you?"

    def _handle_change_bot_name(self, user_text: str) -> str:
        name = self._extract_bot_name(user_text)
        if name:
            self._push_undo("change bot name")
            self.state.bot_name = name
            self.state.awaiting_bot_name = False
            return f"Okay, I’ll go by {name} now."
        self.state.awaiting_bot_name = True
        return "What name would you like to give me?"

    def _handle_user_name_reply(self, user_text: str) -> str:
        # Don't treat questions like "what's my name?" as the name itself
        lowered = user_text.lower()
        if "name" in lowered and ("what" in lowered or "?" in lowered):
            return "I didn’t quite catch a name there – try something like “Ali” or “Sara”."

        # Reuse the same extraction logic (handles "my name is Ali", "I'm Ali", or just "Ali")
        name = self._extract_user_name(user_text) or self._clean_name(user_text)
        if not name:
            return "I didn’t quite catch that – could you give me a short name?"

        # Optional polish: normalise stored names to title case
        self._push_undo("change user name")
        self.state.user_name = name.strip().title()
        self.state.awaiting_user_name = False
        return f"Nice to meet you, {self.state.user_name}!"

    def _handle_bot_name_reply(self, user_text: str) -> str:
        name = self._extract_bot_name(user_text) or self._clean_name(user_text)
        if not name:
            return "I’m not sure that works as a name – try something shorter?"
        self._push_undo("change bot name")
        self.state.bot_name = name
        self.state.awaiting_bot_name = False
        return f"Great, I’ll be {name}."


    def _extract_user_name(self, text: str) -> Optional[str]:
        """
        Extract a USER name from a sentence.

        Handles things like:
        - "my name is Ali"
        - "I am Ali" / "I'm Ali" / "im Ali"
        - "call me Ali"
        - "from now on call me Ali"

        Returns a cleaned name or None if we can't be confident.
        """
        lowered = text.lower()

        patterns = [
            r"\bmy name is\s+(.+)$",
            r"\bi am\s+(.+)$",
            r"\bi'm\s+(.+)$",
            r"\bim\s+(.+)$",
            r"\bcall me\s+(.+)$",
            r"\bname is\s+(.+)$",
            r"\bnew name is\s+(.+)$",
            r"\bfrom now on call me\s+(.+)$",
        ]

        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                start = m.start(1)
                name_raw = text[start:].strip()
                return self._clean_name(name_raw)

        # Fallback: treat the whole message as a name *only* if it's very short
        # and NOT a "command" like "call me" or "my name ..."
        if "call me" not in lowered and "name" not in lowered:
            tokens = re.findall(r"[^\s]+", text.strip())
            if 0 < len(tokens) <= 3:
                return self._clean_name(text)

        return None

    def _extract_bot_name(self, text: str) -> Optional[str]:
        """
        Extract a BOT name from a sentence.

        Handles things like:
        - "your name is Jarvis"
        - "I'll call you Jarvis"
        - "change your name to Jarvis"
        - "im calling you Steve"
        - "rename the bot to Helper"
        """
        lowered = text.lower()

        patterns = [
            r"\byour name is\s+(.+)$",
            r"\bi['’]ll call you\s+(.+)$",
            r"\bill call you\s+(.+)$",
            r"\bim calling you\s+(.+)$",
            r"\bi am calling you\s+(.+)$",
            r"\bchange your name to\s+(.+)$",
            r"\brename (?:you|yourself|the bot|bot) to\s+(.+)$",
            r"\bset bot name to\s+(.+)$",
            r"\bnew name for you\s+(.+)$",
        ]

        for pattern in patterns:
            m = re.search(pattern, lowered)
            if m:
                start = m.start(1)
                name_raw = text[start:].strip()
                return self._clean_name(name_raw)

        # Fallback for the "what should I call you?" follow-up:
        # if user just replies "Jarvis"
        tokens = re.findall(r"[^\s]+", text.strip())
        # Only treat as pure name if it's short and doesn't look like another command
        if not any(kw in lowered for kw in ["your name", "call you", "calling you", "rename", "bot name", "bot"]):
            if 0 < len(tokens) <= 3:
                return self._clean_name(text)

        return None

    @staticmethod
    def _clean_name(raw: str) -> Optional[str]:
        cleaned = raw.strip().strip(".,!?;:\"'()[]{}")
        if not cleaned:
            return None

        # Normalise whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        parts = cleaned.split()
        # Hard cap to avoid absurdly long "names"
        if len(parts) > 4:
            cleaned = " ".join(parts[:4])

        return cleaned

    # ------------------------------------------------------------------
    # Handlers: FAQ-backed intents
    # ------------------------------------------------------------------

    def _handle_opening_hours(self, _: str) -> str:
        # Direct lookup by FAQ ID if present, otherwise search by tag
        faq = self.faq_engine.get_faq_by_id("opening_hours")
        if faq:
            return faq.answer_markdown

        matches = self.faq_engine.search(
            "opening hours",
            top_k=1,
            min_score=self.config.faq_min_score,
            required_tags={"hours"},
        )
        if matches:
            return matches[0].faq.answer_markdown

        return "Our opening hours are currently not available – please check the notice board in the café."

    def _handle_allergens_general(self, _: str) -> str:
        matches = self.faq_engine.search(
            "which allergens do you check for",
            top_k=1,
            min_score=self.config.faq_min_score,
            required_tags={"allergens"},
        )
        if matches:
            return matches[0].faq.answer_markdown

        return random.choice(FALLBACK_MESSAGES)

    def _handle_item_allergens(self, user_text: str) -> str:
        """
        Basic, non-LLM item allergen logic:
        - Try to match a menu item by (fuzzy) name.
        - If found, summarise its allergens and veg/vegan suitability.
        """
        item = self._find_menu_item_in_text(user_text)
        if item is None:
            return (
                "I couldn’t be sure which item you mean. "
                "Try using the exact drink name, like “Misto – oat drink” or “Costa Express Hot Chocolate”."
            )

        return self._format_item_allergen_answer(item)

    def _handle_vegan_options(self, user_text: str) -> str:
        # 1) If the user is asking about a specific item, answer that item.
        item = self._find_menu_item_in_text(user_text) or self._find_menu_item_in_text_relaxed(user_text)
        if item is not None:
            return self._format_item_allergen_answer(item)

        # 2) Otherwise, treat it as a general “what vegan options do you have?” question.
        faq = self.faq_engine.get_faq_by_id("menu_vegan_examples")
        if faq:
            return faq.answer_markdown

        vegan_drinks = [m for m in self.menu_items if getattr(m, "suitable_vegan", False)]
        if not vegan_drinks:
            return (
                "Vegan options are mainly drinks built with plant milks and no dairy toppings. "
                "Please check individual item allergen info in the app."
            )

        names = ", ".join(sorted({m.name for m in vegan_drinks}))
        return f"Vegan-friendly drinks on our current menu include: {names}."

    def _handle_halal_options(self, user_text: str) -> str:
        """
        Handle both general and item-specific halal questions.

        - If we can confidently match a specific menu item, delegate to _answer_item_halal.
        - Otherwise, give a friendly overview of halal-friendly, vegan and clearly non-halal items.
        """
        # 1) Item-specific question?
        item = self._find_menu_item_in_text(user_text)
        if item is not None:
            return self._answer_item_halal(item)

        # 2) General overview from the menu
        halal_examples: List[str] = []
        vegan_examples: List[str] = []
        non_halal_examples: List[str] = []

        for m in self.menu_items:
            flags = self._get_halal_flags(m)

            if flags["halal_friendly"]:
                halal_examples.append(m.name)
            elif flags["vegan_hint"]:
                vegan_examples.append(m.name)
            elif flags["non_halal"]:
                non_halal_examples.append(m.name)

        def _short_list(names: List[str], limit: int) -> str:
            unique = sorted(set(names))
            if not unique:
                return ""
            if len(unique) <= limit:
                return ", ".join(unique)
            return ", ".join(unique[:limit]) + " and others"

        halal_text = _short_list(halal_examples, 3)
        vegan_text = _short_list(vegan_examples, 4)
        non_halal_text = _short_list(non_halal_examples, 3)

        parts: List[str] = []

        parts.append(
            "From a halal point of view, here’s how our menu looks based on the "
            "ingredient and allergen information we have (this is guidance, not a religious ruling):"
        )

        if halal_text:
            parts.append(
                f"- **Halal-friendly by our tags** (no pork or alcohol listed in our data): "
                f"for example {halal_text}."
            )

        if vegan_text:
            parts.append(
                f"- **Vegan by ingredients** (no meat or dairy – many halal-conscious customers "
                f"choose these): for example {vegan_text}."
            )

        if non_halal_text:
            parts.append(
                f"- **Clearly not suitable for a halal diet** (contains pork or similar ingredients): "
                f"for example {non_halal_text}."
            )

        parts.append(
            "If you tell me a specific drink or food name (like “Signature Grilled Cheese Toastie” "
            "or “Americano black”), I can check its ingredients and halal-friendliness for you."
        )

        parts.append(
            "For strict halal decisions, please always rely on the latest official allergen/nutrition "
            "guides, packaging, staff advice, or a trusted scholar – I’m only summarising what’s in our CSV."
        )

        return "\n".join(parts)

    def _handle_gluten_free(self, user_text: str) -> str:
        """
        Answer gluten-free questions.

        - If we can safely match a specific menu item, return its full
          allergen info via _format_item_allergen_answer.
        - If the question *mentions* menu-like tokens but we can't safely
          match, we give the same "unknown item" message as other
          allergen queries (rather than pretending we know).
        - Otherwise, treat it as a general gluten-free menu question and
          fall back to the coeliac FAQ / generic guidance.
        """
        # 1) Try specific item match first
        item = self._find_menu_item_in_text(user_text)
        if item is not None:
            return self._format_item_allergen_answer(item)

        # 2) Looks item-specific but we couldn't safely match
        if self._looks_like_item_query(user_text):
            return (
                "I couldn’t be sure which item you mean. "
                "Try using the exact drink or food name as shown in the app, "
                "for example “Chocolate Brownie” or “Caffè Latte”."
            )

        # 3) General gluten-free question → FAQ based answer
        match = self.faq_engine.get_faq_by_id("coeliac_safe_menu")
        if match:
            return match.answer_markdown

        return (
            "Many items contain cereals with gluten (wheat, barley, oats, rye). "
            "Plain espresso or Americano (black) usually have no gluten ingredients, "
            "but there can still be cross-contact from shared equipment."
        )

    def _handle_calories(self, user_text: str) -> str:
        """
        Answer calorie questions.

        - If the text looks like a 'math-style' expression (e.g. "latte + muffin",
          "latte x 2 + muffin"), we try to parse it as multiple items and sum
          their calories.
        - Otherwise, we treat it as a normal question about a single item and
          look up that item's calories.
        """
        text = (user_text or "").strip()
        if not text:
            # Nothing to go on – generic explanation
            faq = self.faq_engine.get_faq_by_id("calorie_math")
            if faq:
                return (
                        "Tell me the drink or food you’re asking about, "
                        "and I can calculate an approximate calorie total.\n\n"
                        + faq.answer_markdown
                )
            return (
                "Tell me the drink or food you’re asking about, "
                "and I’ll calculate an approximate calorie total based on our menu data."
            )

        # Expression like "latte + muffin" or "latte x 2 + cookie"
        if self._looks_like_calorie_expression(text):
            return self._answer_calorie_expression(text)

        # Simple question like "how many calories in a latte?"
        return self._answer_calorie_simple(text)

    # ------------------------------------------------------------------
    # Handlers: cart / transaction state
    # ------------------------------------------------------------------

    # Inside DialogueManager
    # Inside DialogueManager
    def _start_order_wizard(self, user_text: str) -> str:
        lowered = (user_text or "").lower()

        category: Optional[str] = None
        if "drink" in lowered:
            category = "beverage"
        elif "food" in lowered or "snack" in lowered:
            category = "food"

        if category is not None:
            flow = FlowContext(
                name=self.ORDER_WIZARD_FLOW,
                stage="ask_item",
                data={"category": category},
            )
            self.state.active_flow = flow

            if category == "beverage":
                return (
                    "Let’s pick a drink from the menu.\n\n"
                    "What drink are you thinking about? For example, "
                    "“Latte”, “Americano (black)” or “Mocha”."
                )
            else:
                return (
                    "Let’s pick something to eat.\n\n"
                    "What food or snack are you thinking about? For example, "
                    "“Blueberry Muffin” or “Cheese & Ham Toastie”."
                )

        # Ambiguous “help me choose …” → ask drink vs food first
        flow = FlowContext(
            name=self.ORDER_WIZARD_FLOW,
            stage="ask_category",
            data={},
        )
        self.state.active_flow = flow

        return (
            "Let’s build your order step by step.\n\n"
            "First, are you looking for a **drink** or a **food** item?"
        )

    def _route_active_flow(self, user_text: str) -> str:
        """
        Route the current user turn into whichever flow is active.
        """
        ctx = self.state.active_flow
        if ctx is None:
            # Safety – fall back to normal handling if something got cleared.
            return self._handle_fallback(user_text)

        if ctx.name == self.ORDER_WIZARD_FLOW:
            return self._order_wizard_step(user_text, ctx)

        # Unknown flow – cancel it and fall back
        self.state.active_flow = None
        return self._handle_fallback(user_text)

    def _order_wizard_step(self, user_text: str, ctx: FlowContext) -> str:
        """
        Handle one step of the 'order_wizard' flow.

        Stages:
        - ask_category  -> store 'drink' or 'food'
        - ask_item      -> store the raw item phrase
        - ask_options   -> store extras phrase, then call _handle_add_to_cart
        """
        text = (user_text or "").strip()
        lowered = text.lower()

        # Allow the user to cancel out of the wizard at any time
        if lowered in {"cancel", "stop", "exit", "nevermind", "never mind"}:
            self.state.active_flow = None
            return (
                "No problem – I’ve cancelled the step-by-step ordering. "
                "You can still add items directly, for example "
                "“add a tall latte with oat milk to my order”."
            )

        # -------- Stage 1: ask_category --------
        if ctx.stage == "ask_category":
            if any(w in lowered for w in ["drink", "coffee", "beverage"]):
                ctx.data["category"] = "beverage"
            elif any(w in lowered for w in ["food", "snack", "sandwich", "pastry"]):
                ctx.data["category"] = "food"
            else:
                return (
                    "Please say whether you want a **drink** or a **food** item, "
                    "for example “a drink” or “a food snack”."
                )

            ctx.stage = "ask_item"
            category = ctx.data["category"]

            if category == "beverage":
                return (
                    "Let’s pick a drink from the menu.\n\n"
                    "What drink are you thinking about? For example, "
                    "“Latte”, “Americano (black)” or “Mocha”."
                )
            else:
                return (
                    "Let’s pick something to eat.\n\n"
                    "What food or snack are you thinking about? For example, "
                    "“Blueberry Muffin” or “Cheese & Ham Toastie”."
                )

        # -------- Stage 2: ask_item --------
        if ctx.stage == "ask_item":
            if len(text) < 2:
                category = ctx.data.get("category")
                if category == "food":
                    return (
                        "Could you give me the food name, "
                        "like “Blueberry Muffin” or “Cheese & Ham Toastie”?"
                    )
                else:
                    return (
                        "Could you give me the drink name, "
                        "like “Latte” or “Hot Chocolate”?"
                    )

            # If user accidentally includes "with ...", try to match only the left side
            item_phrase = text
            if re.search(r"\bwith\b", text, flags=re.IGNORECASE):
                item_phrase = re.split(r"\bwith\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()

            qty = self._extract_quantity(item_phrase)
            ctx.data["quantity"] = qty

            # Match item NOW and store it so options can't change it later
            item = self._find_menu_item_in_text(item_phrase) or self._find_menu_item_in_text_relaxed(item_phrase)
            if item is None:
                wanted = "food" if ctx.data.get("category") == "food" else "drink"
                return (
                    f"I couldn’t find that {wanted} in the menu. "
                    "Try the exact name, for example “Blueberry Muffin”, “Latte”, or “Mocha”."
                )

            # Enforce category if we can
            chosen_cat = (ctx.data.get("category") or "").lower()
            item_cat = (getattr(item, "category", "") or "").lower()
            if chosen_cat and item_cat and chosen_cat != item_cat:
                return (
                    f"That sounds like a **{item_cat}**, but you said you want **{chosen_cat}**. "
                    "Could you give me the exact name of the item in the category you want?"
                )

            ctx.data["selected_item_id"] = item.item_id
            ctx.data["selected_item_name"] = item.name

            ctx.stage = "ask_options"

            category = ctx.data.get("category")
            if category == "food":
                return (
                    "Got it. Do you want any extras?\n"
                    "If you’re not sure, say “no extras”. (Food extras are limited in my data.)"
                )

            return (
                "Got it. Do you want any extras like **oat milk**, **vanilla syrup**, "
                "**extra espresso shot**, or toppings?\n"
                "You can say “no extras” if you don’t want anything added."
            )

        # -------- Stage 3: ask_options --------
        if ctx.stage == "ask_options":
            if lowered in {"no", "none", "no extras", "nothing"}:
                options_phrase = ""
            elif "no extra" in lowered and "syrup" not in lowered:
                options_phrase = ""
            else:
                options_phrase = text

            # Get the stored item
            item_id = ctx.data.get("selected_item_id")
            item = next((m for m in self.menu_items if m.item_id == item_id), None)

            # Clear flow BEFORE adding
            self.state.active_flow = None

            if item is None:
                return self._handle_fallback(user_text)

            add_reply = self._add_specific_item_to_cart(
                item=item,
                options_text=options_phrase,
                log_text=user_text,
                quantity=ctx.data.get("quantity", 1),
            )

            return (
                    "Okay, I’ve used your answers to build an order and add it via "
                    "the normal system.\n\n"
                    + add_reply
            )

        # Unknown stage – reset flow and fall back
        self.state.active_flow = None
        return self._handle_fallback(user_text)

    def _handle_add_to_cart(self, user_text: str) -> str:
        """
        Simple add-to-cart handler:

        - Use the more relaxed item matcher so we can handle phrases like
          "add latte + vanilla syrup x 2 + extra shot to cart".
        - Parse options (size, milk, syrups, shots, toppings, sweeteners)
          from the same utterance and attach them to the cart line.
        """
        # Use relaxed matching so "latte + vanilla syrup x 2" still finds "Latte"
        item = self._find_menu_item_in_text_relaxed(user_text)
        if item is None:
            return (
                "I’m not sure which item to add. "
                "Try saying something like “add a tall Misto with oat drink to my order”."
            )
        quantity = self._extract_item_quantity(user_text, item)
        self._push_undo(f"add {item.name}")

        # Detect options for this item (sizes, milks, syrups, shots, etc.)
        option_pairs = self._parse_options_from_text(user_text, base_item=item)
        selected_option_ids: List[str] = []
        for opt, qty in option_pairs:
            for _ in range(max(qty, 1)):
                selected_option_ids.append(opt.option_id)

        # Merge lines only if item_id AND options match
        def same_config(line: CartLine) -> bool:
            return (
                    line.item_id == item.item_id
                    and sorted(line.selected_options) == sorted(selected_option_ids)
            )

        for line in self.state.cart:
            if same_config(line):
                line.quantity += quantity
                break
        else:
            self.state.cart.append(
                CartLine(
                    item_id=item.item_id,
                    item_name=item.name,
                    quantity=quantity,
                    selected_options=selected_option_ids,
                )
            )

        # Log this as a clearly successful add to cart interaction
        try:
            append_success_pair(user_text, IntentLabel.ADD_TO_CART)
        except Exception:
            # Logging must never break the chat
            pass

        qty_label = f" × {quantity}" if quantity > 1 else ""

        if selected_option_ids:
            options_label = self._format_options_summary(selected_option_ids)
            base_reply = f"Added {item.name}{qty_label} ({options_label}) to your cart."
        else:
            base_reply = f"Added {item.name}{qty_label} to your cart."

        # Attach any sugar-free fallback explanations
        notes = self._consume_sugar_free_notes()
        if notes:
            return base_reply + "\n\n" + "\n".join(notes)

        return base_reply

    def _add_specific_item_to_cart(
            self,
            item: MenuItem,
            options_text: str,
            log_text: str,
            quantity: int = 1,
    ) -> str:
        """
        Add a *known* menu item to cart, parsing options only from options_text.
        Used by the order wizard to avoid re-matching the item.
        """
        option_pairs = self._parse_options_from_text(options_text or "", base_item=item)

        selected_option_ids: List[str] = []
        for opt, qty in option_pairs:
            for _ in range(max(qty, 1)):
                selected_option_ids.append(opt.option_id)

        def same_config(line: CartLine) -> bool:
            return (
                    line.item_id == item.item_id
                    and sorted(line.selected_options) == sorted(selected_option_ids)
            )

        for line in self.state.cart:
            if same_config(line):
                line.quantity += quantity
                break
        else:
            self.state.cart.append(
                CartLine(
                    item_id=item.item_id,
                    item_name=item.name,
                    quantity=quantity,
                    selected_options=selected_option_ids,
                )
            )

        try:
            append_success_pair(log_text, IntentLabel.ADD_TO_CART)
        except Exception:
            pass

        qty_label = f" × {quantity}" if quantity > 1 else ""
        if selected_option_ids:
            options_label = self._format_options_summary(selected_option_ids)
            base_reply = f"Added {item.name}{qty_label} ({options_label}) to your cart."
        else:
            base_reply = f"Added {item.name}{qty_label} to your cart."

        notes = self._consume_sugar_free_notes()
        if notes:
            return base_reply + "\n\n" + "\n".join(notes)

        return base_reply

    def _is_clear_cart_command(self, lowered: str) -> bool:
        """
        Detect phrases that clearly mean 'empty the whole cart',
        regardless of which specific items are inside.
        """
        # Normalised phrases
        clear_phrases = [
            "empty cart",
            "empty my cart",
            "empty the cart",
            "clear cart",
            "clear my cart",
            "clear the cart",
            "clear basket",
            "clear my basket",
            "clear the basket",
            "clear order",
            "clear my order",
            "delete my order",
            "delete all items",
            "remove all items",
            "remove everything",
            "delete everything",
            "clear all",
        ]
        if any(p in lowered for p in clear_phrases):
            return True

        # Generic pattern: "empty"/"clear" + "cart"/"basket"/"order"
        if (("empty" in lowered or "clear" in lowered)
                and ("cart" in lowered or "basket" in lowered or "order" in lowered)):
            return True

        return False

    def _handle_remove_from_cart(self, user_text: str) -> str:
        lowered = (user_text or "").lower()

        if not self.state.cart:
            return "Your cart is currently empty."

        # 1) Bulk clear-cart commands: "delete all items", "clear cart", "remove everything", etc.
        if self._is_clear_cart_command(lowered):
            self._push_undo("clear cart")
            self.state.cart.clear()
            return "Your cart is now empty."

        # 2) Try to match a specific menu item name in the text.
        item = self._find_menu_item_in_text(user_text)

        if item is None:
            # 2a) Fallback: if there is only *one distinct line* in the cart,
            #     interpret this as "remove one of whatever is in there",
            #     unless the user clearly asked to remove *all*.
            if len(self.state.cart) == 1:
                line = self.state.cart[0]
                self._push_undo("remove from cart")

                # If they explicitly mentioned "all" / "everything", drop the whole line.
                if re.search(r"\b(all|everything)\b", lowered):
                    removed = self.state.cart.pop()
                    return f"Removed {removed.item_name} from your cart."

                # If quantity > 1: decrement instead of wiping it out.
                if line.quantity > 1:
                    # If they say "one" / "1", be explicit
                    if re.search(r"\b(one|1)\b", lowered):
                        line.quantity -= 1
                        return f"Removed one {line.item_name} from your cart."
                    # Default behaviour: still just remove one
                    line.quantity -= 1
                    return f"Removed one {line.item_name} from your cart."

                # quantity == 1 → safe to remove completely
                removed = self.state.cart.pop()
                return f"Removed {removed.item_name} from your cart."

            # 2b) More than one distinct line in the cart and we couldn't match an item
            return (
                "I couldn’t work out which item to remove. "
                "Please mention the drink or food name that’s in your cart."
            )

        # 3) We matched a specific menu item → remove from that line only
        target_base = self._base_item_name(item.name)

        def same_base(line: CartLine) -> bool:
            return self._base_item_name(line.item_name) == target_base

        will_remove = any(line.item_id == item.item_id for line in self.state.cart)
        if not will_remove:
            base_matches = [line for line in self.state.cart if same_base(line)]
            will_remove = (len(base_matches) == 1)

        if will_remove:
            self._push_undo("remove from cart")

        new_cart: List[CartLine] = []
        removed_any = False
        removed_name: Optional[str] = None

        # First pass: try exact item_id match
        for line in self.state.cart:
            if line.item_id == item.item_id:
                removed_any = True
                removed_name = line.item_name
                if line.quantity > 1:
                    line.quantity -= 1
                    new_cart.append(line)
                # quantity == 1 → drop the line
            else:
                new_cart.append(line)

        # If no exact match, but exactly one cart line shares the same base name
        if not removed_any:
            # restore original cart to try base-name removal
            new_cart = []
            base_matches = [line for line in self.state.cart if same_base(line)]

            if len(base_matches) == 1:
                match_id = base_matches[0].item_id
                for line in self.state.cart:
                    if line.item_id == match_id:
                        removed_any = True
                        removed_name = line.item_name
                        if line.quantity > 1:
                            line.quantity -= 1
                            new_cart.append(line)
                        # quantity == 1 → drop this line
                    else:
                        new_cart.append(line)

        self.state.cart = new_cart

        if not removed_any:
            return (
                "I couldn’t work out which item to remove. "
                "Please mention the drink or food name that’s in your cart."
            )

        return f"Removed one {removed_name} from your cart."

    def _handle_show_cart(self, _: str) -> str:
        if not self.state.cart:
            return "Your cart is empty at the moment."

        lines: List[str] = []
        total_kcal: float = 0.0
        total_price: float = 0.0

        for line in self.state.cart:
            line_kcal = self._estimate_line_calories(line)
            line_price = self._estimate_line_price(line)

            total_kcal += line_kcal
            total_price += line_price

            options_summary = self._format_options_summary(
                getattr(line, "selected_options", []) or []
            )

            if options_summary:
                base_desc = f"{line.item_name} × {line.quantity} ({options_summary})"
            else:
                base_desc = f"{line.item_name} × {line.quantity}"

            parts: List[str] = [base_desc]

            if line_price > 0:
                parts.append(f"~£{line_price:.2f}")
            if line_kcal > 0:
                parts.append(f"~{line_kcal:.0f} kcal")

            lines.append("- " + " — ".join(parts))

        cart_text = "\n".join(lines)

        # Build total line depending on what we have
        if total_price > 0 and total_kcal > 0:
            return (
                f"Here’s what’s in your cart:\n{cart_text}\n\n"
                f"Estimated total: **£{total_price:.2f}** and "
                f"**~{total_kcal:.0f} kcal** for your whole cart."
            )
        elif total_price > 0:
            return (
                f"Here’s what’s in your cart:\n{cart_text}\n\n"
                f"Estimated total: **£{total_price:.2f}** for your whole cart."
            )
        elif total_kcal > 0:
            return (
                f"Here’s what’s in your cart:\n{cart_text}\n\n"
                f"Estimated total: **~{total_kcal:.0f} kcal** for your whole cart."
            )

        return f"Here’s what’s in your cart:\n{cart_text}"

    def _handle_checkout(self, user_text: str) -> str:
        """
        Transaction-state rule:
        - You can only checkout if there is at least one item in the cart.
        """
        if not self.state.cart:
            return "You don’t have anything in your cart yet. Add a drink or snack before checking out."

        # For now, we just summarise and clear the cart;
        # later you can compute price, calories, etc.
        summary = self._handle_show_cart("")
        self._push_undo("checkout")
        self.state.cart.clear()

        # Log this successful CHECKOUT intent
        try:
            append_success_pair(user_text, IntentLabel.CHECKOUT)
        except Exception:
            pass

        return summary + "\n\nYour order has been placed. Thank you!"

    # ------------------------------------------------------------------
    # Handlers: social / feedback
    # ------------------------------------------------------------------

    def _handle_user_thanks(self, user_text: str) -> str:
        """
        Handle USER_THANKS:
        - Reply politely.
        - Log this as a successful 'user_thanks' example.
        - Additionally, if the *previous* intent looked like a contentful task
          (e.g. ASK_ITEM_ALLERGENS, ASK_CALORIES, ADD_TO_CART, CHECKOUT),
          log a synthetic positive-feedback marker for that intent too.
        """
        # 1) Log the thanks utterance itself
        try:
            append_success_pair(user_text, IntentLabel.USER_THANKS)
        except Exception:
            pass

        # 2) Log a 'success' marker for the previous contentful intent, if any
        prev_intent = getattr(self.state, "last_intent", None)
        if prev_intent and prev_intent not in {
            IntentLabel.USER_THANKS,
            IntentLabel.USER_COMPLAINT,
            IntentLabel.USER_ANGRY,
            IntentLabel.USER_INSULT,
            IntentLabel.OUT_OF_SCOPE,
        }:
            try:
                marker_utterance = f"[positive_feedback_for:{prev_intent.value}]"
                append_success_pair(marker_utterance, prev_intent)
            except Exception:
                pass

        replies = [
            "You’re welcome!",
            "Happy to help 😊",
            "Any time – let me know if you need anything else.",
        ]
        return random.choice(replies)

    def _handle_user_complaint(self, _: str) -> str:
        replies = [
            "I’m sorry this is frustrating. I’m still a prototype – your feedback helps improve me.",
            "Sorry it’s not working as you hoped. You can also ask a staff member in person for help.",
        ]
        return random.choice(replies)

    def _handle_user_angry(self, _: str) -> str:
        replies = [
            "I’m really sorry this is frustrating. I’ll try to be clearer – you can ask about allergens, vegan/halal options, or your order.",
            "Sorry it’s not going as you hoped. I can help with allergens, drinks, food options and your cart – what would you like to do?",
        ]
        return random.choice(replies)

    def _handle_user_insult(self, _: str) -> str:
        replies = [
            "I’m here to help, not to upset you. Let’s focus on your drink or allergen question.",
            "I might not get everything right yet, but I’m trying to help. Tell me what you need about the menu or your order.",
        ]
        return random.choice(replies)

    def _handle_out_of_scope(self, _: str) -> str:
        return (
            "I can’t help with that kind of question – I’m just a café assistant. "
            "I can answer questions about drinks, allergens, vegan/halal/gluten-free options, "
            "and manage your order (cart and checkout)."
        )


    # ------------------------------------------------------------------
    # Fallbacks & helpers
    # ------------------------------------------------------------------

    def _handle_fallback(self, _: str) -> str:
        return random.choice(FALLBACK_MESSAGES)

    def _build_menu_teaser(
        self,
        max_categories: int = 2,
        max_items_per_category: int = 3,
    ) -> str:
        """
        Build a short, safe 'menu teaser' used in capability replies.

        Example output:
        Here are a few examples from the menu you can ask about:
        - Beverage: Latte, Cappuccino, Americano (black)
        - Food: Blueberry Muffin, All Butter Belgian Dark Chocolate Chunk Cookie
        """
        # Filter to active items only, to avoid showing anything we’ve deactivated
        active_items = [
            m for m in self.menu_items
            if getattr(m, "is_active", True)
        ]
        if not active_items:
            return ""

        # Group by category (e.g. Beverage, Food, etc.)
        by_cat = {}
        for item in active_items:
            cat = (getattr(item, "category", None) or "Other").strip() or "Other"
            by_cat.setdefault(cat, []).append(item)

        if not by_cat:
            return ""

        lines: List[str] = ["Here are a few examples from the menu you can ask about:"]
        category_names = sorted(by_cat.keys())

        for cat in category_names[:max_categories]:
            # Stable, predictable ordering for examples
            examples = sorted(by_cat[cat], key=lambda m: m.name or "")[:max_items_per_category]
            names = ", ".join(m.name for m in examples if m.name)
            if names:
                lines.append(f"- {cat}: {names}")

        # If we didn’t manage to build any item lines, just skip the teaser
        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    def _tokenise_for_matching(self, s: str) -> List[str]:
        """
        Tokeniser used for fuzzy menu matching:
        - lowercases
        - normalises a few synonyms (e.g. 'choc' -> 'chocolate')
        - drops MATCH_STOPWORDS
        """
        TOKEN_NORMALISATIONS = {
            "choc": "chocolate",  # so "hot choc" matches "Hot Chocolate"
            "soy": "soya",
            "mist": "misto",
            "grand":"grande",
            "charamel":"caramel",
        }

        tokens: List[str] = []
        for raw in re.findall(r"[a-z0-9]+", (s or "").lower()):
            norm = TOKEN_NORMALISATIONS.get(raw, raw)
            if norm and norm not in self.MATCH_STOPWORDS:
                tokens.append(norm)
        return tokens

    def _extract_quantity(self, text: str) -> int:
        t = (text or "").lower().strip()

        # common forms: "x 2", "x2", "2x", "2 x", "× 2"
        m = re.search(r"\b(?:x|×)\s*(\d+)\b", t)
        if m:
            return max(1, int(m.group(1)))

        m = re.search(r"\b(\d+)\s*(?:x|×)\b", t)
        if m:
            return max(1, int(m.group(1)))

        # "2 times", "times 2"
        m = re.search(r"\b(\d+)\s*(?:times|time)\b", t)
        if m:
            return max(1, int(m.group(1)))

        m = re.search(r"\b(?:times|time)\s*(\d+)\b", t)
        if m:
            return max(1, int(m.group(1)))

        # words
        if re.search(r"\b(twice|double)\b", t):
            return 2
        if re.search(r"\b(triple)\b", t):
            return 3

        return 1

    def _extract_item_quantity(self, user_text: str, item: Optional[MenuItem]) -> int:
        """
        Extract quantity for the *base menu item* without accidentally treating
        option multipliers like 'vanilla syrup x 2' as item quantity.
        """
        text = (user_text or "")

        # If the user uses '+', prefer the segment that actually contains the item
        parts = re.split(r"\s*\+\s*", text)
        seg = None
        if item is not None:
            for p in parts:
                p_item = self._find_menu_item_in_text_relaxed(p)
                if p_item and p_item.item_id == item.item_id:
                    seg = p
                    break

        seg = seg or (parts[0] if parts else text)

        # Remove option-level multipliers before extracting item quantity
        cleaned = re.sub(
            r"\b(?:syrup|shot|shots|topping|toppings|sweetener|sweeteners|pump|pumps)\s*(?:x|×)\s*\d+\b",
            "",
            seg,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b\d+\s*(?:x|×)\s*(?:syrup|shot|shots|topping|toppings|sweetener|sweeteners|pump|pumps)\b",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        return self._extract_quantity(cleaned)

    def _build_menu_vocab(self, items: Sequence[MenuItem]) -> set:
        """
        Build a vocabulary of tokens that appear in any menu item
        (brand + name), used to spot when a question looks item-specific.
        """
        vocab: set = set()
        for item in items:
            candidate_text = f"{getattr(item, 'brand', '')} {getattr(item, 'name', '')}"
            for tok in self._tokenise_for_matching(candidate_text):
                vocab.add(tok)
        return vocab

    def _build_option_vocab(self, options: Sequence[MenuOption]) -> set[str]:
        vocab: set[str] = set()
        for opt in options:
            candidate = f"{opt.option_group or ''} {opt.option_value or ''}"
            for tok in re.findall(r"[a-z0-9]+", candidate.lower()):
                vocab.add(tok)
        vocab.update({"milk", "syrup", "shot", "shots", "cream", "whipped", "vanilla", "caramel",
                      "oat", "soya", "soy", "almond", "coconut", "grande", "tall", "venti"})
        return vocab

    def _handle_pending_options_item(self, raw: str) -> str:
        lowered = raw.lower()

        if lowered in {"cancel", "stop", "exit", "nevermind", "never mind"}:
            self._push_undo("cancel customisations")
            self.state.awaiting_item_for_pending_options = False
            self.state.pending_options_queue.clear()
            return "Okay — I won’t apply those customisations. Which drink or food would you like?"

        # If they send MORE options while we're still waiting for the item, queue them too
        item = self._find_menu_item_in_text(raw) or self._find_menu_item_in_text_relaxed(raw)
        if item is not None:
            quantity = self._extract_item_quantity(raw, item)  # helper you’ll add below

            queued_opts = " + ".join(self.state.pending_options_queue)

            raw_tokens = set(self._tokenise_for_matching(raw))
            has_option_tokens = any(t in getattr(self, "_option_vocab", set()) for t in raw_tokens)
            combined_options_text = " + ".join([x for x in [queued_opts, (raw if has_option_tokens else "")] if x])

            self.state.awaiting_item_for_pending_options = False
            self.state.pending_options_queue.clear()

            log_text = f"{item.name} + {queued_opts}".strip(" +")
            return self._add_specific_item_to_cart(
                item=item,
                options_text=combined_options_text,
                log_text=log_text,
                quantity=quantity,
            )

        # 2) Otherwise: if it’s more options, queue them
        if self._looks_like_options_only(raw):
            self.state.pending_options_queue.append(raw)
            return "Got it — I’ve noted those customisations too. Which drink or food should I apply them to?"

        # 3) Otherwise: unknown
        return (
            "I couldn’t find that item in the menu. "
            "Which drink or food should I apply those customisations to? "
            "For example: “Latte”, “Mocha”, or “Blueberry Muffin”."
        )

    def _parse_options_from_text(
            self,
            user_text: str,
            base_item: Optional[MenuItem] = None,
    ) -> List[tuple[MenuOption, int]]:
        """
        Lightweight option parser.

        - Looks for size, milk, syrups, toppings, sweeteners and extra shots
          mentioned in the utterance.
        - Returns a list of (MenuOption, quantity) pairs.

        Behaviour:
        - Syrups:
            * If user says "sugar free ..." or "free sugar ..." we try to use
              sugar_free_* syrups.
            * If there is no sugar_free variant for that drink/flavour, we
              gracefully fall back to the normal syrup.
        - We also collapse duplicates so you get at most ONE option per
          (option_group, flavour core), e.g. only one "vanilla syrup" even
          though options.csv has Starbucks + Costa variants.

        Important detail:
        - We parse options per '+' segment so "vanilla syrup x 2 + extra shot"
          gives vanilla syrup × 2 and extra shot × 1 (not 2).
        """
        # If you haven't wired this yet, DialogueManager.__init__ should include:
        #   self.menu_options = load_option_items()
        if not getattr(self, "menu_options", None):
            return []

        text = (user_text or "").lower()
        if not text.strip():
            return []

        # Detect "sugar free" and "free sugar" variants
        sugar_free_requested = bool(
            re.search(r"\b(sugar[- ]?free|free[- ]?sugar)\b", text)
        )

        # Split text on '+' so quantity like "x 2" only applies inside its segment
        segments = [seg.strip() for seg in re.split(r"\s*\+\s*", text) if seg.strip()]

        NUMBER_WORDS: dict[str, int] = {
            "one": 1, "1": 1,
            "two": 2, "2": 2,
            "three": 3, "3": 3,
            "double": 2,
            "triple": 3,
        }

        def applies_to_item(opt: MenuOption) -> bool:
            """
            Check options.csv `applies_to_scope` (item_id:... | category:...).
            If scope is blank, treat as global.

            Special case:
            - 'size' options are treated as generic (short / tall / grande / venti / regular / large)
              and allowed for any Beverage item. We rely on the text tokens to decide WHICH size
              is actually used.
            """
            if base_item is None:
                # For standalone segments (e.g. pure "vanilla syrup x 2"), treat options as global.
                return True

            item_id = getattr(base_item, "item_id", "")
            category = (getattr(base_item, "category", "") or "").lower()
            group = (opt.option_group or "").lower()

            # Sizes: allow for any beverage, regardless of the CSV scope
            if group == "size" and category == "beverage":
                return True

            if group == "milk" and category == "beverage":
                return True

            scope = (opt.applies_to_scope or "").strip()
            if not scope:
                return True

            for clause in scope.split("|"):
                clause = clause.strip()
                if not clause:
                    continue
                if clause.startswith("item_id:"):
                    if item_id == clause[len("item_id:"):]:
                        return True
                elif clause.startswith("category:"):
                    if category == clause[len("category:"):].lower():
                        return True

            return False

        # For each (group, core) we keep at most:
        #   - one "normal" option (non-sugar-free)
        #   - one "sugar_free" option
        # and aggregate quantities per type.
        bucket: dict[
            tuple[str, str],
            dict[str, tuple[MenuOption, int]],
        ] = {}

        for seg in segments:
            seg_tokens = self._tokenise_for_matching(seg)
            if not seg_tokens:
                continue

            # Segment-level quantity (e.g. "vanilla syrup x 2")
            seg_qty = 1

            # Pattern 1: "... x 2"
            m = re.search(r"(?:x|×|\*)\s*(\d+)", seg)
            if not m:
                # Pattern 2: "2x" or "2 x"
                m = re.search(r"\b(\d+)\s*(?:x|×|\*)", seg)
            if m:
                try:
                    seg_qty = max(int(m.group(1)), 1)
                except ValueError:
                    seg_qty = 1
            else:
                # Words like "double shot", "triple vanilla"
                for word, n in (("triple", 3), ("double", 2), ("three", 3), ("two", 2)):
                    if re.search(rf"\b{word}\b", seg):
                        seg_qty = n
                        break

            for opt in self.menu_options:
                if not applies_to_item(opt):
                    continue

                group = (opt.option_group or "").lower()
                value = (opt.option_value or "").lower()

                # Normalise the option value to tokens
                base_value = value.replace("_flavour", "").replace("_", " ")
                val_tokens = [
                    t for t in re.findall(r"[a-z0-9]+", base_value) if t
                ]
                if not val_tokens:
                    continue

                # We use the last token as the "core" flavour (vanilla, caramel, oat, shot, …)
                core = val_tokens[-1]

                aliases = set(val_tokens)
                # Extra synonyms
                if core == "soya":
                    aliases.add("soy")
                if group == "size":
                    aliases.add(base_value)  # e.g. "venti hot" as a phrase

                # Check if this option is even mentioned in this segment
                if not any(a in seg_tokens for a in aliases):
                    continue

                # Decide whether this is treated as "sugar_free" or "normal"
                is_sugar_free_opt = ("sugar_free" in value) or ("sugar-free" in value)
                sf_key = "sugar_free" if (group == "syrup" and is_sugar_free_opt) else "normal"

                # Which groups should actually respect the segment quantity?
                if group in {"syrup", "extra_shot", "topping", "sweetener"}:
                    qty = seg_qty
                else:
                    # size / milk / some "other" adjustments are not multiplicative
                    qty = 1

                qty = max(qty, 1)

                key = (group, core)
                per_type = bucket.setdefault(key, {})

                existing = per_type.get(sf_key)
                if existing is None:
                    per_type[sf_key] = (opt, qty)
                else:
                    prev_opt, prev_qty = existing
                    per_type[sf_key] = (prev_opt, max(prev_qty, qty))

        # Turn the bucket into a flat list, applying sugar_free preference / fallback
        results: List[tuple[MenuOption, int]] = []

        for (group, core), per_type in bucket.items():
            normal = per_type.get("normal")  # (MenuOption, qty) or None
            sugar_free = per_type.get("sugar_free")

            chosen: Optional[tuple[MenuOption, int]] = None

            if group == "syrup":
                if sugar_free_requested and sugar_free is not None:
                    # User asked for sugar-free and we have it
                    chosen = sugar_free
                elif sugar_free_requested and sugar_free is None and normal is not None:
                    # User asked for sugar-free but we only have the normal syrup
                    chosen = normal

                    opt_obj, _opt_qty = normal
                    try:
                        label = self._format_option_label(opt_obj, 1)
                        note = (
                            "You asked for a sugar-free syrup, but I don’t have a sugar-free "
                            f"version of **{label}** for this drink, so I’ve added the regular syrup instead."
                        )
                    except Exception:
                        note = (
                            "You asked for a sugar-free syrup, but I don’t have a sugar-free "
                            "version for this drink, so I’ve added the regular syrup instead."
                        )
                    self._sugar_free_notes.append(note)
                else:
                    # No sugar-free requested: prefer normal; fall back to sugar-free if needed
                    chosen = normal or sugar_free
            else:
                # Non-syrup groups (size, milk, toppings, sweeteners, extra shots, other):
                # always prefer "normal" over "sugar_free" variant if both exist.
                chosen = normal or sugar_free

            if chosen is not None:
                results.append(chosen)

        # Ensure at most one milk option is applied: keep the last explicitly mentioned one
        milk_indices = [
            i for i, (opt, _qty) in enumerate(results)
            if (opt.option_group or "").lower() == "milk"
        ]
        if len(milk_indices) > 1:
            last_idx = milk_indices[-1]
            chosen_opt, chosen_qty = results[last_idx]
            # Drop all milks, then re-add the last one mentioned
            results = [
                pair for pair in results
                if (pair[0].option_group or "").lower() != "milk"
            ]
            results.append((chosen_opt, chosen_qty))

        return results

    def _format_option_label(self, opt: MenuOption, qty: int = 1) -> str:
        """
        Turn a MenuOption into a short label, optionally with a multiplier.

        Special cases:
        - Syrups: nicely label sugar-free variants as
          "sugar-free vanilla syrup", "sugar-free caramel syrup", etc.
        - Milks: ensure the word 'milk' appears.
        - Extra shots: force "extra espresso shot".
        """
        value = (opt.option_value or "").strip() or opt.option_id
        group = (opt.option_group or "").lower()

        # Tokenise the raw value to reason about flavour / sugar-free status
        tokens_raw = re.findall(r"[a-z0-9]+", value.lower())
        # Drop generic "flavour" token
        tokens_no_flavour = [t for t in tokens_raw if t != "flavour"]

        # Detect sugar-free variants from either the raw value or tokens
        is_sugar_free = (
                "sugar_free" in value.lower()
                or "sugar-free" in value.lower()
                or ("sugar" in tokens_no_flavour and "free" in tokens_no_flavour)
        )

        if group == "syrup":
            # e.g. sugar_free_vanilla_flavour, caramel_syrup, etc.
            core_tokens = [
                t for t in tokens_no_flavour
                if t not in {"sugar", "free", "syrup"}
            ]
            flavour = core_tokens[-1] if core_tokens else "syrup"

            if is_sugar_free:
                base = f"sugar-free {flavour} syrup"
            else:
                base = f"{flavour} syrup"
        else:
            # Generic base label for non-syrup options
            if tokens_no_flavour:
                base = " ".join(tokens_no_flavour)
            else:
                base = value.replace("_", " ")

            if group == "milk":
                # Ensure we end up with "oat milk", "soya milk", etc.
                if "milk" not in base:
                    base = f"{base} milk"
            elif group == "extra_shot":
                base = "extra espresso shot"
            elif group == "size":
                # Keep specific nicer label for venti_hot if used
                if value == "venti_hot":
                    base = "venti (hot)"

        if qty > 1:
            return f"{base} × {qty}"
        return base

    def _format_options_summary(self, option_ids: List[str]) -> str:
        """
        Summarise a list of option_ids as a readable string, e.g.
        'oat milk, vanilla syrup × 2'.
        """
        if not option_ids:
            return ""

        counts = Counter(option_ids)
        parts: List[str] = []
        for opt_id, qty in counts.items():
            opt = self.options_by_id.get(opt_id)
            if not opt:
                continue
            label = self._format_option_label(opt, qty)
            if label:
                parts.append(label)

        return ", ".join(parts)

    def _estimate_line_calories(self, line: CartLine) -> float:
        """
        Estimate total calories for a cart line (quantity × per-drink calories).

        - MenuItem.calories_kcal is treated as the base drink/food calories.
        - Each option in line.selected_options adds MenuOption.calories_delta_kcal.

        This is a simple additive model. If you later want 'replacement milks'
        (subtract default dairy, add plant milk), adjust the logic here.
        """
        # Find the base menu item
        base_item: Optional[MenuItem] = None
        for item in self.menu_items:
            if item.item_id == line.item_id:
                base_item = item
                break

        base_kcal = (getattr(base_item, "calories_kcal", 0.0) or 0.0)
        per_unit = base_kcal

        # Add all selected options (duplicates mean multiples, e.g. vanilla × 2)
        for opt_id in getattr(line, "selected_options", []) or []:
            opt = self.options_by_id.get(opt_id)
            if not opt:
                continue
            delta = getattr(opt, "calories_delta_kcal", 0.0) or 0.0
            per_unit += delta

        return per_unit * max(line.quantity, 1)

    def _estimate_line_price(self, line: CartLine) -> float:
        """
        Estimate total price for a cart line.

        - MenuItem.base_price is the default Tall/Regular (or single) price.
        - Each option in line.selected_options adds MenuOption.price_delta.
        """
        # Find the base menu item
        base_item: Optional[MenuItem] = None
        for item in self.menu_items:
            if item.item_id == line.item_id:
                base_item = item
                break

        base_price = (getattr(base_item, "base_price_gbp", 0.0) or 0.0)
        per_unit = base_price

        # Add all selected options (duplicates mean multiples, e.g. vanilla × 2)
        for opt_id in getattr(line, "selected_options", []) or []:
            opt = self.options_by_id.get(opt_id)
            if not opt:
                continue
            delta = getattr(opt, "price_delta", 0.0) or 0.0
            per_unit += delta

        return per_unit * max(line.quantity, 1)

    def _looks_like_calorie_expression(self, user_text: str) -> bool:
        """
        Heuristic: does this text look like a 'calorie math' expression?

        We treat something as a calorie expression if:
        - it contains arithmetic operators like +, -, x, ×, * or /
        - it mentions at least one menu-like token
        - it does NOT look like a straightforward order command
          (no "add", "order", "buy", etc.)
        """
        lowered = (user_text or "").lower()

        # Must contain at least one arithmetic symbol
        if not re.search(r"[+\-*/×x]", lowered):
            return False

        # Avoid normal ordering phrases – those should stay ADD_TO_CART
        ORDER_WORDS = {
            "add", "order", "buy", "get", "put", "bring", "take",
            "have", "want", "like", "give", "takeaway",
        }
        if any(re.search(rf"\b{w}\b", lowered) for w in ORDER_WORDS):
            return False

        # At least one token must look like it comes from the menu vocabulary
        tokens = set(self._tokenise_for_matching(lowered))
        if not tokens:
            return False

        return any(t in self._menu_vocab for t in tokens)

    def _answer_calorie_simple(self, user_text: str) -> str:
        """
        Handle normal questions like:
        - "how many calories in a latte?"
        - "calories in large latte with vanilla syrup?"
        - "cals in grande flat white with oat milk"
        """
        item = self._find_menu_item_in_text_relaxed(user_text)
        if item is None:
            # We couldn't confidently match any specific menu item
            faq = self.faq_engine.get_faq_by_id("calorie_math")
            if faq:
                return (
                        "Tell me the drink or food you’re asking about "
                        "(for example, “Latte” or “Blueberry Muffin”), "
                        "and I’ll look up its calories.\n\n"
                        + faq.answer_markdown
                )
            return (
                "Tell me the drink or food you’re asking about "
                "(for example, “Latte” or “Blueberry Muffin”), "
                "and I’ll look up its calories."
            )

        base_kcal = (getattr(item, "calories_kcal", 0.0) or 0.0)

        # NEW: parse options (size, milk, syrups, shots, toppings, sweeteners)
        opt_pairs = self._parse_options_from_text(user_text, base_item=item)
        selected_option_ids: List[str] = []
        for opt, qty in opt_pairs:
            for _ in range(max(qty, 1)):
                selected_option_ids.append(opt.option_id)

        if selected_option_ids:
            # Reuse the same logic as the cart so behaviour is consistent
            temp_line = CartLine(
                item_id=item.item_id,
                item_name=item.name,
                quantity=1,
                selected_options=selected_option_ids,
            )
            kcal = self._estimate_line_calories(temp_line)
            options_summary = self._format_options_summary(selected_option_ids)
            label = f"{item.name} ({options_summary})"
        else:
            kcal = base_kcal
            label = item.name

        if kcal <= 0.0:
            faq = self.faq_engine.get_faq_by_id("calorie_math")
            extra = "\n\n" + faq.answer_markdown if faq else ""
            return (
                    f"I couldn’t find calorie data for **{label}** in my menu yet."
                    + extra
            )

        # Sugar-free fallback explanation, if any, same as in cart / expression paths
        sugar_notes = self._consume_sugar_free_notes()
        extra_notes = "\n\n" + "\n".join(sugar_notes) if sugar_notes else ""

        if selected_option_ids:
            main = (
                f"Based on our menu data, **{label}** has about **{kcal:.0f} kcal** "
                f"for this configuration."
            )
        else:
            main = (
                f"Based on our menu data, **{label}** has about **{kcal:.0f} kcal** "
                "per serving before any extra syrups, toppings or size changes."
            )

        return main + extra_notes + "\n\n" + CALORIE_DISCLAIMER

    def _answer_calorie_expression(self, user_text: str) -> str:
        """
        Handle expressions like:
        - "latte + muffin"
        - "latte x 2 + blueberry muffin"
        - "grande latte + vanilla syrup x 2"

        We now:
        - Treat each '+' segment as a mini "order line".
        - For segments that contain a menu item, we attach any sizes / milks /
          syrups / shots mentioned in *that* segment and use _estimate_line_calories.
        - For segments that only contain options (e.g. "vanilla syrup x 2"),
          we treat them as extra standalone kcal on top.
        """
        text = (user_text or "").strip()
        if not text:
            return self._answer_calorie_simple(user_text)

        parts = re.split(r"\s*\+\s*", text)

        # Per-item segments: [{item, qty, option_ids, kcal_total}, ...]
        item_segments: List[dict] = []
        # Standalone options (no associated item in that segment)
        extra_options_agg: dict[str, tuple[MenuOption, int]] = {}
        unknown_segments: List[str] = []

        for raw_part in parts:
            expr_seg = raw_part.strip()
            if not expr_seg:
                continue

            # Expression-level quantity for the whole segment, e.g. "2x latte"
            qty = 1
            segment_for_item = expr_seg

            # Patterns: "latte x 2", "latte × 2", "latte * 2", "latte times 2"
            m = re.search(r"(?:x|×|\*|\btimes\b)\s*(\d+)", expr_seg, flags=re.IGNORECASE)
            if not m:
                # Also support "2x latte", "2 x latte"
                m = re.search(r"(\d+)\s*(?:x|×|\*)", expr_seg, flags=re.IGNORECASE)

            if m:
                try:
                    qty = max(int(m.group(1)), 1)
                except ValueError:
                    qty = 1

                # Strip the quantity marker from the *item-matching* text,
                # but keep the full expr_seg (with "x 2") for option parsing.
                segment_for_item = re.sub(
                    r"(?:x|×|\*|\btimes\b)\s*\d+|\d+\s*(?:x|×|\*)",
                    "",
                    expr_seg,
                    flags=re.IGNORECASE,
                ).strip()

            # Try to identify a menu item in this segment
            item = self._find_menu_item_in_text_relaxed(segment_for_item)
            base_kcal = (getattr(item, "calories_kcal", 0.0) or 0.0) if item else 0.0

            if item is not None and base_kcal > 0.0:
                # Options tied to this item – parse from the full segment so
                # things like "vanilla syrup x 2" are visible to the parser.
                opt_pairs = self._parse_options_from_text(expr_seg, base_item=item)
                selected_option_ids: List[str] = []
                for opt, opt_qty in opt_pairs:
                    for _ in range(max(opt_qty, 1)):
                        selected_option_ids.append(opt.option_id)

                temp_line = CartLine(
                    item_id=item.item_id,
                    item_name=item.name,
                    quantity=max(qty, 1),
                    selected_options=selected_option_ids,
                )
                seg_kcal = self._estimate_line_calories(temp_line)

                item_segments.append(
                    {
                        "item": item,
                        "qty": max(qty, 1),
                        "option_ids": selected_option_ids,
                        "kcal_total": seg_kcal,
                    }
                )
            else:
                # No clear menu item → maybe this is "vanilla syrup x 2" etc.
                opt_pairs = self._parse_options_from_text(expr_seg, base_item=None)
                if opt_pairs:
                    # Aggregate standalone options so "vanilla syrup x 2" +
                    # "vanilla syrup" becomes qty=3.
                    for opt, opt_qty in opt_pairs:
                        # Multiply by the segment quantity if present
                        add_qty = max(opt_qty, 1)
                        current = extra_options_agg.get(opt.option_id)
                        if current is None:
                            extra_options_agg[opt.option_id] = (opt, add_qty)
                        else:
                            _, prev_qty = current
                            extra_options_agg[opt.option_id] = (opt, prev_qty + add_qty)
                elif segment_for_item:
                    # Something we couldn't match at all
                    unknown_segments.append(segment_for_item)

        # If nothing intelligible, fall back to single-item handling
        if not item_segments and not extra_options_agg:
            return self._answer_calorie_simple(user_text)

        lines: List[str] = ["Here’s an approximate calorie breakdown:"]
        total_kcal: float = 0.0

        # --- Item segments (drinks/foods, possibly with options) ---
        for seg in item_segments:
            item = seg["item"]
            qty = seg["qty"]
            option_ids = seg["option_ids"]
            seg_kcal = seg["kcal_total"]
            total_kcal += seg_kcal

            options_summary = self._format_options_summary(option_ids)
            if options_summary:
                label = f"{item.name} ({options_summary})"
            else:
                label = item.name

            if qty <= 1:
                lines.append(f"- {label}: about {seg_kcal:.0f} kcal.")
            else:
                per_unit = seg_kcal / qty if qty > 0 else seg_kcal
                lines.append(
                    f"- {label} × {qty}: about {per_unit:.0f} kcal each = {seg_kcal:.0f} kcal."
                )

        # --- Standalone extras (options-only segments) ---
        if extra_options_agg:
            lines.append("")
            lines.append("Extra options not tied to a specific drink/food:")
            for opt_id, (opt, qty) in extra_options_agg.items():
                delta = getattr(opt, "calories_delta_kcal", 0.0) or 0.0
                label = self._format_option_label(opt, qty)
                if qty <= 1:
                    lines.append(f"- {label}: adds ~{delta:.0f} kcal.")
                    total_kcal += delta
                else:
                    lines.append(
                        f"- {label}: about {delta:.0f} kcal each = {delta * qty:.0f} kcal."
                    )
                    total_kcal += delta * qty

        # --- Unknown bits (we don't pretend to know them) ---
        if unknown_segments:
            unique_unknown = sorted({seg for seg in unknown_segments})
            lines.append("")
            lines.append(
                "I couldn’t match these parts to specific menu items or options yet, "
                "so they’re not included in the total: "
                + ", ".join(f"`{seg}`" for seg in unique_unknown)
                + "."
            )

        # Sugar-free fallback explanations (if any were triggered)
        sugar_notes = self._consume_sugar_free_notes()
        if sugar_notes:
            lines.append("")
            lines.extend(sugar_notes)

        # Final total summary
        lines.append("")
        lines.append(
            f"Estimated total for everything you described: **~{total_kcal:.0f} kcal**."
        )
        lines.append("")
        lines.append(CALORIE_DISCLAIMER)

        return "\n".join(lines)

    def _looks_like_item_query(self, user_text: str) -> bool:
        """
        Heuristic: does this text mention any token that exists in the menu?
        Used to decide whether a gluten-free question is about a *specific*
        item vs general GF advice.
        """
        tokens = set(self._tokenise_for_matching(user_text or ""))
        if not tokens:
            return False
        return any(t in self._menu_vocab for t in tokens)

    def _looks_like_options_only(self, text: str) -> bool:
        toks = set(self._tokenise_for_matching(text or ""))

        # ignore connectors
        toks -= {"and", "or", "plus"}

        # ignore quantity markers and numbers
        qty_noise = {"x", "times", "time", "double", "triple", "twice"}
        toks = {t for t in toks if (t not in qty_noise and not t.isdigit())}

        if not toks:
            return False

        # If it's a single token, only treat it as options-only if it's a strong milk-variant cue.
        # This prevents things like "mocha" being treated as a syrup option when the user likely means the drink.
        if len(toks) == 1:
            return next(iter(toks)) in {"oat", "almond", "soya", "soy", "coconut"}

        option_vocab = getattr(self, "_option_vocab", set())
        return all(t in option_vocab for t in toks)

    @staticmethod
    def _base_item_name(name: str) -> str:
        """
        Normalise an item name to a base form, e.g.
        'Misto - oat drink' -> 'misto'
        'Flat White - almond drink' -> 'flat white'
        """
        if not name:
            return ""
        return name.split(" - ")[0].strip().lower()


    def _find_menu_item_in_text(self, user_text: str) -> Optional[MenuItem]:
        """
        Fuzzy name matching:

        - Tokenise both user text and item names (letters/digits only),
          with MATCH_STOPWORDS removed and simple synonym normalisation
          (e.g. "choc" -> "chocolate").
        - Only keep user tokens that appear somewhere in the menu vocab.
        - Compute token overlap between user and each item (brand + name).
        - Pick the item(s) with the highest overlap, then break ties with
          some safety-aware heuristics.

        Safety rules:
        - If there is only a 1-token overlap and the question also contains
          extra tokens that never appear in the menu (e.g. "unicorn latte"),
          we treat it as an unknown drink (return None).
        - We only allow a 1-token match when *all* user tokens are "menu-ish".
        """
        text = (user_text or "").lower()
        if not text:
            return None

        user_tokens_all = set(self._tokenise_for_matching(text))
        if not user_tokens_all:
            return None

        # Only keep user tokens that appear somewhere in the menu
        recognised_tokens = {t for t in user_tokens_all if t in self._menu_vocab}
        if not recognised_tokens:
            return None

        # Find all items with the maximum overlap
        best_overlap = 0
        candidates: List[tuple[MenuItem, set[str]]] = []

        for item in self.menu_items:
            name = getattr(item, "name", "")
            if not name:
                continue

            candidate_text = f"{getattr(item, 'brand', '')} {name}"
            item_tokens = set(self._tokenise_for_matching(candidate_text))
            if not item_tokens:
                continue

            overlap = len(recognised_tokens & item_tokens)
            if overlap == 0:
                continue

            if overlap > best_overlap:
                best_overlap = overlap
                candidates = [(item, item_tokens)]
            elif overlap == best_overlap:
                candidates.append((item, item_tokens))

        if best_overlap == 0 or not candidates:
            return None

        # If we only have 1-token overlap but there are extra unknown tokens
        # (e.g. "unicorn latte"), treat as suspicious and refuse to match.
        option_tokens = getattr(self, "_option_vocab", set())
        unknown_tokens = user_tokens_all - recognised_tokens - option_tokens

        if best_overlap == 1 and unknown_tokens:
            return None

        # At this point we have one or more "best" candidates that are
        # safe to consider. Use tie-breaking to pick the best one.
        return self._break_tie(user_tokens_all, recognised_tokens, candidates)

    def _find_menu_item_in_text_relaxed(self, user_text: str) -> Optional[MenuItem]:
        """
        A more permissive variant of _find_menu_item_in_text used for calorie
        questions, where we can afford to be less strict than allergen logic.

        Differences from the strict version:
        - We DO NOT block 1-token matches just because there are extra
          unknown tokens (e.g. "grande", "vanilla", "unicorn").
        - Still uses the same tokenisation and tie-breaking logic.
        """
        text = (user_text or "").lower()
        if not text:
            return None

        user_tokens_all = set(self._tokenise_for_matching(text))
        if not user_tokens_all:
            return None

        recognised_tokens = {t for t in user_tokens_all if t in self._menu_vocab}
        if not recognised_tokens:
            return None

        best_overlap = 0
        candidates: List[tuple[MenuItem, set[str]]] = []

        for item in self.menu_items:
            name = getattr(item, "name", "")
            if not name:
                continue

            candidate_text = f"{getattr(item, 'brand', '')} {name}"
            item_tokens = set(self._tokenise_for_matching(candidate_text))
            if not item_tokens:
                continue

            overlap = len(recognised_tokens & item_tokens)
            if overlap == 0:
                continue

            if overlap > best_overlap:
                best_overlap = overlap
                candidates = [(item, item_tokens)]
            elif overlap == best_overlap:
                candidates.append((item, item_tokens))

        if best_overlap == 0 or not candidates:
            return None

        # Reuse the existing tie-breaking heuristics
        return self._break_tie(user_tokens_all, recognised_tokens, candidates)

    def _looks_like_show_menu_request(self, lowered: str) -> bool:
        t = (lowered or "").strip()
        if not t:
            return False

        # Very direct requests
        if t in {
            "menu", "show menu", "show me the menu", "view menu", "list menu",
            "drink menu", "drinks menu", "food menu",
            "starbucks menu", "costa menu", "costa express menu",
        }:
            return True

        # Common phrasings
        patterns = [
            r"\b(show|see|view|display|list)\b.*\bmenu\b",
            r"\bwhat('s| is)\b.*\bon\b.*\bmenu\b",
            r"\bwhat do you have\b",
            r"\bwhat drinks do you have\b",
            r"\bwhat food do you have\b",
            r"\b(starbucks|costa)\b.*\bmenu\b",
        ]
        return any(re.search(p, t) for p in patterns)

    def _handle_show_menu(self, user_text: str) -> str:
        lowered = (user_text or "").lower()

        # ---- filters (lightweight, no extra intents needed) ----
        category = None
        if any(w in lowered for w in ["drink", "drinks", "coffee", "beverage"]):
            category = "beverage"
        if any(w in lowered for w in ["food", "snack", "snacks", "pastry", "sandwich"]):
            category = "food"

        vegan_only = "vegan" in lowered
        vegetarian_only = ("vegetarian" in lowered) and not vegan_only
        halal_only = "halal" in lowered
        gluten_free_only = ("gluten free" in lowered) or ("gluten-free" in lowered)

        show_all = bool(re.search(r"\b(all|full|everything|complete)\b", lowered))

        # ---- get active items ----
        items = [m for m in self.menu_items if getattr(m, "is_active", True)]
        if category:
            items = [m for m in items if (getattr(m, "category", "") or "").lower() == category]

        if vegan_only:
            items = [m for m in items if bool(getattr(m, "suitable_vegan", False))]
        elif vegetarian_only:
            items = [m for m in items if bool(getattr(m, "suitable_vegetarian", False))]

        if halal_only:
            # use your existing halal classifier
            items = [m for m in items if
                     self._get_halal_flags(m).get("halal_friendly") or self._get_halal_flags(m).get("vegan_hint")]

        if gluten_free_only:
            # best-effort: treat any of these as gluten signals if present in your CSV
            GLUTEN_KEYS = {"contains_gluten", "contains_wheat", "contains_barley", "contains_rye", "contains_oats"}

            def looks_gf(mi: MenuItem) -> bool:
                allergens = getattr(mi, "allergens", {}) or {}
                return not any(allergens.get(k, False) for k in GLUTEN_KEYS)

            items = [m for m in items if looks_gf(m)]

        if not items:
            return "I couldn’t find any matching items on the menu with those filters."

        brand = None
        if "starbucks" in lowered:
            brand = "starbucks"
        elif "costa" in lowered:
            brand = "costa"

        if brand:
            items = [
                m for m in items
                if brand in (getattr(m, "brand", "") or "").lower()
            ]

        # ---- group & format ----
        items = sorted(items, key=lambda m: ((getattr(m, "category", "") or ""), (getattr(m, "name", "") or "")))

        MAX_TOTAL = 25
        shown = items if show_all else items[:MAX_TOTAL]
        remaining = max(0, len(items) - len(shown))

        # group by category for readability
        by_cat = {}
        for it in shown:
            cat = (getattr(it, "category", None) or "Other").strip() or "Other"
            by_cat.setdefault(cat, []).append(it)

        lines = ["Here’s the menu I have in my data right now:"]
        for cat in sorted(by_cat.keys()):
            lines.append(f"\n**{cat}**")
            for it in by_cat[cat]:
                name = it.name
                price = getattr(it, "base_price_gbp", 0.0) or 0.0
                kcal = getattr(it, "calories_kcal", 0.0) or 0.0

                meta = []
                if price > 0:
                    meta.append(f"£{price:.2f}")
                if kcal > 0:
                    meta.append(f"{kcal:.0f} kcal")

                tag_bits = []
                if getattr(it, "suitable_vegan", False):
                    tag_bits.append("vegan")
                elif getattr(it, "suitable_vegetarian", False):
                    tag_bits.append("vegetarian")
                if any("halal" in (t.lower()) for t in (getattr(it, "diet_tags", []) or [])):
                    tag_bits.append("halal-friendly")

                suffix = ""
                if tag_bits:
                    suffix = " [" + ", ".join(tag_bits) + "]"

                if meta:
                    lines.append(f"- {name} — {' — '.join(meta)}{suffix}")
                else:
                    lines.append(f"- {name}{suffix}")

        if remaining > 0:
            lines.append(f"\n(Showing the first {len(shown)} items. Say **“show full menu”** to list everything.)")

        lines.append(
            "\nYou can also say things like: “show drink menu”, “show vegan menu”, or “add a latte with oat milk to my order”.")
        return "\n".join(lines)

    def _consume_sugar_free_notes(self) -> List[str]:
        """
        Return any sugar-free fallback notes accumulated during option parsing
        and clear them for the next turn (deduplicated, order-preserving).
        """
        raw = getattr(self, "_sugar_free_notes", [])
        # dedupe while preserving order
        seen = set()
        notes: List[str] = []
        for n in raw:
            if n not in seen:
                seen.add(n)
                notes.append(n)
        self._sugar_free_notes = []
        return notes


    def _break_tie(
        self,
        user_tokens_all: set[str],
        recognised_tokens: set[str],
        candidates: List[tuple[MenuItem, set[str]]],
    ) -> Optional[MenuItem]:
        """
        Tie-breaking heuristics for menu-item matching.

        Order of preference:
        1) If user mentions a specific variant (oat / almond / soya / coconut),
           prefer items whose name includes that token.
        2) If user mentions milk/dairy/lactose and there is a mix of dairy vs
           non-dairy variants, prefer the dairy build (contains_milk=True).
        3) Prefer the item whose tokens introduce the fewest extra non-brand
           tokens beyond what the user said.
        4) Fall back to the first candidate (stable).
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        # 1) Variant keywords (plant milks) – oat / almond / soya / coconut
        VARIANT_TOKENS = {"almond", "oat", "soya", "soy", "coconut"}
        mentioned_variants = VARIANT_TOKENS & user_tokens_all

        if mentioned_variants:
            filtered: List[tuple[MenuItem, set[str]]] = []
            for item, toks in candidates:
                if toks & mentioned_variants:
                    filtered.append((item, toks))
            if filtered:
                candidates = filtered

        if len(candidates) == 1:
            return candidates[0][0]

        # 2) Milk words – only apply *after* variant matching
        MILK_WORDS = {"milk", "dairy", "lactose"}
        if user_tokens_all & MILK_WORDS:
            dairy_candidates: List[tuple[MenuItem, set[str]]] = []
            for item, toks in candidates:
                allergens = getattr(item, "allergens", {}) or {}
                if allergens.get("contains_milk", False):
                    dairy_candidates.append((item, toks))
            if dairy_candidates:
                candidates = dairy_candidates

        if len(candidates) == 1:
            return candidates[0][0]

        # 3) Prefer items with minimal "extra" tokens (ignoring brands)
        BRAND_TOKENS = {"starbucks", "costa", "express"}

        if len(recognised_tokens) == 1:
            target = next(iter(recognised_tokens))
            exact = []
            for item, toks in candidates:
                no_brand = {t for t in toks if t not in BRAND_TOKENS}
                if no_brand == {target}:
                    exact.append((item, toks))
            if exact:
                candidates = exact

        if len(candidates) == 1:
            return candidates[0][0]

        def extra_token_count(item_tokens: set[str]) -> int:
            no_brand = {t for t in item_tokens if t not in BRAND_TOKENS}
            extra = no_brand - recognised_tokens
            return len(extra)

        min_extra = min(extra_token_count(toks) for _, toks in candidates)
        best = [
            (item, toks)
            for item, toks in candidates
            if extra_token_count(toks) == min_extra
        ]

        # 4) Stable fallback: just pick the first among the "best"
        return best[0][0]


    def _format_item_allergen_answer(self, item: MenuItem) -> str:
        """
        Turn contains_* and may_contain_* flags into a readable explanation.
        This assumes domain_models/data_loader created dictionaries like:
        - item.allergens: {"contains_milk": True, ...}
        - item.may_contain: {"may_contain_nuts": False, ...}
        Adjust field names if yours differ.
        """
        allergens_dict = getattr(item, "allergens", {}) or {}
        may_contain_dict = getattr(item, "may_contain", {}) or {}

        contains = [
            key.replace("contains_", "").upper()
            for key, present in allergens_dict.items()
            if present
        ]
        may = [
            key.replace("may_contain_", "").upper()
            for key, present in may_contain_dict.items()
            if present
        ]

        veg = getattr(item, "suitable_vegetarian", None)
        vegan = getattr(item, "suitable_vegan", None)

        lines = [f"Allergen info for **{item.name}**:"]

        if contains:
            lines.append(f"- Contains: {', '.join(contains)}.")
        else:
            lines.append("- No 14 declarable allergens by ingredients (in our data).")

        if may:
            lines.append(f"- May contain (cross-contact risk): {', '.join(may)}.")

        if veg is not None:
            lines.append(f"- Vegetarian: {'yes' if veg else 'no'}.")
        if vegan is not None:
            lines.append(f"- Vegan: {'yes' if vegan else 'no'}.")

        lines.append("Always double-check live allergen guides and packaging if you have a severe allergy.")
        return "\n".join(lines)

    def _answer_item_halal(self, item: MenuItem) -> str:
        """
        Item-specific halal answer built on top of the usual allergen output.
        """
        base_text = self._format_item_allergen_answer(item)
        flags = self._get_halal_flags(item)

        lines: List[str] = [base_text, ""]

        if flags["halal_friendly"]:
            # For things like Signature Grilled Cheese Toastie, Cheese & Marmite Mini Ciabatta, Cortado, etc.
            lines.append(
                "In our menu data this item is flagged as **halal-friendly** "
                "(no pork or alcohol ingredients listed in the official allergen/food guide we used)."
            )

        elif flags["non_halal"]:
            if flags["has_pork"]:
                reason = (
                    "because it includes pork-derived ingredients "
                    "(for example bacon / ham / sausage) in our source data."
                )
            elif flags["has_alcohol"]:
                reason = (
                    "because it includes ingredients that are typically considered "
                    "non-halal (for example alcohol) in our source data."
                )
            else:
                reason = "based on the ingredient notes in our menu data."

            lines.append(
                f"In our menu data this item is listed as **not halal-friendly**, {reason}"
            )

        elif flags["vegan_hint"]:
            # e.g. Costa Express Americano (black), vegan drinks/foods
            lines.append(
                "By ingredients in our menu data, this item is **vegan** (no meat or dairy). "
                "Many halal-conscious customers treat vegan items as safe choices, "
                "but this is not a formal halal certification."
            )

        else:
            lines.append(
                "We don’t have an explicit halal label for this item. "
                "Based on our data, it does not list pork or alcohol, "
                "but we can’t guarantee full halal compliance."
            )

        lines.append(
            "Please treat this as approximate ingredient guidance only – it is **not** a religious fatwa. "
            "For strict halal decisions, always check the latest official allergen/nutrition guides, "
            "product packaging, or ask staff / a trusted scholar."
        )

        return "\n".join(lines)


    def _get_halal_flags(self, item: MenuItem) -> dict:
        """
        Classify an item from our menu data into rough halal buckets.

        Returns a dict with:
        - halal_friendly: True if explicitly tagged as halal-friendly.
        - vegan_hint: True if vegan (by tags / suitable_vegan).
        - non_halal: True if we see pork/alcohol keywords and no halal-friendly tag.
        - has_pork_or_alcohol: True if notes mention pork/bacon/ham/sausage or alcohol terms.
        """
        tags = [t.lower() for t in getattr(item, "diet_tags", [])]
        notes_lower = (getattr(item, "notes", "") or "").lower()

        # 1) Explicit halal-friendly tag wins
        is_halal_friendly = any("halal" in t for t in tags)

        # 2) Vegan by tag or boolean flag
        is_vegan = any("vegan" in t for t in tags) or bool(
            getattr(item, "suitable_vegan", False)
        )

        pork_keywords = ("pork", "bacon", "ham", "sausage")
        alcohol_keywords = ("alcohol", "wine", "beer", "cider", "rum", "brandy", "liqueur")

        has_pork = any(k in notes_lower for k in pork_keywords)
        has_alcohol = any(k in notes_lower for k in alcohol_keywords)
        has_pork_or_alcohol = has_pork or has_alcohol

        # Only treat as non-halal if we see pork/alcohol AND there is no explicit halal-friendly tag
        is_non_halal = has_pork_or_alcohol and not is_halal_friendly

        return {
            "halal_friendly": is_halal_friendly,
            "vegan_hint": is_vegan,
            "non_halal": is_non_halal,
            "has_pork_or_alcohol": has_pork_or_alcohol,
            "has_pork": has_pork,
            "has_alcohol": has_alcohol,
        }

    def _looks_like_undo(self, raw: str) -> bool:
        t = raw.strip().lower()
        return t in {"undo", "undo last action", "undo that", "go back", "back", "revert"} or t.startswith("undo ")

    def _push_undo(self, label: str = "last action") -> None:
        # don’t push empty/no-op states too aggressively; keep it simple
        snap = self.state.make_snapshot()
        self.state.undo_stack.append(UndoEntry(snapshot=snap, label=label))
        # cap history
        MAX_UNDO = 25
        if len(self.state.undo_stack) > MAX_UNDO:
            self.state.undo_stack.pop(0)

    def _handle_undo(self) -> str:
        if not self.state.undo_stack:
            return "Nothing to undo."

        before_user = self.state.user_name
        before_bot = self.state.bot_name
        before_cart = list(self.state.cart)
        before_pending = (self.state.awaiting_item_for_pending_options, list(self.state.pending_options_queue))

        entry = self.state.undo_stack.pop()
        self.state.restore_snapshot(entry.snapshot)

        changes = []
        if self.state.user_name != before_user:
            changes.append(f"user name → {self.state.user_name}")
        if self.state.bot_name != before_bot:
            changes.append(f"bot name → {self.state.bot_name}")
        if list(self.state.cart) != before_cart:
            changes.append("cart reverted")
        if (self.state.awaiting_item_for_pending_options, list(self.state.pending_options_queue)) != before_pending:
            changes.append("customisations step reverted")

        if changes:
            return "Undone — " + ", ".join(changes) + "."
        return "Undone."




