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

from .intents import SklearnIntentClassifier, IntentLabel
from .faq_search import FAQSearchEngine, FAQMatch
from .domain_models import MenuItem, FAQEntry  # adjust imports if needed
# from .logging_utils import log_turn  # TODO: integrate later
# from . import guardrails             # TODO: integrate mood / safety guardrails later


# ---------- Conversation state models ----------


@dataclass
class CartLine:
    """
    Minimal cart line.

    You can later swap this to use a CartItem from domain_models if you
    already defined one there – just update types accordingly.
    """
    item_id: str
    item_name: str
    quantity: int = 1


@dataclass
class ConversationState:
    """All state we need to remember across turns."""
    user_name: str = "Guest"
    bot_name: str = "BrewBot"
    cart: List[CartLine] = field(default_factory=list)
    last_intent: Optional[IntentLabel] = None
    awaiting_user_name: bool = False
    awaiting_bot_name: bool = False


@dataclass
class ChatbotConfig:
    """Tunables for thresholds etc."""
    intent_confidence_threshold: float = 0.10
    faq_min_score: float = 0.25
    faq_top_k: int = 3


# Fallback messages when we don’t understand
FALLBACK_MESSAGES = [
    "I’m not completely sure what you mean. Could you rephrase that?",
    "Sorry, I didn’t quite get that. Try asking about drinks, allergens, or your order.",
    "I might be missing something – could you say it in another way?",
]


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

    GREET_KEYWORDS = {
        "hi","hello","hey","good morning","good afternoon",
        "good evening","hiya","yo","morning","afternoon",
    }

    ANGRY_KEYWORDS = {
        "annoyed", "useless", "waste of time", "frustrating",
        "fed up", "awful", "ridiculous", "worst bot", "not happy",
        "not satisfied", "bad experience", "rubbish",
    }

    INSULT_KEYWORDS = {
        "stupid", "idiot", "dumb bot", "moron", "useless",
        "hate you", "loser", "ugly bot", "shut up", "f off",
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
        "any", "some", "just",
        "what", "which", "how", "many", "much",

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
    }

    def __init__(
        self,
        intent_classifier: SklearnIntentClassifier,
        faq_engine: FAQSearchEngine,
        menu_items: Sequence[MenuItem],
        config: Optional[ChatbotConfig] = None,
    ) -> None:
        self.intent_classifier = intent_classifier
        self.faq_engine = faq_engine
        self.menu_items = list(menu_items)
        self._menu_vocab = self._build_menu_vocab(self.menu_items)
        self.config = config or ChatbotConfig()
        self.state = ConversationState()

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

        # 1) Pending “name” answers override normal intent detection
        if self.state.awaiting_user_name:
            return self._handle_user_name_reply(raw)

        if self.state.awaiting_bot_name:
            return self._handle_bot_name_reply(raw)

        lowered = raw.lower()

        # Direct question about the stored name
        if "what is my name" in lowered:
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
            return (
                "I can help with café questions, such as: allergens, vegan/halal/gluten-free options, "
                "calories calculations, and managing your order – adding/removing items, "
                "showing your cart, and checking out."
            )

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

        # 2) Intent detection
        intent_label, confidence = self.intent_classifier.predict_intent(raw)
        self.state.last_intent = intent_label

        if intent_label is None or confidence < self.config.intent_confidence_threshold:
            # Try insult / angry keywords before a generic fallback
            if any(kw in lowered for kw in self.INSULT_KEYWORDS):
                return self._handle_user_insult(raw)
            if any(kw in lowered for kw in self.ANGRY_KEYWORDS):
                return self._handle_user_angry(raw)
            # Otherwise: generic fallback
            return self._handle_fallback(raw)

        # 3) Dispatch to intent-specific handlers
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
            IntentLabel.CHECKOUT: self._handle_checkout,
            IntentLabel.CHANGE_USER_NAME: self._handle_change_user_name,
            IntentLabel.CHANGE_BOT_NAME: self._handle_change_bot_name,
            IntentLabel.USER_THANKS: self._handle_user_thanks,
            IntentLabel.USER_COMPLAINT: self._handle_user_complaint,
            IntentLabel.USER_ANGRY: self._handle_user_angry,
            IntentLabel.USER_INSULT: self._handle_user_insult,
            IntentLabel.OUT_OF_SCOPE: self._handle_out_of_scope,
        }

        handler = handler_map.get(intent_label)

        if handler is None:
            # Unknown / not yet wired intent label
            return self._handle_fallback(raw)

        response = handler(raw)

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
            "You can ask about allergens, vegan options, or start an order."
        )

    def _handle_change_user_name(self, user_text: str) -> str:
        """
        Try to extract the USER's name from this message.
        If not clear, ask a follow-up and wait for the next turn.
        """
        name = self._extract_user_name(user_text)
        if name:
            self.state.user_name = name
            self.state.awaiting_user_name = False
            return f"Got it, I’ll call you {name} from now on."

        self.state.awaiting_user_name = True
        return "Sure – what should I call you?"

    def _handle_change_bot_name(self, user_text: str) -> str:
        name = self._extract_bot_name(user_text)
        if name:
            self.state.bot_name = name
            self.state.awaiting_bot_name = False
            return f"Okay, I’ll go by {name} now."
        self.state.awaiting_bot_name = True
        return "What name would you like to give me?"

    def _handle_user_name_reply(self, user_text: str) -> str:
        # Reuse the same extraction logic (handles "my name is Ali", "I'm Ali", or just "Ali")
        name = self._extract_user_name(user_text) or self._clean_name(user_text)
        if not name:
            return "I didn’t quite catch that – could you give me a short name?"

        self.state.user_name = name
        self.state.awaiting_user_name = False
        return f"Nice to meet you, {name}!"

    def _handle_bot_name_reply(self, user_text: str) -> str:
        name = self._extract_bot_name(user_text) or self._clean_name(user_text)
        if not name:
            return "I’m not sure that works as a name – try something shorter?"
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
        faq = self.faq_engine.get_faq_by_id("openning_hours")
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

    def _handle_vegan_options(self, _: str) -> str:
        # Use FAQ if present, otherwise derive from menu.
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

    def _handle_halal_options(self, _: str) -> str:
        faq = self.faq_engine.get_faq_by_id("non_halal_items")
        if faq:
            # That FAQ explains non-halal items and filtering
            return faq.answer_markdown

        return (
            "Some food items contain pork or bacon and are not halal-friendly. "
            "Drinks built without alcoholic ingredients are usually halal-friendly. "
            "Please check item details or ask staff if unsure."
        )

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

    def _handle_calories(self, _: str) -> str:
        faq = self.faq_engine.get_faq_by_id("calorie_math")
        if faq:
            return faq.answer_markdown

        return (
            "I can add up calories based on each item’s base value plus extras "
            "(like syrups and extra espresso shots). Tell me your drink and customisations."
        )

    # ------------------------------------------------------------------
    # Handlers: cart / transaction state
    # ------------------------------------------------------------------

    def _handle_add_to_cart(self, user_text: str) -> str:
        """
        Very simple first pass:
        - Try to match a menu item by name.
        - If we find one, add quantity 1.
        Later you can extend this to parse size, milk choice, etc.
        """
        item = self._find_menu_item_in_text(user_text)
        if item is None:
            return (
                "I’m not sure which item to add. "
                "Try saying something like “add a tall Misto with oat drink to my order”."
            )

        # Add or increment in cart
        for line in self.state.cart:
            if line.item_id == item.item_id:
                line.quantity += 1
                break
        else:
            self.state.cart.append(CartLine(item_id=item.item_id, item_name=item.name, quantity=1))

        return f"Added {item.name} to your cart."

    def _handle_remove_from_cart(self, user_text: str) -> str:
        if not self.state.cart:
            return "Your cart is currently empty."

        item = self._find_menu_item_in_text(user_text)
        if item is None:
            # Try a very simple heuristic: if only one thing is in the cart, remove it
            if len(self.state.cart) == 1:
                removed = self.state.cart.pop()
                return f"Removed {removed.item_name} from your cart."
            return (
                "I couldn’t work out which item to remove. "
                "Please mention the drink or food name that’s in your cart."
            )

        new_cart: List[CartLine] = []
        removed_any = False
        for line in self.state.cart:
            if line.item_id == item.item_id:
                removed_any = True
                # remove one instance; if more than one, decrement quantity
                if line.quantity > 1:
                    line.quantity -= 1
                    new_cart.append(line)
                # else, drop it completely
            else:
                new_cart.append(line)

        self.state.cart = new_cart

        if not removed_any:
            return f"I couldn’t find {item.name} in your cart."
        return f"Removed one {item.name} from your cart."

    def _handle_show_cart(self, _: str) -> str:
        if not self.state.cart:
            return "Your cart is empty at the moment."
        lines = []
        for line in self.state.cart:
            lines.append(f"- {line.item_name} × {line.quantity}")
        cart_text = "\n".join(lines)
        return f"Here’s what’s in your cart:\n{cart_text}"

    def _handle_checkout(self, _: str) -> str:
        """
        Transaction-state rule:
        - You can only checkout if there is at least one item in the cart.
        """
        if not self.state.cart:
            return "You don’t have anything in your cart yet. Add a drink or snack before checking out."

        # For now, we just summarise and clear the cart;
        # later you can compute price, calories, etc.
        summary = self._handle_show_cart("")
        self.state.cart.clear()
        return summary + "\n\nYour order has been placed. Thank you!"

    # ------------------------------------------------------------------
    # Handlers: social / feedback
    # ------------------------------------------------------------------

    def _handle_user_thanks(self, _: str) -> str:
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

    def _tokenise_for_matching(self, s: str) -> List[str]:
        """
        Tokeniser used for fuzzy menu matching:
        - lowercases
        - normalises a few synonyms (e.g. 'choc' -> 'chocolate')
        - drops MATCH_STOPWORDS
        """
        TOKEN_NORMALISATIONS = {
            "choc": "chocolate",  # so "hot choc" matches "Hot Chocolate"
            "soy": "soya",  # treat "soy" the same as "soya"
        }

        tokens: List[str] = []
        for raw in re.findall(r"[a-z0-9]+", (s or "").lower()):
            norm = TOKEN_NORMALISATIONS.get(raw, raw)
            if norm and norm not in self.MATCH_STOPWORDS:
                tokens.append(norm)
        return tokens

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
        unknown_tokens = user_tokens_all - recognised_tokens
        if best_overlap == 1 and unknown_tokens:
            return None

        # At this point we have one or more "best" candidates that are
        # safe to consider. Use tie-breaking to pick the best one.
        return self._break_tie(user_tokens_all, recognised_tokens, candidates)

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
