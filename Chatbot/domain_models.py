# chatbot/domain_models.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque


# ---------- Core enums ----------

class UserMood(str, Enum):
    """Simple mood labels for guardrails / UX tweaks."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    CONFUSED = "confused"


# ---------- User profile & naming ----------

@dataclass
class UserProfile:
    """
    Stores basic user-related state that must persist during a session.

    - user_name: what we call the user (can be changed via 'change_user_name' intent)
    - bot_name:  what the bot calls itself (can be changed via 'change_bot_name' intent)
    """
    user_name: str = "Guest"
    bot_name: str = "BrewBot"


# ---------- Cart & ordering ----------

@dataclass
class CartItem:
    """
    Represents a single line in the cart.

    We intentionally keep it generic:
    - item_id: matches `menu.csv` item_id
    - name: user-facing name (copied from menu at the moment of adding)
    - size: e.g. 'short', 'tall', 'grande', 'regular', 'large' (optional)
    - quantity: how many of this configuration
    - selected_options: list of option_ids from `options.csv` (milk, syrups, toppings)
    - base_price/base_calories: copied from menu so we can compute totals even if
      menu lookups are not repeated every time.

    Price and calories *including* options are computed by higher-level logic
    (dialogue_manager) using options.csv — we just store identifiers here.
    """
    item_id: str
    name: str
    size: Optional[str] = None
    quantity: int = 1
    base_price: float = 0.0
    base_calories: float = 0.0
    selected_options: List[str] = field(default_factory=list)

    def add_option(self, option_id: str) -> None:
        if option_id not in self.selected_options:
            self.selected_options.append(option_id)

    def remove_option(self, option_id: str) -> None:
        if option_id in self.selected_options:
            self.selected_options.remove(option_id)


@dataclass
class Cart:
    """
    Simple cart model for the conversation.

    - items: list of CartItem lines.
    We don't deduplicate by (item_id + options) here; that logic can be added later
    if you want merging. This keeps the model simple and easy to reason about.
    """
    items: List[CartItem] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def add_item(self, item: CartItem) -> None:
        self.items.append(item)

    def clear(self) -> None:
        self.items.clear()

    def remove_item(self, index: int) -> bool:
        """
        Remove an item by its index in the list.
        Returns True if removal succeeded, False if index was invalid.
        """
        if 0 <= index < len(self.items):
            del self.items[index]
            return True
        return False

    def total_base_price(self) -> float:
        """Sum of base prices * quantity (options not included)."""
        return sum(i.base_price * i.quantity for i in self.items)

    def total_base_calories(self) -> float:
        """Sum of base kcal * quantity (options not included)."""
        return sum(i.base_calories * i.quantity for i in self.items)

@dataclass
class MenuItem:
    """
    Represents one row from menu.csv.

    Only the fields actually used by DialogueManager are mandatory here:
    - item_id, name
    - suitable_vegetarian / suitable_vegan (optional flags)
    - allergens / may_contain: dicts of boolean flags
    """
    item_id: str
    name: str

    # optional metadata (you can extend to mirror menu.csv exactly)
    brand: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    base_price_gbp: float = 0.0
    is_active: bool = True
    calories_kcal: float = 0.0

    # allergen maps, e.g. {"contains_milk": True, "contains_soya": False, ...}
    allergens: Dict[str, bool] = field(default_factory=dict)
    may_contain: Dict[str, bool] = field(default_factory=dict)

    suitable_vegetarian: Optional[bool] = None
    suitable_vegan: Optional[bool] = None

    ingredients_text: str = ""
    nutrition_source_url: str = ""
    diet_tags: List[str] = field(default_factory=list)
    notes: str = ""

    @dataclass
    class OptionItem:
        option_id: str  # e.g. "opt_vanilla_pump"
        name: str  # e.g. "Vanilla syrup (1 pump)"
        category: str  # e.g. "syrup", "extra_shot", "plant_milk"
        calories_kcal: float = 0.0
        brand: str | None = None  # for brand-specific options

@dataclass
class MenuOption:
    """
    Represents one row from options.csv.

    - option_id: unique ID (e.g. "milk_oat", "size_grande")
    - option_group: logical group (size, milk, syrup, topping, sweetener, extra_shot, other)
    - option_value: the value shown to the user (e.g. "oat", "grande")
    - price_delta: additional price in GBP added to the base item
    - calories_delta_kcal: additional calories added to the base item
    - applies_to_scope: scope string (e.g. "item_id:..." or "category:Beverage")
                         to be interpreted by higher-level logic
    - allergens / may_contain: boolean flags (same naming as menu.csv)
    - suitable_vegetarian_delta / suitable_vegan_delta:
        "no_change" | "force_true" | "force_false" | None
    - notes: free-text notes from CSV
    """
    option_id: str
    option_group: str
    option_value: str

    price_delta: float = 0.0
    calories_delta_kcal: float = 0.0
    applies_to_scope: str = ""

    allergens: Dict[str, bool] = field(default_factory=dict)
    may_contain: Dict[str, bool] = field(default_factory=dict)

    suitable_vegetarian_delta: Optional[str] = None
    suitable_vegan_delta: Optional[str] = None

    notes: str = ""

    def scope_rules(self) -> List[Tuple[str, str]]:
        """
        Parse applies_to_scope like:
          - "category:Beverage"
          - "item_id:a|item_id:b|item_id:c"

        Returns a list of (key, value) pairs.
        """
        raw = (self.applies_to_scope or "").strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        rules: List[Tuple[str, str]] = []
        for p in parts:
            if ":" in p:
                k, v = p.split(":", 1)
                rules.append((k.strip(), v.strip()))
        return rules


# ---------- Intent / FAQ-related light models ----------

@dataclass
class IntentExample:
    """
    One labelled training example for intent detection.

    This corresponds to a row in intents.csv:
    - utterance (text)
    - intent label (string)
    """
    utterance: str
    intent: str


@dataclass
class FAQEntry:
    """
    Represents a row from faqs.csv.

    Fields mirror the CSV schema; types are kept simple because
    most code just needs strings.
    """
    faq_id: str
    question: str
    answer_markdown: str
    tags: str
    source_url: str
    last_verified_yyyy_mm_dd: str


# ---------- Dialogue stack / queue frames ----------

@dataclass
class DialogueFrame:
    """
    A single 'frame' of dialogue, used for stack/queue-style state.

    Examples:
    - user started a 'build_drink' flow, and we are still collecting slots
    - user is in the middle of 'checkout' confirmation

    `slot_values` stores partial info like:
    {
        "item_id": "sbx_misto_almond",
        "size": "tall",
        "milk": "almond"
    }
    """
    intent: str
    slot_values: Dict[str, Any] = field(default_factory=dict)


# ---------- Global chat state ----------

@dataclass
class ChatState:
    """
    Global, mutable state for one user session.

    This is the object the DialogueManager will read & update after each turn.

    Key responsibilities:
    - Track names (user_name, bot_name)
    - Track mood (for guardrails)
    - Track cart contents and whether checkout is allowed
    - Track last recognised intent and raw user utterance
    - Maintain a small history and a queue/stack of dialogue frames for
      multi-step flows (e.g., building a drink over multiple turns).
    """
    user_profile: UserProfile = field(default_factory=UserProfile)
    mood: UserMood = UserMood.NEUTRAL
    cart: Cart = field(default_factory=Cart)

    # Last N utterances (text only) for context / debugging
    history: deque[str] = field(default_factory=lambda: deque(maxlen=50))

    # For intent + slot based dialogue flows
    pending_frames: deque[DialogueFrame] = field(default_factory=deque)

    # For learning-from-successful-interactions / logging
    last_user_utterance: Optional[str] = None
    last_detected_intent: Optional[str] = None
    successful_interactions: int = 0

    def push_history(self, user_text: str, bot_text: Optional[str] = None) -> None:
        """
        Append latest turn(s) to the short history buffer for light context
        and debugging. Not used for any heavy NLP.
        """
        self.history.append(f"USER: {user_text}")
        if bot_text is not None:
            self.history.append(f"BOT: {bot_text}")

    # --- Dialogue frame helpers (queue-style) ---

    def enqueue_frame(self, frame: DialogueFrame) -> None:
        """Add a frame to the end of the pending queue."""
        self.pending_frames.append(frame)

    def dequeue_frame(self) -> Optional[DialogueFrame]:
        """Pop the next frame from the queue, or None if empty."""
        if self.pending_frames:
            return self.pending_frames.popleft()
        return None

    def peek_current_frame(self) -> Optional[DialogueFrame]:
        """Look at the current frame without removing it (front of the queue)."""
        return self.pending_frames[0] if self.pending_frames else None

    # --- Name helpers ---

    def set_user_name(self, new_name: str) -> None:
        self.user_profile.user_name = new_name

    def set_bot_name(self, new_name: str) -> None:
        self.user_profile.bot_name = new_name
