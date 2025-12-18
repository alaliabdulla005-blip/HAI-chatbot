"""
constants.py
Stores static configuration, keywords, regex patterns, and messages.
"""
import re

ORDER_WIZARD_FLOW = "order_wizard"

# --- Messages ---

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

# --- Keywords & Cues ---

GREET_KEYWORDS = {
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "hiya", "yo", "morning", "afternoon",
}

ALLERGEN_CUES = {
    "allergen", "allergens", "allergy", "allergic",
    "contain", "contains", "may contain",
    "milk", "dairy", "lactose",
    "egg", "eggs", "nuts", "peanut", "gluten", "wheat", "soy", "soya", "sesame", "mustard",
}

CALORIE_CUES = {"calorie", "calories", "kcal", "cals"}

MATCH_STOPWORDS = {
    "do", "does", "did", "is", "are", "was", "were",
    "the", "a", "an", "this", "that", "these", "those",
    "with", "without", "of", "for", "to", "in", "on", "at",
    "my", "your", "our", "their", "its", "it", "i", "you", "we", "they",
    "please", "pls", "can", "could", "would", "will", "shall", "should",
    "any", "some", "just", "and", "or", "plus",
    "what", "which", "how", "many", "much", "milk", "dairy", "lactose", "drink", "semi",
    "skimmed", "whole",
    "contains", "contain", "containing", "has", "have", "having",
    "allergen", "allergens", "allergy", "allergic", "safe", "suitable", "free",
    "info", "information", "list", "menu", "guide", "check", "show", "tell", "ingredients",
    "gluten", "calorie", "calories", "kcal", "sugar", "fat", "fats", "fatty",
    "vegan", "veg", "vegetarian", "veggie",
    "cart", "basket", "order", "bag",
    "add", "remove", "delete", "take", "put", "get", "buy", "drop",
    "cancel", "undo", "clear", "empty", "want", "like", "have",
    "one", "two", "three", "1", "2", "3",
}

OPTION_MARKERS = {
    "milk", "oat", "soya", "soy", "almond", "coconut",
    "syrup", "vanilla", "caramel", "hazelnut", "brown sugar", "sugar free",
    "shot", "espresso shot", "extra shot",
    "whipped cream", "drizzle",
    "short", "tall", "grande", "venti", "regular", "large",
    "sweetener", "sugar",
}

YES_LIKE = {"yes", "y", "yeah", "yep", "ok", "okay", "sure"}

SINGLE_CHOICE_GROUPS = {"milk", "size"}

GENERIC_TOKENS_BY_GROUP = {
    # remove generic words so we don't match everything just because user said "milk" or "syrup"
    "milk":    {"milk", "drink", "dairy", "lactose"},
    "syrup":   {"syrup", "syrups", "pump", "pumps", "flavour", "flavor", "sugar", "free"},
    "topping": {"topping", "toppings", "drizzle", "whip", "whipped", "cream", "sprinkle", "sprinkles"},
    "size":    {"size", "hot", "iced"},
    "sweetener": {"sweetener", "sweeteners", "sugar"},
    "extra_shot": {"extra"},
}

# --- Regex Patterns ---

USER_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi am\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi'm\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bim\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bcall me\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bname is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bnew name is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bfrom now on call me\s+(.+)$", re.IGNORECASE),
]

BOT_NAME_PATTERNS = [
    re.compile(r"\byour name is\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi['’]ll call you\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bill call you\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bim calling you\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bi am calling you\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bchange your name to\s+(.+)$", re.IGNORECASE),
    re.compile(r"\brename (?:you|yourself|the bot|bot) to\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bset bot name to\s+(.+)$", re.IGNORECASE),
    re.compile(r"\bnew name for you\s+(.+)$", re.IGNORECASE),
]