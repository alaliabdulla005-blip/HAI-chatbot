# Chatbot/state_models.py
from dataclasses import dataclass, field
from typing import List, Optional
from .domain_models import MenuItem, MenuOption  # if needed

@dataclass
class CartLine:
    item_id: str
    item_name: str
    quantity: int = 1
    selected_options: List[str] = field(default_factory=list)

@dataclass
class ConversationState:
    user_name: str = "Guest"
    bot_name: str = "BrewBot"
    cart: List[CartLine] = field(default_factory=list)
    last_intent: Optional[str] = None
    awaiting_user_name: bool = False
    awaiting_bot_name: bool = False
    anger_count: int = 0
    abuse_count: int = 0
