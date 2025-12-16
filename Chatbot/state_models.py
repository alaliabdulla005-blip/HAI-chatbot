# Chatbot/state_models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
from typing import Deque
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

@dataclass
class FlowContext:
    """
    State for a multi-step flow such as a guided ordering wizard.

    - name: which flow is running (e.g. "order_wizard").
    - stage: current step name within that flow (e.g. "ask_category").
    - data: arbitrary key/value store for partial answers.
    """
    name: str
    stage: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatState(ConversationState):
    """
    Extended state which keeps everything from ConversationState plus:

    - active_flow: the flow currently in progress (if any).
    - flow_queue: queue of pending flows (for future use).
    """
    active_flow: Optional[FlowContext] = None
    flow_queue: List[FlowContext] = field(default_factory=list)
    awaiting_item_for_pending_options: bool = False
    pending_options_queue: Deque[str] = field(default_factory=deque)