# Chatbot/state_models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import deque
from typing import Deque
from copy import deepcopy

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
    dietary_flags: set[str] = field(default_factory=set)

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
class UndoEntry:
    snapshot: Dict[str, Any]
    label: str = "last action"

@dataclass
class ChatStateSnapshot:
    user_name: str
    bot_name: str
    cart: list  # (CartLine list) keep as list to avoid circular typing headaches
    last_intent: Optional[str]
    awaiting_user_name: bool
    awaiting_bot_name: bool
    active_flow: Optional["FlowContext"]
    awaiting_item_for_pending_options: bool
    pending_options_queue: Deque[str]
    dietary_flags: set[str] = field(default_factory=set)

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
    undo_stack: List[UndoEntry] = field(default_factory=list)
    last_referenced_line_id: Optional[str] = None
    awaiting_disambiguation: bool = False
    disambiguation_candidates: List[str] = field(default_factory=list)
    disambiguation_reason: Optional[str] = None

    def make_snapshot(self) -> ChatStateSnapshot:
        return ChatStateSnapshot(
            user_name=self.user_name,
            bot_name=self.bot_name,
            cart=deepcopy(self.cart),
            last_intent=self.last_intent,
            awaiting_user_name=self.awaiting_user_name,
            awaiting_bot_name=self.awaiting_bot_name,
            active_flow=deepcopy(self.active_flow),
            awaiting_item_for_pending_options=self.awaiting_item_for_pending_options,
            pending_options_queue=deque(self.pending_options_queue),
            dietary_flags=set(self.dietary_flags),
        )

    def restore_snapshot(self, snap: ChatStateSnapshot) -> None:
        self.user_name = snap.user_name
        self.bot_name = snap.bot_name
        self.cart = deepcopy(snap.cart)
        self.last_intent = snap.last_intent
        self.awaiting_user_name = snap.awaiting_user_name
        self.awaiting_bot_name = snap.awaiting_bot_name
        self.active_flow = deepcopy(snap.active_flow)
        self.awaiting_item_for_pending_options = snap.awaiting_item_for_pending_options
        self.pending_options_queue = deque(snap.pending_options_queue)
        self.dietary_flags = set(getattr(snap, "dietary_flags", []))
