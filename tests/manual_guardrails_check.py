# tests/manual_guardrails_check.py

import sys
from pathlib import Path

# Make sure the project root is on sys.path so "import Chatbot" works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Chatbot.state_models import ConversationState
from Chatbot import guardrails


def run_abuse_escalation_demo() -> None:
    print("=== Abuse escalation demo ===")
    state = ConversationState()

    msgs = [
        "you are a stupid bot",
        "you are a stupid bot again",
        "you are the worst bot ever",  # should hit escalation once threshold reached
        "ok sorry, can you show me vegan options?",  # neutral
    ]

    for i, m in enumerate(msgs, 1):
        reply = guardrails.apply_guardrails(m, state)
        print(f"\nTurn {i}: {m!r}")
        print(" guardrails reply:", repr(reply))
        print(" anger_count:", getattr(state, "anger_count", None),
              "abuse_count:", getattr(state, "abuse_count", None))


def run_anger_decay_demo() -> None:
    print("\n=== Anger + decay demo ===")
    state = ConversationState()

    msgs = [
        "this is useless and really annoying",   # angry
        "still awful and frustrating",           # angry again
        "just show me my cart",                  # neutral → should decay anger a bit
        "this is ridiculous",                    # angry again
    ]

    for i, m in enumerate(msgs, 1):
        reply = guardrails.apply_guardrails(m, state)
        print(f"\nTurn {i}: {m!r}")
        print(" guardrails reply:", repr(reply))
        print(" anger_count:", getattr(state, "anger_count", None),
              "abuse_count:", getattr(state, "abuse_count", None))


if __name__ == "__main__":
    run_abuse_escalation_demo()
    run_anger_decay_demo()
