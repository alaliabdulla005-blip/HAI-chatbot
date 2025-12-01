#!/usr/bin/env python3
"""
chatbot/cli.py

Simple command-line interface for the allergen-aware café chatbot.

Run from the project root as:

    python -m chatbot.cli

Options:
    --retrain-intents   Retrain the intent classifier from data/intents.csv
                        before starting the chat loop.
"""

from __future__ import annotations

import argparse
from typing import Optional

from Chatbot.data_loader import load_menu_items
from Chatbot.faq_search import FAQSearchEngine
from Chatbot.intents import get_intent_model
from Chatbot.dialogue_manager import DialogueManager


# ---------- Construction helpers ----------

def build_dialogue_manager(retrain_intents: bool = False) -> DialogueManager:
    """
    Initialise all core components and return a ready-to-use DialogueManager.
    """
    intent_model = get_intent_model(retrain=retrain_intents)  # first time: use --retrain-intents
    faq_engine = FAQSearchEngine.from_csv()
    menu_items = load_menu_items()

    dm = DialogueManager(
        intent_classifier=intent_model,
        faq_engine=faq_engine,
        menu_items=menu_items,
    )
    return dm


# ---------- CLI loop ----------

def run_cli(dm: DialogueManager) -> None:
    """
    Simple REPL-style loop:

    - Read user input from stdin
    - Pass to DialogueManager.handle_message
    - Print bot reply
    """
    print("Café Bot")
    print("Type 'quit', 'exit' or press Ctrl+C to leave.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋")
            break

        if not user_text:
            continue

        if user_text.lower() in {"quit", "exit", "q"}:
            print("Bot: Goodbye! 👋")
            break

        reply = dm.handle_message(user_text)
        # Use the current bot_name from state so renaming works
        bot_name = getattr(dm.state, "bot_name", "Bot")
        print(f"{bot_name}: {reply}\n")


# ---------- CLI argument parsing ----------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-line interface for the allergen-aware café chatbot.",
    )
    parser.add_argument(
        "--retrain-intents",
        action="store_true",
        help=(
            "Retrain the intent classifier from data/intents.csv before starting. "
            "Use this after you edit intents.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    dm = build_dialogue_manager(retrain_intents=args.retrain_intents)
    run_cli(dm)


if __name__ == "__main__":
    main()
