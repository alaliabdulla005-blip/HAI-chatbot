# Chatbot/logging_utils.py

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Any

# successful_intents.csv will live under data/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUCCESS_LOG_PATH = PROJECT_ROOT / "data" / "successful_intents.csv"
GUARDRAIL_LOG_PATH = PROJECT_ROOT / "data" / "guardrail_events.csv"

def append_success_pair(utterance: str, intent: Any) -> None:
    """
    Append a single (timestamp, utterance, intent) row to successful_intents.csv.

    - Creates data/successful_intents.csv on first use, with a header row.
    - Silently does nothing if something goes badly wrong – callers should
      never crash because logging failed.
    """
    try:
        utterance = (utterance or "").strip()
        # Support passing either a plain string or an Enum like IntentLabel
        if hasattr(intent, "value"):
            intent_label = str(getattr(intent, "value"))
        else:
            intent_label = str(intent or "").strip()

        SUCCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = SUCCESS_LOG_PATH.exists()

        with SUCCESS_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp_utc_iso", "utterance", "intent"])

            timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
            writer.writerow([timestamp, utterance, intent_label])
    except Exception:
        # Hard fail is not acceptable for logging – swallow all errors.
        return

def append_guardrail_event(event_type: str, utterance: str, details: str = "") -> None:
    """
    Append a single guardrail event row to data/guardrail_events.csv.

    - event_type: short code like 'self_harm_pattern', 'card_block',
      'abuse_turn', 'abuse_escalated', 'anger_turn', 'anger_escalated'.
    - utterance: the original user text (sanitised as needed).
    - details: optional extra info (e.g. 'abuse_count=3').
    """
    try:
        utterance = (utterance or "").strip()
        event_type = (event_type or "").strip()
        details = (details or "").strip()

        GUARDRAIL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = GUARDRAIL_LOG_PATH.exists()

        with GUARDRAIL_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp_utc_iso", "event_type", "utterance", "details"])

            timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")
            writer.writerow([timestamp, event_type, utterance, details])
    except Exception:
        # Logging must never crash the bot
        return