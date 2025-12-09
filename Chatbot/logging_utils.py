# Chatbot/logging_utils.py

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Any

# successful_intents.csv will live under data/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUCCESS_LOG_PATH = PROJECT_ROOT / "data" / "successful_intents.csv"


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
