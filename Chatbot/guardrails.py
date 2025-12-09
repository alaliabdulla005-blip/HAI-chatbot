# guardrails.py

from __future__ import annotations

import re
from typing import Optional

from .state_models import ConversationState
from .intents import IntentLabel  # for emotional-intent filtering

# ----------- Simple regexes for sensitive info -----------

CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b\+?\d{8,15}\b")

# Phrases we treat as strong emotional / angry signals
ANGRY_WORDS = {
    "annoyed", "useless", "waste of time", "frustrating",
    "fed up", "awful", "ridiculous", "worst bot", "not happy",
    "not satisfied", "bad experience", "rubbish",
}

# More direct insults / abuse aimed at the bot
ABUSE_KEYWORDS = {
    "stupid", "idiot", "dumb bot", "moron", "loser",
    "hate you", "shut up", "f off",
}

# Complaint-like phrasing
COMPLAINT_PHRASES = {
    "complaint",
    "make a complaint",
    "file a complaint",
    "not working",
    "isn't working",
    "isnt working",
    "doesn't work",
    "doesnt work",
    "working properly",
}

# Very simple self-harm / crisis patterns
SELF_HARM_PATTERNS = [
    r"\bi want to die\b",
    r"\bwant to die\b",
    r"\bkill myself\b",
    r"\bending my life\b",
    r"\b(end|ending) it all\b",
    r"\bsuicide\b",
    r"\bhurt myself\b",
]


def get_emotion_flags(lowered: str) -> dict[str, bool]:
    """
    Central place to detect emotional signals in text.

    Returns a dict with:
    - 'anger': strong frustration / anger words
    - 'insult': abusive / insulting language towards the bot
    - 'complaint': explicit complaint phrasing
    """
    has_anger = any(word in lowered for word in ANGRY_WORDS)
    has_insult = any(word in lowered for word in ABUSE_KEYWORDS)
    has_complaint = any(phrase in lowered for phrase in COMPLAINT_PHRASES)
    return {
        "anger": has_anger,
        "insult": has_insult,
        "complaint": has_complaint,
    }


def filter_emotional_intent(
    intent_label: Optional[IntentLabel],
    lowered: str,
) -> Optional[IntentLabel]:
    """
    Guard against the classifier mislabelling neutral questions as
    emotional intents (USER_ANGRY / USER_INSULT / USER_COMPLAINT).

    If there are no supporting keywords in the text, we downgrade
    those intents back to None so the DialogueManager can fall back
    gracefully.
    """
    if intent_label is None:
        return None

    flags = get_emotion_flags(lowered)
    is_anger = flags["anger"]
    is_insult = flags["insult"]
    is_complaint = flags["complaint"]

    if intent_label in {IntentLabel.USER_ANGRY, IntentLabel.USER_INSULT}:
        # Only keep these intents if we actually see anger/insult signals
        if not (is_anger or is_insult):
            return None

    elif intent_label is IntentLabel.USER_COMPLAINT:
        # Complaint needs explicit complaint phrasing OR anger/insult context
        if not (is_complaint or is_anger or is_insult):
            return None

    return intent_label


def apply_guardrails(user_text: str, state: ConversationState) -> Optional[str]:
    """
    Global guardrail pass.

    - Self-harm / crisis messages → short-circuit to a supportive message.
    - Obvious privacy leaks (card, email, phone) → short-circuit to a warning.
    - Repeated abuse / anger → escalate to "please talk to staff / stopping chat".
    - Otherwise → return None so DialogueManager proceeds as normal.
    """
    lowered = (user_text or "").lower()

    # ------------------------------------------------------
    # 1) Self-harm / crisis support
    # ------------------------------------------------------
    for pattern in SELF_HARM_PATTERNS:
        if re.search(pattern, lowered):
            return (
                "I’m really sorry that you’re feeling like this. "
                "I’m just a café assistant and I can’t provide the right kind of help, "
                "but it might help to talk to someone you trust or a professional support "
                "service in your area."
            )

    # ------------------------------------------------------
    # 2) Privacy guardrails: obvious sensitive info
    # ------------------------------------------------------
    if CARD_PATTERN.search(user_text):
        return (
            "Please don’t enter card numbers or payment details here – this chat is only for "
            "menu questions and placing café orders. You can pay safely using the payment terminal."
        )

    if EMAIL_PATTERN.search(user_text) or PHONE_PATTERN.search(user_text):
        return (
            "You don’t need to share your email address or phone number here. "
            "I can still help with menu questions and your cart without that."
        )

    # ------------------------------------------------------
    # 3) Escalation for repeated abuse towards the bot
    # ------------------------------------------------------
    flags = get_emotion_flags(lowered)

    if flags["insult"]:
        abuse_count = getattr(state, "abuse_count", 0) + 1
        setattr(state, "abuse_count", abuse_count)

        # After 3 abusive turns, stop the chat politely
        if abuse_count >= 3:
            return (
                "Because of the language being used, I’m going to stop this chat now. "
                "If you still need help with your order, a staff member can assist you in person."
            )
        # For the first / second insult, let the normal USER_INSULT intent reply handle it.
        # So: no short-circuit, just continue.
        return None

    # ------------------------------------------------------
    # 4) Escalation for repeated strong anger (without abuse)
    # ------------------------------------------------------
    if flags["anger"]:
        anger_count = getattr(state, "anger_count", 0) + 1
        setattr(state, "anger_count", anger_count)

        # After 3 strongly angry turns, gently redirect to human staff
        if anger_count >= 3:
            return (
                "It sounds like this is really frustrating. I’m only a simple café bot – "
                "you might find it easier to speak directly to a staff member who can help you."
            )

    # ------------------------------------------------------
    # 5) Default: allow normal handling
    # ------------------------------------------------------
    return None
