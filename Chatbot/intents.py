"""
Intent detection module.

- Loads labelled utterances from intents.csv via data_loader.
- Trains a TF–IDF + LogisticRegression classifier (scikit-learn).
- Saves / loads the fitted vectorizer + classifier from .pkl files.
- Exposes a small API the rest of the chatbot can call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from enum import Enum

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data_loader import load_intent_examples
from .domain_models import IntentExample

# Default paths for model artifacts (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
INTENT_VECTORIZER_PATH = MODEL_DIR / "intent_vectorizer.pkl"
INTENT_CLASSIFIER_PATH = MODEL_DIR / "intent_classifier.pkl"
INTENTS_CSV_PATH = PROJECT_ROOT / "data" / "intents.csv"

class IntentLabel(str, Enum):
    """Enum of supported intent labels."""
    GREET = "greet"
    ASK_OPENING_HOURS = "ask_opening_hours"
    ASK_ALLERGENS_GENERAL = "ask_allergens_general"
    ASK_ITEM_ALLERGENS = "ask_item_allergens"
    ASK_VEGAN_OPTIONS = "ask_vegan_options"
    ASK_HALAL_OPTIONS = "ask_halal_options"
    ASK_GLUTEN_FREE = "ask_gluten_free"
    ASK_CALORIES = "ask_calories"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    SHOW_CART = "show_cart"
    SHOW_MENU = "show_menu"
    CHECKOUT = "checkout"
    CHANGE_USER_NAME = "change_user_name"
    CHANGE_BOT_NAME = "change_bot_name"
    USER_THANKS = "user_thanks"
    USER_COMPLAINT = "user_complaint"
    USER_ANGRY = "user_angry"
    USER_INSULT = "user_insult"
    OUT_OF_SCOPE = "out_of_scope"
    UNDO = "undo"

INTENT_STR_TO_LABEL: Dict[str, IntentLabel] = {
    label.value: label for label in IntentLabel
}

@dataclass
class IntentPrediction:
    """Simple container for an intent prediction result."""
    intent: Optional[str]          # None if we treat it as "unknown"
    confidence: float              # max predicted probability in [0, 1]
    probabilities: Dict[str, float]  # per-intent probability distribution


class SklearnIntentClassifier:
    """
    Thin wrapper around a TF–IDF + LogisticRegression pipeline.

    Usage:
        clf = SklearnIntentClassifier.train_from_csv()
        pred = clf.predict("hi there")
        print(pred.intent, pred.confidence)
    """

    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        classifier: LogisticRegression,
        label_order: List[str],
        unknown_threshold: float = 0.10,
    ) -> None:
        """
        :param vectorizer: fitted TfidfVectorizer.
        :param classifier: fitted LogisticRegression.
        :param label_order: list of class labels in classifier.classes_order.
        :param unknown_threshold: below this max prob, we treat as "unknown".
        """
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.label_order = label_order
        self.unknown_threshold = float(unknown_threshold)

    # ---------- Core prediction API ----------

    def predict(self, text: str) -> IntentPrediction:
        """
        Predict intent for a single user utterance.

        Returns IntentPrediction with:
        - intent = label string, or None if below threshold
        - confidence = max probability
        - probabilities = dict[intent -> prob]
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return IntentPrediction(intent=None, confidence=0.0, probabilities={})

        X = self.vectorizer.transform([cleaned])
        proba = self.classifier.predict_proba(X)[0]  # shape: (n_classes,)

        probs_by_label = {
            label: float(p) for label, p in zip(self.label_order, proba)
        }
        best_label = max(probs_by_label, key=probs_by_label.get)
        best_score = probs_by_label[best_label]

        if best_score < self.unknown_threshold:
            # Let the dialogue manager handle fallback.
            return IntentPrediction(intent=None, confidence=best_score, probabilities=probs_by_label)

        return IntentPrediction(intent=best_label, confidence=best_score, probabilities=probs_by_label)

    # ---------- Persistence helpers ----------

    def save(self, model_dir: Optional[Path] = None) -> None:
        """
        Save the vectorizer + classifier to .pkl files under model_dir.
        """
        if model_dir is None:
            model_dir = MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, model_dir / "intent_vectorizer.pkl")
        joblib.dump(self.classifier, model_dir / "intent_classifier.pkl")

    @classmethod
    def load(
        cls,
        model_dir: Optional[Path] = None,
        unknown_threshold: float = 0.10,
    ) -> Optional["SklearnIntentClassifier"]:
        """
        Load a previously-saved classifier.

        Returns None if artifacts do not exist yet.
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        vec_path = model_dir / "intent_vectorizer.pkl"
        clf_path = model_dir / "intent_classifier.pkl"

        if not vec_path.exists() or not clf_path.exists():
            return None

        vectorizer: TfidfVectorizer = joblib.load(vec_path)
        classifier: LogisticRegression = joblib.load(clf_path)
        labels = list(classifier.classes_)
        return cls(vectorizer, classifier, labels, unknown_threshold=unknown_threshold)

    # ---------- Training helpers ----------

    @classmethod
    def train_from_examples(
        cls,
        examples: List[IntentExample],
        unknown_threshold: float = 0.10,
    ) -> "SklearnIntentClassifier":
        """
        Train a fresh classifier from a list of IntentExample objects.
        """
        texts: List[str] = [ex.utterance for ex in examples]
        labels: List[str] = [ex.intent for ex in examples]

        # TF–IDF over unigrams + bigrams is a strong baseline for short intents.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            lowercase=True,
            strip_accents="unicode",
        )
        X = vectorizer.fit_transform(texts)

        classifier = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            n_jobs=None,
        )
        classifier.fit(X, labels)

        label_order = list(classifier.classes_)
        return cls(vectorizer, classifier, label_order, unknown_threshold=unknown_threshold)

    @classmethod
    def train_from_csv(
        cls,
        csv_path: Optional[Path] = None,
        unknown_threshold: float = 0.10,
    ) -> "SklearnIntentClassifier":
        """
        Convenience helper: load intents.csv and train.

        :param csv_path: override path to intents.csv if needed.
        """
        if csv_path is None:
            csv_path = INTENTS_CSV_PATH

        examples = load_intent_examples(csv_path)
        if not examples:
            raise ValueError(f"No intent examples found in {csv_path}")

        return cls.train_from_examples(examples, unknown_threshold=unknown_threshold)

    def predict_intent(self, text: str) -> Tuple[Optional[IntentLabel], float]:
        """
        Wrapper used by DialogueManager.
        Returns (IntentLabel or None, confidence float).
        """
        pred = self.predict(text)  # reuse existing method

        if pred.intent is None:
            return None, pred.confidence

        label = INTENT_STR_TO_LABEL.get(pred.intent)
        # If for some reason we get a label we don't know, treat as unknown
        if label is None:
            return None, pred.confidence

        return label, pred.confidence

# ---------- Convenience function for the rest of the app ----------

def get_intent_model(
    retrain: bool = False,
    intents_csv: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    unknown_threshold: float = 0.10,
) -> SklearnIntentClassifier:
    """
    Entry point used by cli_app / streamlit_app:

    - If retrain=True, always train from CSV and overwrite .pkl files.
    - Else, try to load saved model; if missing, train and save once.

    This keeps training logic in one place and avoids re-training at every run.
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    if retrain:
        model = SklearnIntentClassifier.train_from_csv(
            csv_path=intents_csv or INTENTS_CSV_PATH,
            unknown_threshold=unknown_threshold,
        )
        model.save(model_dir)
        return model

    # Try loading existing artifacts
    loaded = SklearnIntentClassifier.load(
        model_dir=model_dir,
        unknown_threshold=unknown_threshold,
    )
    if loaded is not None:
        return loaded

    # First-ever run: train then save
    model = SklearnIntentClassifier.train_from_csv(
        csv_path=intents_csv or INTENTS_CSV_PATH,
        unknown_threshold=unknown_threshold,
    )
    model.save(model_dir)
    return model
