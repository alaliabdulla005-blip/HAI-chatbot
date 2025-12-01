# chatbot/data_loader.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Mapping, Iterable, Optional

from .domain_models import FAQEntry, IntentExample, MenuItem, MenuOption


# ---------- Paths ----------

def get_data_dir() -> Path:
    """Return the root data/ directory for the project."""
    return Path(__file__).resolve().parents[1] / "data"


# ---------- Generic helpers ----------

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file and return a list of dict rows (all values as strings).

    We keep everything as strings here to avoid making assumptions about
    schema; conversion to float/int/bool is done later at the point of use.
    """
    rows: List[Dict[str, str]] = []
    with path.open(mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k: (v if v is not None else "") for k, v in raw_row.items()}
            rows.append(row)
    return rows


def _rows_to_dict_by_key(
    rows: Iterable[Mapping[str, str]],
    key_field: str,
) -> Dict[str, Dict[str, str]]:
    """
    Turn a list of CSV rows into a dict keyed by `key_field`.

    Example:
        rows -> {"sbx_misto_almond": {...}, "costa_express_latte": {...}}
    """
    result: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row.get(key_field, "")
        if not key:
            continue
        result[key] = dict(row)
    return result


def _parse_float(value: Optional[str], default: float = 0.0) -> float:
    """Best-effort conversion to float; fall back to default on errors/blank."""
    if value is None:
        return default
    v = value.strip()
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _parse_bool(value: Optional[str], default: Optional[bool] = False) -> Optional[bool]:
    """
    Best-effort conversion to bool.

    - "yes", "true", "1", "y", "t"  -> True
    - "no", "false", "0", "n", "f"  -> False
    - "", None or unknown           -> default
    """
    if value is None:
        return default
    v = value.strip().lower()
    if not v:
        return default
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    return default


# ---------- menu.csv ----------

def load_menu(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load menu.csv as a dict keyed by item_id (raw strings).

    Returns:
        {
            "sbx_misto_almond": { "item_id": "...", "brand": "starbucks", ... },
            ...
        }
    """
    if path is None:
        path = get_data_dir() / "menu.csv"
    rows = _read_csv_rows(path)
    return _rows_to_dict_by_key(rows, key_field="item_id")


def load_menu_items(path: Optional[Path] = None) -> List[MenuItem]:
    """
    Construct rich MenuItem objects from menu.csv.

    Uses:
    - base_price column -> MenuItem.base_price_gbp
    - is_active (0/1)
    - suitable_vegetarian / suitable_vegan (0/1)
    - calories_kcal
    - diet_tags: semicolon-separated tags, e.g. "vegetarian;halal-friendly"
    - ingredients_text, nutrition_source_url, notes
    - allergens / may_contain: all "contains_*" / "may_contain_*" columns
    """
    raw_by_id = load_menu(path)
    items: List[MenuItem] = []

    for item_id, row in raw_by_id.items():
        name = (row.get("name") or "").strip() or item_id

        brand = (row.get("brand") or "").strip() or None
        category = (row.get("category") or "").strip() or None
        subcategory = (row.get("subcategory") or "").strip() or None

        # base_price in CSV → base_price_gbp in model
        base_price_gbp = _parse_float(row.get("base_price"), default=0.0)
        is_active = bool(_parse_bool(row.get("is_active"), default=True))

        calories_kcal = _parse_float(row.get("calories_kcal"), default=0.0)

        # Allergen flags
        allergens: Dict[str, bool] = {}
        may_contain: Dict[str, bool] = {}
        for col_name, value in row.items():
            key_norm = (col_name or "").strip().lower()
            if key_norm.startswith("contains_"):
                allergens[key_norm] = bool(_parse_bool(value, default=False))
            elif key_norm.startswith("may_contain_"):
                may_contain[key_norm] = bool(_parse_bool(value, default=False))

        suitable_vegetarian = _parse_bool(row.get("suitable_vegetarian"), default=None)
        suitable_vegan = _parse_bool(row.get("suitable_vegan"), default=None)

        ingredients_text = (row.get("ingredients_text") or "").strip()
        nutrition_source_url = (row.get("nutrition_source_url") or "").strip()

        raw_tags = (row.get("diet_tags") or "").strip()
        diet_tags = [t.strip() for t in raw_tags.split(";") if t.strip()]

        notes = (row.get("notes") or "").strip()

        item = MenuItem(
            item_id=item_id,
            name=name,
            brand=brand,
            category=category,
            subcategory=subcategory,
            base_price_gbp=base_price_gbp,
            is_active=is_active,
            calories_kcal=calories_kcal,
            allergens=allergens,
            may_contain=may_contain,
            suitable_vegetarian=suitable_vegetarian,
            suitable_vegan=suitable_vegan,
            ingredients_text=ingredients_text,
            nutrition_source_url=nutrition_source_url,
            diet_tags=diet_tags,
            notes=notes,
        )
        items.append(item)

    return items


# ---------- options.csv ----------

def load_options(path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """
    Load options.csv as a dict keyed by option_id (raw strings).

    We keep this raw loader for any generic code that wants direct CSV rows.
    """
    if path is None:
        path = get_data_dir() / "options.csv"
    rows = _read_csv_rows(path)
    return _rows_to_dict_by_key(rows, key_field="option_id")


def load_option_items(path: Optional[Path] = None) -> List[MenuOption]:
    """
    Typed loader for options.csv → MenuOption objects.

    - price_delta, calories_delta_kcal parsed as floats
    - allergens / may_contain built from "contains_*" / "may_contain_*" cols
    - suitable_vegetarian_delta / suitable_vegan_delta left as strings
      ("no_change", "force_true", "force_false") or None
    """
    if path is None:
        path = get_data_dir() / "options.csv"

    raw_by_id = load_options(path)
    options: List[MenuOption] = []

    for option_id, row in raw_by_id.items():
        option_group = (row.get("option_group") or "").strip()
        option_value = (row.get("option_value") or "").strip()

        price_delta = _parse_float(row.get("price_delta"), default=0.0)
        calories_delta = _parse_float(row.get("calories_delta_kcal"), default=0.0)

        applies_to_scope = (row.get("applies_to_scope") or "").strip()

        allergens: Dict[str, bool] = {}
        may_contain: Dict[str, bool] = {}
        for col_name, value in row.items():
            key_norm = (col_name or "").strip().lower()
            if key_norm.startswith("contains_"):
                allergens[key_norm] = bool(_parse_bool(value, default=False))
            elif key_norm.startswith("may_contain_"):
                may_contain[key_norm] = bool(_parse_bool(value, default=False))

        veg_delta = (row.get("suitable_vegetarian_delta") or "").strip() or None
        vegan_delta = (row.get("suitable_vegan_delta") or "").strip() or None

        notes = (row.get("notes") or "").strip()

        opt = MenuOption(
            option_id=option_id,
            option_group=option_group,
            option_value=option_value,
            price_delta=price_delta,
            calories_delta_kcal=calories_delta,
            applies_to_scope=applies_to_scope,
            allergens=allergens,
            may_contain=may_contain,
            suitable_vegetarian_delta=veg_delta,
            suitable_vegan_delta=vegan_delta,
            notes=notes,
        )
        options.append(opt)

    return options


# ---------- faqs.csv ----------

def load_faqs(path: Optional[Path] = None) -> List[FAQEntry]:
    """
    Load faqs.csv into a list of FAQEntry objects.

    Expected header:
        faq_id, question, answer_markdown, tags, source_url, last_verified_yyyy_mm_dd
    """
    if path is None:
        path = get_data_dir() / "faqs.csv"

    rows = _read_csv_rows(path)
    faqs: List[FAQEntry] = []

    for r in rows:
        faqs.append(
            FAQEntry(
                faq_id=r.get("faq_id", "").strip(),
                question=r.get("question", "").strip(),
                answer_markdown=r.get("answer_markdown", "").strip(),
                tags=r.get("tags", "").strip(),
                source_url=r.get("source_url", "").strip(),
                last_verified_yyyy_mm_dd=r.get("last_verified_yyyy_mm_dd", "").strip(),
            )
        )

    return faqs


# ---------- intents.csv ----------

def load_intent_examples(path: Optional[Path] = None) -> List[IntentExample]:
    """
    Load intents.csv into a list of IntentExample objects.

    Minimal schema:
        utterance,intent

    Extra columns are ignored. Empty rows are skipped.
    """
    if path is None:
        path = get_data_dir() / "intents.csv"

    rows = _read_csv_rows(path)
    examples: List[IntentExample] = []

    for r in rows:
        utterance = r.get("utterance", "").strip()
        intent = r.get("intent", "").strip()
        if not utterance or not intent:
            continue
        examples.append(IntentExample(utterance=utterance, intent=intent))

    return examples
