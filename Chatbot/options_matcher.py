import re
from typing import Dict, List, Optional, Tuple
from .domain_models import MenuOption as Option

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def option_applies_to_item(opt: Option, item_id: str, item_category: str) -> bool:
    rules = opt.scope_rules()
    if not rules:
        return True

    for k, v in rules:
        if k == "item_id" and v == item_id:
            return True
        if k == "category" and v.lower() == (item_category or "").lower():
            return True

    return False

def build_phrase_index(options_by_id: Dict[str, Option]) -> Dict[str, List[str]]:
    """
    Maps normalized phrases -> [option_id,...]
    Uses option_value and also some common variations.
    """
    idx: Dict[str, List[str]] = {}

    def add(phrase: str, oid: str):
        p = _norm(phrase)
        idx.setdefault(p, []).append(oid)

    for oid, opt in options_by_id.items():
        # Base phrases
        add(opt.option_value, oid)
        add(f"{opt.option_group} {opt.option_value}", oid)

        # Some helpful synonyms for your CSV values
        if opt.option_group == "milk":
            add(f"{opt.option_value} milk", oid)
        if opt.option_group == "extra_shot" and opt.option_value == "espresso_shot":
            add("extra shot", oid)
            add("extra espresso", oid)
            add("espresso shot", oid)
        if opt.option_group == "topping":
            add(opt.option_value.replace("_", " "), oid)
        if opt.option_group == "size":
            add(opt.option_value.replace("_", " "), oid)

        # sugar-free phrasing
        if "sugar_free" in opt.option_value:
            add(opt.option_value.replace("sugar_free", "sugar free"), oid)

    return idx

def extract_options_from_text(
    text: str,
    options_by_id: Dict[str, Option],
    phrase_index: Dict[str, List[str]],
    item_id: str,
    item_category: str,
) -> List[Option]:
    """
    Finds options mentioned in free text by exact phrase hits over the index,
    then filters by applies_to_scope for that item.
    """
    t = _norm(text)

    hits: List[Tuple[int, int, str]] = []  # (start, end, phrase)
    for phrase in phrase_index.keys():
        p = phrase
        if not p or len(p) < 3:
            continue
        # simple substring match (works well for your controlled vocab)
        pos = t.find(p)
        if pos != -1:
            hits.append((pos, pos + len(p), p))

    # pick non-overlapping hits (longest-first)
    hits.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    chosen: List[Tuple[int, int, str]] = []
    used_ranges: List[Tuple[int, int]] = []

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for h in hits:
        r = (h[0], h[1])
        if any(overlaps(r, ur) for ur in used_ranges):
            continue
        chosen.append(h)
        used_ranges.append(r)

    # resolve phrases -> option_ids -> Options, filtered by scope
    result: List[Option] = []
    for _, _, phrase in chosen:
        candidate_ids = phrase_index.get(phrase, [])
        candidates = [
            options_by_id[oid] for oid in candidate_ids
            if option_applies_to_item(options_by_id[oid], item_id, item_category)
        ]
        # If multiple survive (e.g., vanilla syrup in Starbucks + Costa), keep them all;
        # your cart logic can dedupe by group later.
        result.extend(candidates)

    # dedupe by option_id while keeping order
    seen = set()
    deduped: List[Option] = []
    for opt in result:
        if opt.option_id not in seen:
            seen.add(opt.option_id)
            deduped.append(opt)

    return deduped
