import json
import os
from functools import lru_cache


_CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "..", "categories.json")


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    with open(_CATEGORIES_PATH, encoding="utf-8") as f:
        content = f.read().replace("\u00a0", " ")
        return json.loads(content)


def get_categories() -> dict[str, list[dict]]:
    """Return the raw category dict: {cat_name: [{name, description}, ...]}."""
    return _load_raw()


@lru_cache(maxsize=1)
def get_subcat_to_cat() -> dict[str, str]:
    """Build reverse map: subcat_name -> cat_name."""
    mapping: dict[str, str] = {}
    for cat, subcats in _load_raw().items():
        for entry in subcats:
            mapping[entry["name"]] = cat
    return mapping


@lru_cache(maxsize=1)
def get_all_subcats() -> list[str]:
    """Flat sorted list of all subcategory names."""
    return sorted(get_subcat_to_cat().keys())


@lru_cache(maxsize=1)
def get_all_cats() -> list[str]:
    """Sorted list of all category names."""
    return sorted(_load_raw().keys())


def get_cat_for_subcat(subcat: str) -> str:
    """Look up the parent category for a subcategory."""
    return get_subcat_to_cat().get(subcat, "")


@lru_cache(maxsize=1)
def get_grouped_subcats() -> list[str]:
    """Subcategories ordered by their parent category for nicer dropdown grouping.

    Returns list of subcat names in the order: cat1/sub1, cat1/sub2, cat2/sub1, ...
    """
    result: list[str] = []
    for cat in sorted(_load_raw().keys()):
        for entry in _load_raw()[cat]:
            result.append(entry["name"])
    return result


@lru_cache(maxsize=1)
def get_subcat_descriptions() -> dict[str, str]:
    """Map subcat_name -> description."""
    descs: dict[str, str] = {}
    for subcats in _load_raw().values():
        for entry in subcats:
            descs[entry["name"]] = entry["description"]
    return descs
