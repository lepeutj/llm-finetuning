"""Strict JSON extraction metrics used for both model variants."""

import json

from .common import FIELDS


def parse_prediction(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != set(FIELDS):
        return None
    if any(not isinstance(value[key], str) for key in FIELDS if key != "since"):
        return None
    if type(value["since"]) is not int:
        return None
    return value


def score(rows):
    if not rows:
        raise ValueError("Cannot score an empty prediction set")
    valid = exact = 0
    matches = {field: 0 for field in FIELDS}
    for row in rows:
        predicted = parse_prediction(row["prediction"])
        if predicted is None:
            continue
        valid += 1
        reference = row["reference"]
        exact += predicted == reference
        for field in FIELDS:
            matches[field] += predicted[field] == reference[field]
    n = len(rows)
    # Each field has one gold and one expected prediction per example. A wrong or
    # missing value counts as both a false positive and a false negative, so
    # micro F1 equals exact field accuracy over the full test set.
    return {
        "count": n,
        "valid_json": valid / n,
        "exact_match": exact / n,
        "field_f1": {field: matches[field] / n for field in FIELDS},
        "global_f1": sum(matches.values()) / (n * len(FIELDS)),
    }
