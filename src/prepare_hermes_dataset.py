"""Download, strictly normalize, split, and balance Hermes single-turn tool calls."""
import argparse
import ast
import json
import random
import re
from pathlib import Path

import yaml
import jsonschema

from .prepare_tool_dataset import build

ROOT = Path(__file__).absolute().parents[1]
CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def structured(value):
    if not isinstance(value, str):
        return value
    value = value.strip()
    # Some source rows contain the two literal characters ``\n`` around the
    # serialized object rather than actual whitespace.
    while value.startswith("\\n"):
        value = value[2:].lstrip()
    while value.endswith("\\n"):
        value = value[:-2].rstrip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(value)
        except (ValueError, SyntaxError):
            pass
    raise ValueError("not JSON or a Python literal")


def normalize_tools(raw):
    items = structured(raw)
    if not isinstance(items, list) or not items:
        raise ValueError("tools must be a nonempty list")
    result = []
    for item in items:
        function = item.get("function", item) if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("invalid tool definition")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ValueError("invalid parameter schema")
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        result.append({"name": function["name"], "description": function.get("description", ""),
                       "parameters": parameters})
    return result


def normalize_row(row):
    turns = row.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("missing conversations")
    user = next((turn.get("value") for turn in turns if turn.get("from") in ("human", "user")), None)
    assistant = next((turn.get("value") for turn in turns if turn.get("from") in ("gpt", "assistant")), None)
    if not isinstance(user, str) or not isinstance(assistant, str):
        raise ValueError("missing user or assistant turn")
    calls = CALL_PATTERN.findall(assistant)
    if len(calls) != 1:
        raise ValueError("requires exactly one tool call")
    call = structured(calls[0])
    repaired = False
    # The source dataset documents a block where `name` was accidentally put
    # inside `arguments`. Repair only that exact, unambiguous shape.
    if isinstance(call, dict) and set(call) == {"arguments"} and isinstance(call["arguments"], dict):
        nested = dict(call["arguments"])
        if isinstance(nested.get("name"), str):
            call = {"name": nested.pop("name"), "arguments": nested}
            repaired = True
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("invalid call shape")
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        raise ValueError("invalid call values")
    tools = normalize_tools(row.get("tools"))
    names = {tool["name"] for tool in tools}
    if call["name"] not in names:
        raise ValueError("called tool is absent from definitions")
    schema = next(tool["parameters"] for tool in tools if tool["name"] == call["name"])
    try:
        jsonschema.validate(call["arguments"], schema)
    except (jsonschema.SchemaError, jsonschema.ValidationError) as error:
        raise ValueError("arguments do not match the tool schema") from error
    expected = {"type": "tool_call", "name": call["name"], "arguments": call["arguments"]}
    tags = ["tool_call", "hermes", row.get("category") or "uncategorized"]
    if repaired:
        tags.append("source_shape_repaired")
    return {"id": row.get("id"), "tools": tools, "request": user, "expected_action": expected,
            "trajectory": [{"role": "user", "content": user},
                           {"role": "assistant", "content": expected}],
            "tags": tags}


def negatives(count, rng, split):
    candidates = build(count * 6, rng, split)
    result = [row for row in candidates if row["expected_action"]["type"] != "tool_call"]
    if len(result) < count:
        raise RuntimeError("could not generate enough negative examples")
    for index, row in enumerate(result[:count]):
        row["id"] = f"{split}-negative-{index:05d}"
        row["tags"].append("synthetic_balance")
    return result[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "hermes_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from datasets import load_dataset
    data_cfg = cfg["data"]
    source = load_dataset(data_cfg["dataset_name"], data_cfg["dataset_subset"],
                          split="train", revision=data_cfg["dataset_revision"])
    accepted, rejected = [], 0
    for raw in source:
        try:
            accepted.append(normalize_row(raw))
        except (KeyError, TypeError, ValueError):
            rejected += 1
    rng = random.Random(cfg["project"]["seed"])
    # Keep one row per tool name so every evaluation tool is unseen during
    # training. This measures schema-guided generalization rather than tool-name
    # memorization.
    rng.shuffle(accepted)
    distinct = {}
    for row in accepted:
        distinct.setdefault(row["expected_action"]["name"], row)
    tool_disjoint = list(distinct.values()); rng.shuffle(tool_disjoint)
    plan = {"train": (300, 200), "validation": (50, 50), "test": (90, 60)}
    required = sum(calls for calls, _ in plan.values())
    if len(tool_disjoint) < required:
        raise RuntimeError(f"Only {len(tool_disjoint)} distinct valid tools; {required} required")
    offset = 0
    split_tools = {}
    for split, (call_count, negative_count) in plan.items():
        rows = tool_disjoint[offset:offset + call_count]; offset += call_count
        split_tools[split] = {row["expected_action"]["name"] for row in rows}
        rows.extend(negatives(negative_count, rng, split)); rng.shuffle(rows)
        target = ROOT / cfg["data"][f"{split}_path"]; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows ({call_count} Hermes calls, {negative_count} balanced negatives)")
    if any(split_tools[left] & split_tools[right] for left in split_tools for right in split_tools
           if left < right):
        raise RuntimeError("Tool names overlap across splits")
    (ROOT / "data/hermes/import_report.json").write_text(json.dumps({
        "source": data_cfg["dataset_name"], "subset": data_cfg["dataset_subset"],
        "revision": data_cfg["dataset_revision"],
        "license": "Apache-2.0", "source_rows": len(source), "accepted_rows": len(accepted),
        "rejected_rows": rejected, "distinct_valid_tools": len(distinct),
        "tool_name_overlap_across_splits": 0,
        "split_tool_counts": {key: len(value) for key, value in split_tools.items()},
        "seed": cfg["project"]["seed"]}, indent=2) + "\n", encoding="utf-8")
    print(f"Accepted {len(accepted)}/{len(source)} source rows; rejected {rejected} malformed or unsupported rows")


if __name__ == "__main__":
    main()
