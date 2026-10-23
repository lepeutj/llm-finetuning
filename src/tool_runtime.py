"""Prompting, parsing, scoring, and configuration for tool-calling experiments."""

import hashlib
import json
from pathlib import Path

import jsonschema
import yaml

from .common import lora_target_modules, load_model, quantization_options
from .tool_catalog import TOOL_MAP

ROOT = Path(__file__).absolute().parents[1]
ACTION_TYPES = {"tool_call", "respond", "clarify"}


def add_arguments(parser):
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--quantization", choices=("none", "int8", "nf4", "fp4"))


def settings(args):
    cfg = yaml.safe_load((ROOT / "tool_config.yaml").read_text(encoding="utf-8"))
    if args.model:
        cfg["project"]["model_name"] = args.model
        if not args.revision:
            cfg["project"]["model_revision"] = None
    if args.revision:
        cfg["project"]["model_revision"] = args.revision
    if args.quantization:
        cfg["quantization"]["mode"] = args.quantization
    return cfg


def read_rows(cfg, split):
    path = ROOT / cfg["data"][f"{split}_path"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def identity(cfg):
    result = {"pipeline_version": 1, "task": "tool_calling", "model_name": cfg["project"]["model_name"],
              "model_revision": cfg["project"].get("model_revision"),
              "quantization": cfg["quantization"], "seed": cfg["project"]["seed"]}
    result["data_sha256"] = {split: hashlib.sha256(
        (ROOT / cfg["data"][f"{split}_path"]).read_bytes()).hexdigest()
        for split in ("train", "validation", "test")}
    return result


def run_dir(cfg):
    name = cfg["project"]["model_name"].split("/")[-1].lower().replace(".", "-")
    digest = hashlib.sha256(json.dumps({"identity": identity(cfg), "training": cfg["training"]},
                                      sort_keys=True).encode()).hexdigest()[:10]
    return ROOT / cfg["training"]["results_dir"] / f"{name}-{cfg['quantization']['mode']}-{digest}"


def prompt(row):
    tools = json.dumps(row["tools"], ensure_ascii=False, separators=(",", ":"))
    return ("You are a tool router. Available tools:\n" + tools +
            "\nReturn exactly one JSON object. Use "
            "{\"type\":\"tool_call\",\"name\":...,\"arguments\":{...}} when a tool is needed; "
            "{\"type\":\"clarify\",\"question\":...} when required information is missing; or "
            "{\"type\":\"respond\",\"content\":...} when no tool is needed. Never invent arguments."
            "\n\nUser request:\n" + row["request"])


def messages(row):
    return [{"role": "user", "content": prompt(row)}]


def followup_messages(row):
    call = row["expected_action"]
    result = row["trajectory"][2]["content"]
    return messages(row) + [
        {"role": "assistant", "content": json.dumps(call, ensure_ascii=False)},
        {"role": "user", "content": "Tool result:\n" + json.dumps(result, ensure_ascii=False)},
    ]


def parse_action(text, tools=None):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("type") not in ACTION_TYPES:
        return None
    kind = value["type"]
    if kind == "tool_call":
        if set(value) != {"type", "name", "arguments"} or not isinstance(value["arguments"], dict):
            return None
        catalog = {item["name"]: item for item in tools} if tools else TOOL_MAP
        if value["name"] not in catalog:
            return None
        try:
            jsonschema.validate(value["arguments"], catalog[value["name"]]["parameters"])
        except jsonschema.ValidationError:
            return None
    elif kind == "clarify":
        if set(value) != {"type", "question"} or not isinstance(value["question"], str):
            return None
    elif set(value) != {"type", "content"} or not isinstance(value["content"], str):
        return None
    return value


def score(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot score empty predictions")
    valid = type_ok = tool_ok = args_ok = exact = 0
    calls = no_calls = call_schema_ok = 0
    for row in rows:
        expected = row["reference"]
        if expected["type"] == "tool_call":
            calls += 1
        else:
            no_calls += 1
        predicted = parse_action(row["prediction"], row["tools"])
        if predicted is None:
            continue
        valid += 1
        type_ok += predicted["type"] == expected["type"]
        exact += predicted == expected
        if expected["type"] == "tool_call":
            if predicted["type"] == "tool_call":
                call_schema_ok += 1
                tool_ok += predicted["name"] == expected["name"]
                args_ok += predicted["name"] == expected["name"] and predicted["arguments"] == expected["arguments"]
    return {"count": n, "valid_action": valid / n, "action_type_accuracy": type_ok / n,
            "exact_match": exact / n, "tool_accuracy": tool_ok / calls if calls else None,
            "argument_exact_match": args_ok / calls if calls else None,
            "executable_call_rate": call_schema_ok / calls if calls else None,
            "no_call_accuracy": sum(parse_action(r["prediction"], r["tools"]) is not None and
                                    parse_action(r["prediction"], r["tools"])["type"] == r["reference"]["type"]
                                    for r in rows if r["reference"]["type"] != "tool_call") / no_calls if no_calls else None}


def predict_messages(model, tokenizer, turns, generation):
    import torch
    text = tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=int(generation["max_new_tokens"]),
                                do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def predict_action(model, tokenizer, row, generation):
    return predict_messages(model, tokenizer, messages(row), generation)


__all__ = ["add_arguments", "followup_messages", "identity", "lora_target_modules", "load_model", "messages",
           "parse_action", "predict_action", "predict_messages", "quantization_options", "read_rows", "run_dir", "score", "settings"]
