"""Generate reproducible single-tool agent trajectories with held-out test wording."""

import argparse
import json
import random
from pathlib import Path

import yaml

from .tool_catalog import TOOLS, execute

ROOT = Path(__file__).absolute().parents[1]

TRAIN_TEMPLATES = {
    "get_weather": ["What is the weather in {city} on {date}?", "Check {city}'s forecast for {date}."],
    "convert_currency": ["Convert {amount} {source} to {target}.", "How much is {amount} {source} in {target}?"],
    "search_web": ["Search the web for {query}.", "Find current information about {query}."],
    "calculate": ["Calculate {expression}.", "What is {expression}?"],
    "find_files": ["Find {pattern} files in {directory}.", "Search {directory} for files matching {pattern}."],
}
TEST_TEMPLATES = {
    "get_weather": ["Could you look up the {date} forecast for {city}?"],
    "convert_currency": ["Exchange {amount} {source} into {target}."],
    "search_web": ["Look online and bring me recent information on {query}."],
    "calculate": ["Work out the value of {expression}."],
    "find_files": ["Locate anything named {pattern} below {directory}."],
}
CITIES = ["Paris", "Berlin", "Dublin", "Madrid", "Montreal", "Tokyo", "Lisbon", "Oslo",
          "Helsinki", "Rome", "Vienna", "Prague", "Warsaw", "Athens", "Brussels", "Zurich",
          "Sydney", "Seoul", "Toronto", "Boston", "Chicago", "Seattle", "Nairobi", "Cairo"]
DATES = ["today", "tomorrow", "next Monday"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
TOPICS = ["small language models", "European AI regulation", "Python", "renewable energy",
          "database security", "space exploration", "medical imaging", "robotics", "climate policy",
          "quantum computing", "electric vehicles", "open source licensing"]
FOCUSES = ["latest news", "recent research", "2026 statistics", "technical documentation",
           "market overview", "safety guidance", "benchmark results", "industry adoption"]
PROJECTS = ["alpha", "atlas", "client-a", "forecast", "research", "archive", "finance", "website"]
SUBDIRS = ["data", "reports", "src", "exports", "documents"]
PATTERNS = ["*.json", "*.csv", "report*", "*.pdf", "invoice*", "*.yaml", "notes*", "*.parquet"]


def action(kind, **values):
    return {"type": kind, **values}


def make_call(rng, name, split):
    date = rng.choice(DATES) if rng.random() < 0.4 else f"2027-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    values = {"city": rng.choice(CITIES), "date": date,
              "amount": round(rng.uniform(5, 5000), 2), "source": rng.choice(CURRENCIES),
              "target": rng.choice(CURRENCIES),
              "query": f"{rng.choice(TOPICS)} {rng.choice(FOCUSES)}",
              "expression": f"{rng.randint(2, 500)} {rng.choice(['+', '-', '*'])} {rng.randint(2, 100)}",
              "directory": f"/workspace/{rng.choice(PROJECTS)}/{rng.choice(SUBDIRS)}",
              "pattern": rng.choice(PATTERNS)}
    while values["source"] == values["target"]:
        values["target"] = rng.choice(CURRENCIES)
    args = {
        "get_weather": {"city": values["city"], "date": values["date"]},
        "convert_currency": {"amount": values["amount"], "from_currency": values["source"],
                             "to_currency": values["target"]},
        "search_web": {"query": values["query"]},
        "calculate": {"expression": values["expression"]},
        "find_files": {"pattern": values["pattern"], "directory": values["directory"]},
    }[name]
    request = rng.choice(TEST_TEMPLATES[name] if split == "test" else TRAIN_TEMPLATES[name]).format(**values)
    expected = action("tool_call", name=name, arguments=args)
    result = execute(name, args)
    final = action("respond", content=f"Tool {name} completed successfully: "
                   f"{json.dumps(result, ensure_ascii=False, sort_keys=True)}")
    return request, expected, result, final, ["tool_call", name]


def make_special(rng, index, split):
    if index % 2:
        topic = rng.choice(TOPICS)
        request = (f"In one sentence, describe the general meaning of {topic}." if split == "test"
                   else f"Give a one-sentence general definition of {topic}.")
        expected = action("respond", content=f"{topic.capitalize()} is a general knowledge topic.")
        return request, expected, None, expected["content"], ["no_tool"]
    variant = rng.randrange(3)
    if variant == 0:
        date = f"2027-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        request = (f"I need the {date} forecast." if split == "test" else f"What is the weather on {date}?")
        question = "Which city should I use?"
    elif variant == 1:
        amount, source = round(rng.uniform(5, 5000), 2), rng.choice(CURRENCIES)
        request = (f"Exchange {amount} {source} for me." if split == "test" else f"Convert {amount} {source}.")
        question = "Which target currency should I use?"
    else:
        pattern = rng.choice(PATTERNS)
        request = (f"Locate {pattern} files." if split == "test" else f"Find files matching {pattern}.")
        question = "Which directory should I search?"
    expected = action("clarify", question=question)
    return request, expected, None, question, ["clarify"]


def build(size, rng, split):
    rows = []
    names = list(TRAIN_TEMPLATES)
    for index in range(size):
        if index % 5 == 0:
            request, expected, result, final, tags = make_special(rng, index, split)
        else:
            request, expected, result, final, tags = make_call(rng, rng.choice(names), split)
        trajectory = [{"role": "user", "content": request},
                      {"role": "assistant", "content": expected}]
        if expected["type"] == "tool_call":
            trajectory.extend([{"role": "tool", "name": expected["name"], "content": result},
                               {"role": "assistant", "content": final}])
        rows.append({"id": f"{split}-{index:05d}", "tools": TOOLS, "request": request,
                     "expected_action": expected, "trajectory": trajectory, "tags": tags})
    rng.shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tool_config.yaml")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rng = random.Random(settings["project"]["seed"])
    for split in ("train", "validation", "test"):
        rows = build(int(settings["data"][f"{split}_size"]), rng, split)
        target = ROOT / settings["data"][f"{split}_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} trajectories -> {target}")


if __name__ == "__main__":
    main()
