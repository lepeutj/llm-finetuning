"""Generate reproducible French person/job extraction examples.

Run from the repository root: python -m src.prepare_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("name", "job", "company", "city", "since")

# Names are partitioned before examples are generated. A person's full name
# therefore cannot appear in more than one split.
FIRST_NAMES = (
    "Paul", "Léa", "Karim", "Camille", "Hugo", "Inès", "Thomas", "Sarah",
    "Mehdi", "Chloé", "Louis", "Nora", "Lucas", "Emma", "Yanis", "Manon",
    "Antoine", "Lina", "Gabriel", "Zoé", "Rayan", "Clara", "Nicolas", "Julie",
    "Amine", "Alice", "Maxime", "Sofia", "Raphaël", "Eva", "Adam", "Maya",
    "Bastien", "Jeanne", "Omar", "Élise", "Mathis", "Salomé", "Idriss", "Louise",
    "Samuel", "Aïcha", "Victor", "Noémie",
)
LAST_NAMES = (
    "Martin", "Bernard", "Dubois", "Petit", "Durand", "Leroy", "Moreau",
    "Simon", "Laurent", "Michel", "Lefebvre", "Roux", "Fontaine", "Chevalier",
    "François", "Legrand", "Garnier", "Faure", "Rousseau", "Vincent",
)
JOBS = (
    "ingénieur", "analyste", "développeur", "comptable", "architecte",
    "consultant", "technicien", "chercheur", "designer", "juriste",
    "responsable marketing", "chef de projet", "data scientist", "recruteur",
)
COMPANIES = (
    "Safran", "Airbus", "Thales", "Orange", "SNCF", "Michelin", "Renault",
    "Danone", "Carrefour", "Capgemini", "Dassault Systèmes", "Decathlon",
)
CITIES = (
    "Bordeaux", "Paris", "Lyon", "Toulouse", "Nantes", "Lille",
    "Marseille", "Grenoble", "Rennes", "Strasbourg", "Montpellier", "Nice",
)
TEMPLATES = (
    "{name} est {job} chez {company} à {city} depuis {since}.",
    "Depuis {since}, {name} travaille comme {job} chez {company} à {city}.",
    "À {city}, {name} occupe un poste de {job} chez {company} depuis {since}.",
    "{name} travaille chez {company} à {city} en tant que {job} depuis {since}.",
    "{company} emploie {name} comme {job} à {city} depuis {since}.",
)


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _make_names(rng: random.Random) -> list[str]:
    names = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    rng.shuffle(names)
    return names


def _generate_split(names: list[str], size: int, rng: random.Random) -> list[dict]:
    if size > len(names):
        raise ValueError(f"Split size {size} exceeds {len(names)} available unique names")
    rows = []
    for name in names[:size]:
        answer = {
            "name": name,
            "job": rng.choice(JOBS),
            "company": rng.choice(COMPANIES),
            "city": rng.choice(CITIES),
            "since": rng.randint(2000, 2025),
        }
        rows.append({
            "input": rng.choice(TEMPLATES).format(**answer),
            "output": answer,
        })
    return rows


def prepare(config_path: Path) -> dict[str, int]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["project"]["seed"])
    settings = config["data"]
    rng = random.Random(seed)
    names = _make_names(rng)
    sizes = {split: int(settings[f"{split}_size"]) for split in ("train", "validation", "test")}
    if any(size < 1 for size in sizes.values()):
        raise ValueError("All split sizes must be positive")
    if sum(sizes.values()) > len(names):
        raise ValueError(f"Requested {sum(sizes.values())} examples, but only {len(names)} unique names are available")

    offset = 0
    for split, size in sizes.items():
        rows = _generate_split(names[offset:offset + size], size, rng)
        offset += size
        destination = _resolve(settings[f"{split}_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {destination}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
