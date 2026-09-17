"""Generate reproducible easy and harder employment-extraction examples.

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
    "Girard", "André", "Mercier", "Blanc", "Guérin", "Boyer", "Duval",
    "Lemoine", "Perrin", "Marchand", "Arnaud", "Renard", "Muller", "Colin",
    "David", "Caron", "Aubert", "Brun", "Picard", "Masson",
)
JOBS = (
    ("ingénieur", "engineer"), ("analyste", "analyst"),
    ("développeur backend", "backend developer"), ("comptable", "accountant"),
    ("architecte", "architect"), ("consultant", "consultant"),
    ("technicien", "technician"), ("chercheur", "researcher"),
    ("designer", "designer"), ("juriste", "legal counsel"),
    ("responsable marketing", "marketing manager"),
    ("chef de projet", "project manager"), ("data scientist", "data scientist"),
    ("recruteur", "recruiter"),
)
COMPANIES = (
    "Safran", "Airbus", "Thales", "Orange", "SNCF", "Michelin", "Renault",
    "Danone", "Carrefour", "Capgemini", "Dassault Systèmes", "Decathlon", "CNRS",
)
CITIES = (
    "Bordeaux", "Paris", "Lyon", "Toulouse", "Nantes", "Lille",
    "Marseille", "Grenoble", "Rennes", "Strasbourg", "Montpellier", "Nice", "Cannes",
)

# The hard test uses different surface templates from training/validation.
# All dates and distractors are generated from the same source facts as the label.
HARD_TEMPLATES = {
    "train": {
        "former": (
            ("fr", "En {old_year}, {name} travaillait chez {old_company} à {old_city}. "
             "Depuis {since}, {company} emploie {name} à {city} comme {job}."),
            ("en", "{name} worked for {old_company} in {old_city} in {old_year}. "
             "In {since}, {name} joined {company} in {city} in the role of {job}."),
        ),
        "relative": (
            ("fr", "{name} a quitté {old_city} en {previous_year}. Deux ans plus tard, "
             "{company} a recruté {name} comme {job} à {city}."),
            ("en", "After leaving {old_city} in {previous_year}, {name} joined "
             "{company} in {city} as {job} two years later."),
        ),
        "missing": (
            ("fr", "{name} a quitté {old_company} à {old_city} en {old_year}. "
             "Aujourd'hui, {name} est {job} chez {company} à {city}."),
            ("en", "{name} left {old_company} in {old_city} in {old_year}. "
             "Currently, {name} works for {company} in {city} as {job}."),
        ),
        "multi": (
            ("fr", "À {old_city}, {name} était chez {old_company} en {old_year}. "
             "Le poste actuel de {name} est {job} chez {company}; son bureau est à {city} "
             "et son arrivée date de {since}."),
            ("en", "{old_company} employed {name} in {old_city} in {old_year}. "
             "{name}'s current role is {job} at {company}. The office is in {city}; "
             "the start year was {since}."),
        ),
    },
    "test": {
        "former": (
            ("fr", "Avant {company}, {name} était chez {old_company} à {old_city} "
             "en {old_year}. À présent, {name} travaille comme {job} à {city} "
             "pour {company} depuis {since}."),
            ("en", "Back in {old_year}, {name} had a position at {old_company} "
             "in {old_city}. Since {since}, the current employer has been {company}: "
             "{name} works in the role of {job} in {city}."),
        ),
        "relative": (
            ("fr", "Après son départ de {old_city} en {previous_year}, {name} "
             "a attendu deux ans avant de rejoindre {company} à {city} "
             "au poste de {job}."),
            ("en", "{name} moved away from {old_city} in {previous_year}. "
             "Two years afterwards, {name} started at {company} as {job}, "
             "based in {city}."),
        ),
        "missing": (
            ("fr", "Le CV de {name} mentionne {old_company} à {old_city} "
             "en {old_year}. Son emploi actuel: {job} chez {company}, à {city}."),
            ("en", "A {old_year} record places {name} at {old_company} in {old_city}. "
             "The present job is {job} for {company}, based in {city}."),
        ),
        "multi": (
            ("fr", "{old_year}: {name} travaillait pour {old_company}, {old_city}. "
             "{since}: entrée chez {company}. Aujourd'hui, {name} exerce comme {job} "
             "au bureau de {city}."),
            ("en", "{name}'s earlier employer was {old_company} ({old_city}, {old_year}). "
             "The current company is {company}; {name} became its {job} in {since} "
             "and works from {city}."),
        ),
    },
}
EASY_TEMPLATES = (
    ("fr", "{name} travaille comme {job} chez {company} à {city} depuis {since}."),
    ("en", "Since {since}, {name} has worked for {company} in {city} as {job}."),
)


def _resolve(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _different(rng: random.Random, values: tuple[str, ...], current: str) -> str:
    return rng.choice([value for value in values if value != current])


def _example(name: str, category: str, rng: random.Random, template_split: str) -> dict:
    templates = EASY_TEMPLATES if category == "easy" else HARD_TEMPLATES[template_split][category]
    language, template = rng.choice(templates)
    job_fr, job_en = rng.choice(JOBS)
    company = rng.choice(COMPANIES)
    city = rng.choice(CITIES)
    year = rng.randint(2007, 2024)
    previous_year = year - 2
    old_year = year - rng.randint(3, 7)
    facts = {
        "name": name, "job": job_fr if language == "fr" else job_en,
        "company": company, "city": city, "since": year,
        "old_company": _different(rng, COMPANIES, company),
        "old_city": _different(rng, CITIES, city),
        "old_year": old_year, "previous_year": previous_year,
    }
    answer = {field: facts[field] for field in FIELDS}
    if category == "missing":
        answer["since"] = None
    tags = [category, language]
    if category != "easy":
        tags.append("hard")
    return {"input": template.format(**facts), "output": answer, "tags": tags}


def _generate_split(names: list[str], rng: random.Random, split: str) -> list[dict]:
    categories = ("former", "relative", "missing", "multi")
    rows = []
    for index, name in enumerate(names):
        category = "easy" if split == "easy_test" or (split != "test" and index % 4 == 0) else categories[index % 4]
        # Rotate the non-easy categories too; every split must contain all four.
        if split in ("train", "validation") and category != "easy":
            category = categories[(index - index // 4 - 1) % 4]
        rows.append(_example(name, category, rng, "test" if split == "test" else "train"))
    rng.shuffle(rows)
    return rows


def prepare(config_path: Path) -> dict[str, int]:
    with config_path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    rng = random.Random(int(settings["project"]["seed"]))
    names = [f"{first} {last}" for first in FIRST_NAMES for last in LAST_NAMES]
    rng.shuffle(names)
    data = settings["data"]
    sizes = {split: int(data[f"{split}_size"]) for split in ("train", "validation", "test", "easy_test")}
    if any(size < 1 for size in sizes.values()):
        raise ValueError("All split sizes must be positive")
    if sum(sizes.values()) > len(names):
        raise ValueError(f"Requested {sum(sizes.values())} unique names but only {len(names)} are available")
    offset = 0
    for split, size in sizes.items():
        rows = _generate_split(names[offset:offset + size], rng, split)
        offset += size
        destination = _resolve(data[f"{split}_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {destination}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    prepare(parser.parse_args().config)


if __name__ == "__main__":
    main()
