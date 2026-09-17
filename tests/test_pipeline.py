"""Fast checks that do not download a model."""

import json
import re
import tempfile
import unittest
import argparse
from pathlib import Path

from src.common import lora_target_modules, messages, run_directory, selected_config
from src.metrics import parse_prediction, score
from src.evaluate import _baseline
from src.prepare_dataset import FIELDS, HARD_TEMPLATES, prepare


class DatasetTests(unittest.TestCase):
    def test_generated_splits_are_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test", "easy_test")}
            settings = {
                "project": {"seed": 7},
                "data": {**{f"{split}_path": str(value) for split, value in paths.items()},
                         "train_size": 20, "validation_size": 8, "test_size": 12,
                         "easy_test_size": 4},
            }
            config_path = root / "config.yaml"
            import yaml
            config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
            prepare(config_path)
            first = {split: value.read_bytes() for split, value in paths.items()}
            prepare(config_path)
            self.assertEqual(first, {split: value.read_bytes() for split, value in paths.items()})
            rows = {split: [json.loads(line) for line in value.read_text(encoding="utf-8").splitlines()]
                    for split, value in paths.items()}
            names = {split: {row["output"]["name"] for row in data} for split, data in rows.items()}
            for left in names:
                for right in names:
                    if left != right:
                        self.assertTrue(names[left].isdisjoint(names[right]))
            self.assertTrue(all(set(row["output"]) == set(FIELDS) for data in rows.values() for row in data))
            self.assertTrue(all("easy" in row["tags"] for row in rows["easy_test"]))
            self.assertTrue(all("hard" in row["tags"] for row in rows["test"]))
            self.assertEqual({"former", "relative", "missing", "multi"},
                             {row["tags"][0] for row in rows["test"]})
            for row in rows["test"]:
                if "missing" in row["tags"]:
                    self.assertIsNone(row["output"]["since"])
                    self.assertRegex(row["input"], r"\d{4}")  # Old year is a distractor.
                if "relative" in row["tags"]:
                    old_year = int(re.search(r"\d{4}", row["input"]).group())
                    self.assertEqual(row["output"]["since"], old_year + 2)

    def test_hard_test_templates_are_held_out(self):
        train = {template for variants in HARD_TEMPLATES["train"].values() for _, template in variants}
        test = {template for variants in HARD_TEMPLATES["test"].values() for _, template in variants}
        self.assertTrue(train.isdisjoint(test))


class MetricTests(unittest.TestCase):
    def test_strict_json_and_scores(self):
        reference = {"name": "Paul Martin", "job": "ingénieur", "company": "Safran",
                     "city": "Bordeaux", "since": 2021}
        good = json.dumps(reference, ensure_ascii=False)
        wrong_type = dict(reference, since="2021")
        self.assertEqual(parse_prediction(good), reference)
        self.assertIsNone(parse_prediction(json.dumps(wrong_type)))
        self.assertIsNotNone(parse_prediction(json.dumps(dict(reference, since=None))))
        result = score([{"reference": reference, "prediction": good},
                        {"reference": reference, "prediction": "not JSON"}])
        self.assertEqual(result["valid_json"], 0.5)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(result["global_f1"], 0.5)

    def test_missing_year_scoring(self):
        reference = {"name": "Sophie Martin", "job": "researcher", "company": "CNRS",
                     "city": "Grenoble", "since": None}
        result = score([{"reference": reference, "prediction": json.dumps(reference)}])
        self.assertEqual(result["since_missing_accuracy"], 1.0)
        self.assertEqual(result["since_missing_count"], 1)

    def test_few_shot_messages_preserve_example_and_query(self):
        example = {"input": "Sample sentence", "output": {"since": None}}
        turns = messages("New sentence", [example])
        self.assertEqual([turn["role"] for turn in turns], ["user", "assistant", "user"])
        self.assertIn("null", turns[1]["content"])
        self.assertIn("New sentence", turns[-1]["content"])


class RunSelectionTests(unittest.TestCase):
    def test_partial_baseline_requires_explicit_smoke_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"model_name": "test"}
            (root / "zero_hard_manifest.json").write_text(json.dumps(expected), encoding="utf-8")
            one = {"input": "one", "output": {"since": None}, "tags": ["missing"]}
            (root / "zero_hard_predictions.jsonl").write_text(
                json.dumps({"input": "one", "reference": one["output"],
                            "tags": one["tags"], "prediction": "{}"}) + "\n", encoding="utf-8"
            )
            test = [one, {"input": "two", "output": {}, "tags": []}]
            with self.assertRaises(SystemExit):
                _baseline(argparse.ArgumentParser(), root, "zero", "hard", expected, test, False)
            self.assertEqual(len(_baseline(argparse.ArgumentParser(), root, "zero", "hard",
                                           expected, test, True)), 1)

    def test_lora_target_configuration(self):
        self.assertEqual(lora_target_modules(["q_proj", "v_proj"]), ["q_proj", "v_proj"])
        self.assertEqual(lora_target_modules("all-linear"), "all-linear")
        with self.assertRaises(ValueError):
            lora_target_modules("q_proj")

    def test_models_and_modes_get_separate_directories(self):
        class Arguments:
            model = None
            revision = None
            quantization = None

        default = selected_config(Arguments())
        original_dir = run_directory(default)
        another = selected_config(Arguments())
        another["project"]["model_name"] = "Qwen/Qwen2.5-1.5B-Instruct"
        self.assertNotEqual(original_dir, run_directory(another))
        another["project"]["model_name"] = default["project"]["model_name"]
        another["quantization"]["mode"] = "int8"
        self.assertNotEqual(original_dir, run_directory(another))
        another = selected_config(Arguments())
        another["training"]["learning_rate"] = 0.0001
        self.assertNotEqual(original_dir, run_directory(another))
        another = selected_config(Arguments())
        another["project"]["model_revision"] = "specific-commit"
        self.assertNotEqual(original_dir, run_directory(another))


if __name__ == "__main__":
    unittest.main()
