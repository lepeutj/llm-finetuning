"""Fast checks that do not download a model."""

import json
import tempfile
import unittest
from pathlib import Path

from src.common import run_directory, selected_config
from src.metrics import parse_prediction, score
from src.prepare_dataset import FIELDS, prepare


class DatasetTests(unittest.TestCase):
    def test_generated_splits_are_disjoint_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {split: root / f"{split}.jsonl" for split in ("train", "validation", "test")}
            settings = {
                "project": {"seed": 7},
                "data": {**{f"{split}_path": str(value) for split, value in paths.items()},
                         "train_size": 20, "validation_size": 5, "test_size": 5},
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
            self.assertTrue(names["train"].isdisjoint(names["validation"]))
            self.assertTrue(names["train"].isdisjoint(names["test"]))
            self.assertTrue(names["validation"].isdisjoint(names["test"]))
            self.assertTrue(all(set(row["output"]) == set(FIELDS) for data in rows.values() for row in data))


class MetricTests(unittest.TestCase):
    def test_strict_json_and_scores(self):
        reference = {"name": "Paul Martin", "job": "ingénieur", "company": "Safran",
                     "city": "Bordeaux", "since": 2021}
        good = json.dumps(reference, ensure_ascii=False)
        wrong_type = dict(reference, since="2021")
        self.assertEqual(parse_prediction(good), reference)
        self.assertIsNone(parse_prediction(json.dumps(wrong_type)))
        result = score([{"reference": reference, "prediction": good},
                        {"reference": reference, "prediction": "not JSON"}])
        self.assertEqual(result["valid_json"], 0.5)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(result["global_f1"], 0.5)


class RunSelectionTests(unittest.TestCase):
    def test_models_and_modes_get_separate_directories(self):
        class Arguments:
            model = None
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


if __name__ == "__main__":
    unittest.main()
