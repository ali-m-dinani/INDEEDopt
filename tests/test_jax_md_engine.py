import tempfile
import unittest
from pathlib import Path

from ffield_generator.jax_md_engine import write_legacy_results


class JaxMdEngineTest(unittest.TestCase):
    def test_legacy_result_columns(self):
        report = {
            "loss": 2.25,
            "items": [
                {"prediction": -3.125, "target": -1.5, "weight": 0.5},
                {"prediction": 4.75, "target": 5.0, "weight": 2.0},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            write_legacy_results(report, temporary_dir)
            lines = (Path(temporary_dir) / "fort.99").read_text().splitlines(keepends=True)
            self.assertEqual(len(lines), 4)
            for line, item in zip(lines[2:], report["items"]):
                self.assertEqual(len(line), 121)
                self.assertAlmostEqual(float(line[61:72]), item["prediction"])
                self.assertAlmostEqual(float(line[73:84]), item["target"])
                self.assertAlmostEqual(float(line[85:96]), item["weight"])
            self.assertAlmostEqual(
                float((Path(temporary_dir) / "fort.13").read_text()), report["loss"]
            )


if __name__ == "__main__":
    unittest.main()
