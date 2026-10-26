"""Merged and original datasets must agree on source ROT6D semantics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "marvin_converter", ROOT / "scripts/convert_marvin_lerobot_v3.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name)
        (self.source / "meta").mkdir()
        features = {}
        for name, prefixes in converter.EXPECTED_SOURCE_NAMES.items():
            names = ["unused"] * 14
            for key, count in [("left_hand_prefix", 22), ("right_hand_prefix", 22),
                               ("left_eef_prefix", 9), ("right_eef_prefix", 9)]:
                names.extend(prefixes[key] + str(i) for i in range(count))
            features[name] = {"shape": [76], "names": names}
        features.update({name: {"dtype": "video"} for name in converter.CAMERAS})
        self.write("info.json", dict(codebase_version="v2.1", robot_type="marvin_pro_sharpa_wave",
                   features=features, total_episodes=1, total_frames=2, total_tasks=1))
        self.write("episodes.jsonl", dict(episode_index=0, length=2, task_index=0))
        self.write("tasks.jsonl", dict(task_index=0, task="pick"))

    def write(self, name, value):
        (self.source / "meta" / name).write_text(json.dumps(value) + "\n")

    def test_top_level_merged_layout_is_accepted(self):
        self.write("marvin_conversion.json", {"rotation_6d_layout": converter.SOURCE_ROT6D_LAYOUT})
        info, episodes, _ = converter._validate_source(self.source)
        self.assertEqual(info["total_frames"], 2)
        self.assertEqual(len(episodes), 1)

    def test_original_nested_layout_is_accepted(self):
        self.write("marvin_conversion.json", {"eef_action": {"rotation_6d_layout": converter.SOURCE_ROT6D_LAYOUT}})
        converter._validate_source(self.source)

    def test_conflicting_top_level_and_nested_layouts_are_rejected(self):
        self.write("marvin_conversion.json", {
            "eef_action": {"rotation_6d_layout": converter.SOURCE_ROT6D_LAYOUT},
            "rotation_6d_layout": converter.OPENWAM_ROT6D_LAYOUT,
        })
        with self.assertRaisesRegex(ValueError, "incompatible markers"):
            converter._validate_source(self.source)

    def test_already_column_layout_is_rejected(self):
        self.write("marvin_conversion.json", {"rotation_6d_layout": converter.OPENWAM_ROT6D_LAYOUT})
        with self.assertRaisesRegex(ValueError, "incompatible markers"):
            converter._validate_source(self.source)

    def test_missing_layout_is_still_rejected(self):
        self.write("marvin_conversion.json", {"aggregation": "merged_datasets"})
        with self.assertRaisesRegex(ValueError, "cannot verify"):
            converter._validate_source(self.source)


if __name__ == "__main__":
    unittest.main()
