from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from controller import LILACSharedAutonomyController
from data import build_training_arrays
from language import (
    CanonicalLanguageIndex,
    CanonicalLanguageDataset,
    CanonicalUtterance,
    DatasetAlphaLabeler,
    GeminiUtteranceSelector,
    LanguageStack,
)
from lilac_model import LILACModel


def make_dataset():
    return CanonicalLanguageDataset([
        CanonicalUtterance(
            "pick_up_cup_pour_into_bowl",
            "Pick up the cup and pour water into the bowl.",
            "instruction",
            1.0,
            aliases=(
                "Pick up the cup, move it to the left bowl, and pour water into the bowl.",
                "move the cup to the left bowl and pour",
                "pour water from the cup into the bowl",
            ),
        ),
        CanonicalUtterance(
            "pick_up_remote_put_in_box",
            "pick up remote controller put in box",
            "instruction",
            1.0,
            aliases=(
                "pick up the remote controller and put it in the box",
                "remote controller to box",
            ),
        ),
        CanonicalUtterance("right", "right", "correction", 0.0),
        CanonicalUtterance(
            "pour_water",
            "pour water",
            "correction",
            0.0,
            aliases=("pour", "tilt the cup", "start pouring"),
        ),
        CanonicalUtterance("left", "left", "correction", 0.0),
        CanonicalUtterance("down", "down", "correction", 0.0),
        CanonicalUtterance("up", "up", "correction", 0.0),
        CanonicalUtterance("front", "front", "correction", 0.0, aliases=("forward",)),
        CanonicalUtterance("back", "back", "correction", 0.0, aliases=("backward",)),
    ])


class LanguageControllerTest(unittest.TestCase):
    def test_canonical_dataset_rejects_duplicates_and_labels_alpha(self):
        dataset = make_dataset()
        labeler = DatasetAlphaLabeler(dataset)

        self.assertEqual(labeler("pick_up_cup_pour_into_bowl"), 1.0)
        self.assertEqual(labeler("move the cup to the left bowl and pour"), 1.0)
        self.assertEqual(labeler("up!"), 0.0)
        self.assertEqual(labeler("Pour water"), 0.0)
        self.assertEqual(dataset.get("tilt the cup").id, "pour_water")
        self.assertEqual(labeler("forward"), 0.0)
        self.assertEqual(dataset.get("backward").id, "back")

        with self.assertRaises(ValueError):
            CanonicalLanguageDataset([
                CanonicalUtterance("up", "up", "correction", 0.0),
                CanonicalUtterance("up", "move up", "correction", 0.0),
            ])

    def test_canonical_dataset_loads_list_payload(self):
        path = PROJECT_DIR / "tests" / "_tmp_language_payload.json"
        try:
            path.write_text(
                '[{"id": "up", "text": "up", "kind": "correction", "alpha": 0.0}]',
                encoding="utf-8",
            )
            dataset = CanonicalLanguageDataset.load(path)
            self.assertEqual(dataset.get("up").id, "up")
        finally:
            if path.exists():
                path.unlink()

    def test_gemini_selector_exact_match_and_missing_key_error(self):
        selector = GeminiUtteranceSelector(make_dataset(), api_key=None)

        self.assertEqual(selector.select("up!").text, "up")
        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
            selector.select("move upward please")

    def test_gemini_selector_rejects_invalid_model_response(self):
        class FakeModels:
            def generate_content(self, model, contents):
                return type("Response", (), {"text": "sideways"})()

        class FakeClient:
            models = FakeModels()

        selector = GeminiUtteranceSelector(make_dataset(), client=FakeClient())
        with self.assertRaisesRegex(RuntimeError, "unknown canonical id"):
            selector.select("move somewhere")

    def test_gemini_selector_maps_noisy_right_input_to_right(self):
        class FakeModels:
            def generate_content(self, model, contents):
                self.last_prompt = contents
                return type("Response", (), {"text": "right"})()

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        selector = GeminiUtteranceSelector(make_dataset(), client=FakeClient())
        self.assertEqual(selector.select("het go to the right").text, "right")

    def test_gemini_selector_filters_candidates_by_command_kind(self):
        class FakeModels:
            def __init__(self):
                self.last_prompt = ""

            def generate_content(self, model, contents):
                self.last_prompt = contents
                return type("Response", (), {"text": "pick_up_cup_pour_into_bowl"})()

        class FakeClient:
            def __init__(self):
                self.models = FakeModels()

        client = FakeClient()
        selector = GeminiUtteranceSelector(make_dataset(), client=client)
        entry = selector.select("grab cup and pour it", kind="instruction")

        self.assertEqual(entry.id, "pick_up_cup_pour_into_bowl")
        self.assertIn('"kind": "instruction"', client.models.last_prompt)
        self.assertNotIn('"id": "pour_water"', client.models.last_prompt)

    def test_gemini_selector_rejects_wrong_kind_response(self):
        class FakeModels:
            def generate_content(self, model, contents):
                return type("Response", (), {"text": "pour_water"})()

        class FakeClient:
            models = FakeModels()

        selector = GeminiUtteranceSelector(make_dataset(), client=FakeClient())
        with self.assertRaisesRegex(RuntimeError, "unknown canonical id"):
            selector.select("pour it", kind="instruction")

    def test_gemini_prompt_includes_front_and_back_candidates(self):
        prompt = GeminiUtteranceSelector(make_dataset(), client=object())._build_prompt(
            "go forward",
            kind="correction",
        )
        self.assertIn('"id": "front"', prompt)
        self.assertIn('"id": "back"', prompt)
        self.assertNotIn('"kind": "instruction"', prompt)

    def test_language_stack_lifo(self):
        stack = LanguageStack()
        stack.set_instruction("Pick up the cup and pour water into the bowl.")
        stack.push("up")
        stack.push("left")

        self.assertEqual(stack.active(), "left")
        self.assertEqual(stack.pop(), "left")
        self.assertEqual(stack.active(), "up")
        stack.clear()
        self.assertEqual(stack.active(), "Pick up the cup and pour water into the bowl.")

    def test_controller_without_model_does_not_apply_fallback_motion(self):
        dataset = make_dataset()
        controller = LILACSharedAutonomyController(
            model=None,
            language_index=None,
            language_dataset=dataset,
            utterance_selector=GeminiUtteranceSelector(dataset, api_key=None),
        )
        controller.set_instruction("Pick up the cup and pour water into the bowl.")

        T_curr = np.eye(4)
        T_next, info = controller.safe_update_target(T_curr, np.zeros(16), np.zeros(2))

        np.testing.assert_allclose(T_next, T_curr)
        np.testing.assert_allclose(info["action"], np.zeros(6))
        self.assertEqual(info["source"], "error")
        self.assertIn("fallback has been removed", info["error"])

    def test_controller_apply_utterance_uses_local_exact_match_before_gemini(self):
        dataset = make_dataset()
        controller = LILACSharedAutonomyController(
            model=None,
            language_index=None,
            language_dataset=dataset,
            utterance_selector=GeminiUtteranceSelector(dataset, api_key=None),
        )

        entry, event_type = controller.apply_utterance("remote controller to box")
        self.assertEqual(entry.id, "pick_up_remote_put_in_box")
        self.assertEqual(event_type, "instruction")
        self.assertEqual(controller.active_utterance(), "pick up remote controller put in box")

        entry, event_type = controller.apply_utterance("right")
        self.assertEqual(entry.id, "right")
        self.assertEqual(event_type, "push")
        self.assertEqual(controller.active_utterance(), "right")

        with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
            controller.apply_utterance("move somewhere novel")

    def test_canonical_language_index_save_load_and_model_shapes(self):
        class FakeEmbedder:
            def encode(self, utterance):
                vec = np.zeros(768, dtype=np.float32)
                vec[len(str(utterance)) % 768] = 1.0
                return vec

        dataset = make_dataset()
        index = CanonicalLanguageIndex.build(dataset, FakeEmbedder())
        path = PROJECT_DIR / "tests" / "_tmp_language_index.npz"
        try:
            index.save(path)
            loaded = CanonicalLanguageIndex.load(path)
            self.assertEqual(loaded.lookup("up")["id"], "up")
            self.assertEqual(loaded.lookup("pick_up_cup_pour_into_bowl")["embedding"].shape, (768,))
        finally:
            if path.exists():
                path.unlink()

        import torch

        model = LILACModel(state_dim=16, language_dim=768, action_dim=6, latent_dim=2, hidden_dim=128)
        model.eval()
        state = torch.zeros(2, 16)
        language = torch.zeros(2, 768)
        alpha = torch.ones(2)
        z = torch.zeros(2, 2)
        action = model.decoder(state, language, alpha, z)
        self.assertEqual(tuple(action.shape), (2, 6))

    def test_training_arrays_use_gt_canonical_utterances_without_gemini(self):
        dataset = make_dataset()
        episode_path = PROJECT_DIR / "tests" / "_tmp_episode.npz"
        meta_path = PROJECT_DIR / "tests" / "_tmp_episode.json"
        try:
            np.savez_compressed(
                episode_path,
                q_arm=np.zeros((2, 7), dtype=np.float64),
                ee_pose=np.zeros((2, 6), dtype=np.float64),
                active_utterance=np.asarray(["right", "right"], dtype=object),
            )
            meta_path.write_text(
                '{"instruction": "Pick up the cup and pour water into the bowl.", "episode_id": "tmp"}',
                encoding="utf-8",
            )
            arrays = build_training_arrays(
                [episode_path],
                language_dataset=dataset,
            )
            self.assertEqual(arrays["utterances"].tolist(), ["right"])
            self.assertEqual(arrays["alphas"].tolist(), [0.0])

            np.savez_compressed(
                episode_path,
                q_arm=np.zeros((2, 7), dtype=np.float64),
                ee_pose=np.zeros((2, 6), dtype=np.float64),
                active_utterance=np.asarray(["het go to the right", "het go to the right"], dtype=object),
            )
            with self.assertRaisesRegex(ValueError, "Gemini canonicalization is used only at deployment time"):
                build_training_arrays([episode_path], language_dataset=dataset)
        finally:
            if episode_path.exists():
                episode_path.unlink()
            if meta_path.exists():
                meta_path.unlink()


if __name__ == "__main__":
    unittest.main()
