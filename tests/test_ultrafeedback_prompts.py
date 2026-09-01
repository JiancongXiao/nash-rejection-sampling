import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from examples.prepare_ultrafeedback_full_train import read_prompt_digests
from examples.prepare_ultrafeedback_prompts import normalize_prompt, prompt_digest


class UltraFeedbackPromptPreparationTest(unittest.TestCase):
    def test_normalization_is_stable(self):
        self.assertEqual(normalize_prompt("  hello  \nworld   \n"), "hello\nworld")
        self.assertEqual(normalize_prompt("\n\t\n"), "")

    def test_digest_uses_normalized_content(self):
        left = prompt_digest(normalize_prompt("hello  \n"))
        right = prompt_digest(normalize_prompt("hello"))
        self.assertEqual(left, right)

    def test_heldout_digest_reader_verifies_recorded_digest(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            prompt = "hello"
            path.write_text(
                json.dumps(
                    {"prompt": prompt, "prompt_sha256": prompt_digest(prompt)}
                )
                + "\n"
            )
            self.assertEqual(read_prompt_digests(path), {prompt_digest(prompt)})

            path.write_text(
                json.dumps({"prompt": prompt, "prompt_sha256": "incorrect"}) + "\n"
            )
            with self.assertRaisesRegex(ValueError, "incorrect prompt digest"):
                read_prompt_digests(path)


if __name__ == "__main__":
    unittest.main()
