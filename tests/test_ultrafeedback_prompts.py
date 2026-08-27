import unittest

from examples.prepare_ultrafeedback_prompts import normalize_prompt, prompt_digest


class UltraFeedbackPromptPreparationTest(unittest.TestCase):
    def test_normalization_is_stable(self):
        self.assertEqual(normalize_prompt("  hello  \nworld   \n"), "hello\nworld")
        self.assertEqual(normalize_prompt("\n\t\n"), "")

    def test_digest_uses_normalized_content(self):
        left = prompt_digest(normalize_prompt("hello  \n"))
        right = prompt_digest(normalize_prompt("hello"))
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
