"""Tests for renovation prompt building."""

from __future__ import annotations

import unittest

from app.pipeline.prompts import BASE_PROMPT, NEGATIVE_PROMPT, SHARED_NEGATIVE_PROMPT, build_prompt


class PromptBuilderTests(unittest.TestCase):
    """Validate style prompt generation."""

    def test_build_prompt_returns_style_specific_prompt(self) -> None:
        """Each supported style should produce a focused renovation prompt."""

        expected_fragments = {
            "modern": "premium stone finishes",
            "scandinavian": "warm neutral tones",
            "industrial": "textured materials",
        }

        for style, fragment in expected_fragments.items():
            with self.subTest(style=style):
                prompt = build_prompt(style)
                self.assertIn(BASE_PROMPT, prompt)
                self.assertIn(fragment, prompt)
                self.assertNotIn(SHARED_NEGATIVE_PROMPT, prompt)
                self.assertNotIn(NEGATIVE_PROMPT, prompt)

    def test_build_prompt_rejects_unsupported_style(self) -> None:
        """Unsupported styles should raise a clear validation error."""

        with self.assertRaisesRegex(ValueError, "Unsupported style"):
            build_prompt("boho")
