"""Tests for interior concept prompt building."""

from __future__ import annotations

import unittest

from app.pipeline.prompts import (
    BASE_PROMPT,
    NEGATIVE_PROMPT,
    ROOM_PROMPTS,
    SHARED_NEGATIVE_PROMPT,
    STYLE_PROMPTS,
    build_prompt,
)


class PromptBuilderTests(unittest.TestCase):
    """Validate room and style prompt generation."""

    def test_build_prompt_returns_room_and_style_specific_prompt(self) -> None:
        """Supported room/style combinations should produce an interior concept prompt."""

        prompt = build_prompt("Living Room", "Modern Luxury")

        self.assertIn(
            BASE_PROMPT.format(room_type="Living Room", style="Modern Luxury"),
            prompt,
        )
        self.assertIn(ROOM_PROMPTS["Living Room"], prompt)
        self.assertIn(STYLE_PROMPTS["Modern Luxury"], prompt)
        self.assertNotIn(SHARED_NEGATIVE_PROMPT, prompt)
        self.assertNotIn(NEGATIVE_PROMPT, prompt)

    def test_build_prompt_removes_strict_structure_preservation_language(self) -> None:
        """The concept prompt should not ask for surface-only strict renovation."""

        prompt = build_prompt("Bedroom", "Japandi").lower()

        forbidden_fragments = [
            "preserve original room layout",
            "preserve wall positions",
            "preserve door positions",
            "surface level renovation",
            "same room structure",
        ]
        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, prompt)

    def test_build_prompt_rejects_unsupported_room_type(self) -> None:
        """Unsupported room types should raise a clear validation error."""

        with self.assertRaisesRegex(ValueError, "Unsupported room type"):
            build_prompt("Dining Room", "Modern Luxury")

    def test_build_prompt_rejects_unsupported_style(self) -> None:
        """Unsupported styles should raise a clear validation error."""

        with self.assertRaisesRegex(ValueError, "Unsupported style"):
            build_prompt("Living Room", "Boho")
