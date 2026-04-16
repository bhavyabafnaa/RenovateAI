"""Tests for interior concept prompt building."""

from __future__ import annotations

import unittest

from app.pipeline.prompts import (
    BASE_PROMPT,
    FURNITURE_PRIORITY_SUFFIX,
    INTERIOR_LAYOUT_DIRECTIVE,
    LAYOUT_PROMPT,
    NEGATIVE_PROMPT,
    ROOM_PROMPTS,
    SHARED_NEGATIVE_PROMPT,
    STYLE_PROMPTS,
    WIDE_ANGLE_INTERIOR_PHOTOGRAPHY,
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
        self.assertIn(INTERIOR_LAYOUT_DIRECTIVE, prompt)
        self.assertIn(FURNITURE_PRIORITY_SUFFIX, prompt)
        self.assertIn(WIDE_ANGLE_INTERIOR_PHOTOGRAPHY, prompt)
        self.assertNotIn(LAYOUT_PROMPT, prompt)
        self.assertNotIn(SHARED_NEGATIVE_PROMPT, prompt)
        self.assertNotIn(NEGATIVE_PROMPT, prompt)
        self.assertEqual(
            prompt,
            (
                f"{BASE_PROMPT.format(room_type='Living Room', style='Modern Luxury')}, "
                f"{INTERIOR_LAYOUT_DIRECTIVE}, "
                f"{ROOM_PROMPTS['Living Room']}, "
                f"{STYLE_PROMPTS['Modern Luxury']}, "
                f"{FURNITURE_PRIORITY_SUFFIX}, "
                f"{WIDE_ANGLE_INTERIOR_PHOTOGRAPHY}"
            ),
        )

    def test_build_prompt_orders_directive_before_furniture_and_style(self) -> None:
        """Prompt order should separate scene, directive, furniture, style, and suffix."""

        prompt = build_prompt("Kitchen", "Japandi")

        ordered_fragments = [
            BASE_PROMPT.format(room_type="Kitchen", style="Japandi"),
            INTERIOR_LAYOUT_DIRECTIVE,
            ROOM_PROMPTS["Kitchen"],
            STYLE_PROMPTS["Japandi"],
            FURNITURE_PRIORITY_SUFFIX,
            WIDE_ANGLE_INTERIOR_PHOTOGRAPHY,
        ]

        previous_index = -1
        for fragment in ordered_fragments:
            current_index = prompt.index(fragment)
            with self.subTest(fragment=fragment):
                self.assertGreater(current_index, previous_index)
            previous_index = current_index

    def test_build_prompt_includes_room_specific_staging_tokens(self) -> None:
        """Room prompts should push visible furniture, decor, and lighting into outputs."""

        expected_fragments = {
            "Living Room": [
                "fully furnished",
                "professionally staged",
                "designer sofa",
                "media wall",
                "coffee table",
                "false ceiling design",
                "recessed ceiling lights",
                "curtains",
                "area rug",
                "indoor plants",
            ],
            "Bedroom": [
                "fully furnished",
                "professionally staged",
                "beautiful bed",
                "upholstered headboard",
                "full-height wardrobe",
                "study table",
                "dressing table",
                "false ceiling design",
                "elegant bedding",
            ],
            "Kitchen": [
                "fully furnished",
                "professionally staged",
                "modular kitchen cabinets",
                "upper and lower cabinets",
                "premium countertop",
                "organized utensils",
                "island counter",
                "under-cabinet lighting",
                "dining table if layout supports it",
            ],
        }

        for room_type, fragments in expected_fragments.items():
            prompt = build_prompt(room_type, "Modern Luxury")
            for fragment in fragments:
                with self.subTest(room_type=room_type, fragment=fragment):
                    self.assertIn(fragment, prompt)

    def test_build_prompt_removes_strict_structure_preservation_language(self) -> None:
        """The concept prompt should not ask for surface-only strict renovation."""

        prompt = build_prompt("Bedroom", "Japandi").lower()

        forbidden_fragments = [
            "preserve original room layout",
            "preserve wall positions",
            "preserve door positions",
            "surface level renovation",
            "surface-level renovation only",
            "same room structure",
        ]
        for fragment in forbidden_fragments:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, prompt)

    def test_negative_prompt_discourages_empty_unfurnished_outputs(self) -> None:
        """The negative prompt should explicitly reject empty or plain rooms."""

        for fragment in [
            "empty room",
            "unfurnished room",
            "minimal furniture",
            "blank interior",
            "bare interior",
            "bare space",
            "unfinished room",
        ]:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, NEGATIVE_PROMPT)

    def test_build_prompt_rejects_unsupported_room_type(self) -> None:
        """Unsupported room types should raise a clear validation error."""

        with self.assertRaisesRegex(ValueError, "Unsupported room type"):
            build_prompt("Dining Room", "Modern Luxury")

    def test_build_prompt_rejects_unsupported_style(self) -> None:
        """Unsupported styles should raise a clear validation error."""

        with self.assertRaisesRegex(ValueError, "Unsupported style"):
            build_prompt("Living Room", "Boho")
