"""Tests for the Diffusers generator wrapper."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app.pipeline.generator import ControlNetSDXLGenerator, InferenceConfig, RuntimeConfig


class _FakeTorch:
    """Minimal torch stand-in for unit testing the orchestration logic."""

    float16 = "float16"
    float32 = "float32"

    class cuda:
        @staticmethod
        def is_available():
            return False

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeCudaTorch(_FakeTorch):
    """Torch stand-in with CUDA enabled for pipeline load tests."""

    class cuda:
        @staticmethod
        def is_available():
            return True


class _FakePipeline:
    """Callable pipeline stub that records the generation inputs."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None
        self.to_device: str | None = None
        self.progress_bar_disabled: bool | None = None
        self.attention_slicing_enabled = False
        self.vae_slicing_enabled = False
        self.vae_tiling_enabled = False

    def enable_attention_slicing(self) -> None:
        self.attention_slicing_enabled = True

    def enable_vae_slicing(self) -> None:
        self.vae_slicing_enabled = True

    def enable_vae_tiling(self) -> None:
        self.vae_tiling_enabled = True

    def to(self, device: str):
        self.to_device = device
        return self

    def set_progress_bar_config(self, disable: bool) -> None:
        self.progress_bar_disabled = disable

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return type("PipelineOutput", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]), "gray")]})()


class _BlankPipeline(_FakePipeline):
    """Pipeline stub that simulates a blank generation result."""

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return type("PipelineOutput", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]), "black")]})()


class _FakeControlNetModel:
    """ControlNet stub that records load kwargs."""

    last_kwargs: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_kwargs = {"args": args, "kwargs": kwargs}
        return cls()


class _FakeAutoencoderKL:
    """VAE stub that records load kwargs."""

    last_kwargs: dict[str, object] | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_kwargs = {"args": args, "kwargs": kwargs}
        return cls()


class _FakeImg2ImgPipelineFactory:
    """Pipeline factory stub that records pipeline construction kwargs."""

    last_kwargs: dict[str, object] | None = None
    last_instance: _FakePipeline | None = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.last_kwargs = {"args": args, "kwargs": kwargs}
        cls.last_instance = _FakePipeline()
        return cls.last_instance


class GeneratorTests(unittest.TestCase):
    """Verify the generator produces a real output path from the new inputs."""

    def test_load_pipeline_falls_back_to_cpu_with_float32(self) -> None:
        """The pipeline should use float32 and `pipe.to(\"cpu\")` when CUDA is unavailable."""

        generator = ControlNetSDXLGenerator()
        _FakeAutoencoderKL.last_kwargs = None

        with patch.object(
            generator,
            "_import_runtime_dependencies",
            return_value=(_FakeTorch, _FakeControlNetModel, _FakeImg2ImgPipelineFactory, _FakeAutoencoderKL),
        ):
            pipeline = generator.load_pipeline()

        self.assertIsNotNone(pipeline)
        self.assertEqual(_FakeControlNetModel.last_kwargs["kwargs"]["torch_dtype"], "float32")
        self.assertIsNone(_FakeControlNetModel.last_kwargs["kwargs"]["variant"])
        self.assertEqual(_FakeImg2ImgPipelineFactory.last_kwargs["kwargs"]["torch_dtype"], "float32")
        self.assertIsNone(_FakeImg2ImgPipelineFactory.last_kwargs["kwargs"]["variant"])
        self.assertEqual(_FakeImg2ImgPipelineFactory.last_instance.to_device, "cpu")
        self.assertTrue(_FakeImg2ImgPipelineFactory.last_instance.attention_slicing_enabled)
        self.assertIsNotNone(_FakeAutoencoderKL.last_kwargs)
        self.assertEqual(_FakeAutoencoderKL.last_kwargs["kwargs"]["torch_dtype"], "float32")

    def test_load_pipeline_uses_fp16_safe_vae_on_cuda(self) -> None:
        """The CUDA path should load the fp16-safe SDXL VAE and wire it into the pipeline."""

        generator = ControlNetSDXLGenerator()
        _FakeAutoencoderKL.last_kwargs = None

        with self.assertLogs("app.pipeline.generator", level="INFO") as captured_logs:
            with patch.object(
                generator,
                "_import_runtime_dependencies",
                return_value=(_FakeCudaTorch, _FakeControlNetModel, _FakeImg2ImgPipelineFactory, _FakeAutoencoderKL),
            ):
                pipeline = generator.load_pipeline()

        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(_FakeAutoencoderKL.last_kwargs)
        self.assertEqual(_FakeAutoencoderKL.last_kwargs["args"][0], generator.model_config.vae_model_id)
        self.assertEqual(_FakeAutoencoderKL.last_kwargs["kwargs"]["torch_dtype"], "float16")
        self.assertEqual(
            _FakeImg2ImgPipelineFactory.last_kwargs["kwargs"]["vae"].__class__,
            _FakeAutoencoderKL,
        )
        self.assertEqual(_FakeImg2ImgPipelineFactory.last_instance.to_device, "cuda")
        self.assertTrue(_FakeImg2ImgPipelineFactory.last_instance.vae_slicing_enabled)
        self.assertTrue(_FakeImg2ImgPipelineFactory.last_instance.vae_tiling_enabled)
        log_output = "\n".join(captured_logs.output)
        self.assertIn("Selected pipeline class", log_output)
        self.assertIn("VAE model ID", log_output)

    @patch.dict("os.environ", {"RENOVATEAI_DEVICE": "gpu", "RENOVATEAI_REQUIRE_CUDA": "1"})
    def test_runtime_config_reads_gpu_environment(self) -> None:
        """RuntimeConfig should support explicit GPU configuration from env vars."""

        runtime_config = RuntimeConfig()

        self.assertEqual(runtime_config.device, "cuda")
        self.assertTrue(runtime_config.require_cuda)

    def test_resolve_device_raises_when_cuda_is_required_but_missing(self) -> None:
        """Requiring CUDA should fail clearly instead of silently falling back to CPU."""

        generator = ControlNetSDXLGenerator(
            runtime_config=RuntimeConfig(device="cuda", require_cuda=True)
        )

        with self.assertRaisesRegex(RuntimeError, "CUDA GPU required"):
            generator._resolve_device(_FakeTorch)

    def test_generate_uses_input_image_and_edge_map_and_writes_new_output(self) -> None:
        """The generator should use img2img inputs and save a new file under the output directory."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_image_path = temp_path / "room.png"
            edge_map_path = temp_path / "room_edges.png"
            output_dir = temp_path / "outputs"

            Image.new("RGB", (640, 480), "white").save(input_image_path)
            Image.new("RGB", (640, 480), "black").save(edge_map_path)

            fake_pipeline = _FakePipeline()
            generator = ControlNetSDXLGenerator(
                inference_config=InferenceConfig(output_dir=output_dir, target_long_side=512)
            )

            with self.assertLogs("app.pipeline.generator", level="INFO") as captured_logs:
                with patch.object(generator, "load_pipeline", return_value=fake_pipeline):
                    with patch.object(
                        generator,
                        "_import_runtime_dependencies",
                        return_value=(_FakeTorch, None, None, None),
                    ):
                        with patch.object(generator, "_resolve_device", return_value="cpu"):
                            result = generator.generate(
                                input_image_path=input_image_path,
                                edge_map_path=edge_map_path,
                                room_type="Living Room",
                                style="Modern Luxury",
                            )
                            self.assertTrue(result.output_image_path.exists())
                            self.assertEqual(result.input_image_path, input_image_path.resolve())
                            self.assertEqual(result.edge_map_path, edge_map_path.resolve())
                            self.assertNotEqual(result.output_image_path, input_image_path.resolve())
                            self.assertNotEqual(result.output_image_path, edge_map_path.resolve())
                            self.assertEqual(result.room_type, "Living Room")
                            self.assertEqual(result.style, "Modern Luxury")

                            self.assertIsNotNone(fake_pipeline.last_kwargs)
                            assert fake_pipeline.last_kwargs is not None
                            self.assertEqual(fake_pipeline.last_kwargs["strength"], 0.80)
                            self.assertEqual(fake_pipeline.last_kwargs["guidance_scale"], 6.0)
                            self.assertEqual(fake_pipeline.last_kwargs["controlnet_conditioning_scale"], 0.55)
                            self.assertEqual(fake_pipeline.last_kwargs["num_inference_steps"], 35)
                            self.assertEqual(
                                fake_pipeline.last_kwargs["image"].size,
                                fake_pipeline.last_kwargs["control_image"].size,
                            )

        log_output = "\n".join(captured_logs.output)
        self.assertIn("Init image size/mode", log_output)
        self.assertIn("Control image size/mode", log_output)
        self.assertIn("Output image pixel stats before save", log_output)

    def test_generate_rejects_blank_outputs(self) -> None:
        """The generator should fail clearly when the model returns a blank image."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_image_path = temp_path / "room.png"
            edge_map_path = temp_path / "room_edges.png"

            Image.new("RGB", (640, 480), "white").save(input_image_path)
            Image.new("RGB", (640, 480), "black").save(edge_map_path)

            fake_pipeline = _BlankPipeline()
            generator = ControlNetSDXLGenerator()

            with patch.object(generator, "load_pipeline", return_value=fake_pipeline):
                with patch.object(
                    generator,
                    "_import_runtime_dependencies",
                    return_value=(_FakeTorch, None, None, None),
                ):
                    with patch.object(generator, "_resolve_device", return_value="cpu"):
                        with self.assertRaisesRegex(RuntimeError, "blank or nearly black"):
                            generator.generate(
                                input_image_path=input_image_path,
                                edge_map_path=edge_map_path,
                                room_type="Living Room",
                                style="Modern Luxury",
                            )
