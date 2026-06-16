"""Tests for model implementations."""

from unittest.mock import Mock, patch


class TestHuggingFaceModelQuantization:
    """Test HuggingFace model quantization configurations."""

    def test_8bit_quantization_config(self):
        """Test that 8-bit quantization uses BitsAndBytesConfig instead of deprecated args."""
        from ragicamp.models.huggingface import HuggingFaceModel

        with (
            patch("ragicamp.models.huggingface.AutoTokenizer") as mock_tokenizer,
            patch("ragicamp.models.huggingface.AutoModelForCausalLM") as mock_model_cls,
        ):
            # Create mocks
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model_cls.from_pretrained.return_value = Mock()

            # Initialize with 8-bit
            HuggingFaceModel(model_name="test-model", device="cuda", load_in_8bit=True)

            # Verify AutoModelForCausalLM.from_pretrained was called correctly
            call_args = mock_model_cls.from_pretrained.call_args

            # Check that quantization_config was passed
            assert "quantization_config" in call_args[1]
            assert call_args[1]["quantization_config"] is not None

            # Check that deprecated load_in_8bit and load_in_4bit are NOT in call args
            assert "load_in_8bit" not in call_args[1]
            assert "load_in_4bit" not in call_args[1]

            # Check that device_map is set to auto for quantization
            assert call_args[1]["device_map"] == "auto"

    def test_4bit_quantization_config(self):
        """Test that 4-bit quantization uses BitsAndBytesConfig instead of deprecated args."""
        from ragicamp.models.huggingface import HuggingFaceModel

        with (
            patch("ragicamp.models.huggingface.AutoTokenizer") as mock_tokenizer,
            patch("ragicamp.models.huggingface.AutoModelForCausalLM") as mock_model_cls,
        ):
            # Create mocks
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model_cls.from_pretrained.return_value = Mock()

            # Initialize with 4-bit
            HuggingFaceModel(model_name="test-model", device="cuda", load_in_4bit=True)

            # Verify AutoModelForCausalLM.from_pretrained was called correctly
            call_args = mock_model_cls.from_pretrained.call_args

            # Check that quantization_config was passed
            assert "quantization_config" in call_args[1]
            assert call_args[1]["quantization_config"] is not None

            # Check that deprecated load_in_8bit and load_in_4bit are NOT in call args
            assert "load_in_8bit" not in call_args[1]
            assert "load_in_4bit" not in call_args[1]

            # Check that device_map is set to auto for quantization
            assert call_args[1]["device_map"] == "auto"

    def test_no_quantization(self):
        """Test that no quantization doesn't use quantization_config."""
        from ragicamp.models.huggingface import HuggingFaceModel

        with (
            patch("ragicamp.models.huggingface.AutoTokenizer") as mock_tokenizer,
            patch("ragicamp.models.huggingface.AutoModelForCausalLM") as mock_model_cls,
            patch("ragicamp.models.huggingface.torch.cuda.is_available", return_value=True),
        ):
            # Create mocks
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model_instance = Mock()
            mock_model_cls.from_pretrained.return_value = mock_model_instance

            # Initialize without quantization
            HuggingFaceModel(model_name="test-model", device="cuda")

            # Verify AutoModelForCausalLM.from_pretrained was called correctly
            call_args = mock_model_cls.from_pretrained.call_args

            # Check that quantization_config is None
            assert call_args[1]["quantization_config"] is None

            # Check that device_map is None (not using auto device mapping)
            assert call_args[1]["device_map"] is None

            # Check that model.to() was called (manual device placement)
            mock_model_instance.to.assert_called_once_with("cuda")

    def test_4bit_takes_precedence_over_8bit(self):
        """Test that 4-bit quantization takes precedence when both are specified."""
        from ragicamp.models.huggingface import HuggingFaceModel

        with (
            patch("ragicamp.models.huggingface.AutoTokenizer") as mock_tokenizer,
            patch("ragicamp.models.huggingface.AutoModelForCausalLM") as mock_model_cls,
        ):
            # Create mocks
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model_cls.from_pretrained.return_value = Mock()

            # Initialize with both (4-bit should take precedence)
            HuggingFaceModel(
                model_name="test-model", device="cuda", load_in_8bit=True, load_in_4bit=True
            )

            # Verify quantization_config was created
            call_args = mock_model_cls.from_pretrained.call_args
            assert "quantization_config" in call_args[1]
            assert call_args[1]["quantization_config"] is not None
