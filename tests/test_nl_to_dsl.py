"""
Tests for Natural Language → DSL translation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.options.llm.client import OllamaClient
from src.options.llm.nl_to_dsl import NLToDSL
from src.options.llm.prompts import (
    build_system_prompt,
    build_few_shot_examples,
    build_error_recovery_prompt,
    build_context_prompt,
)


class TestPrompts:
    """Test prompt construction."""

    def test_system_prompt_contains_rules(self):
        """Test that system prompt contains key rules."""
        prompt = build_system_prompt()

        assert "UPPERCASE" in prompt
        assert "QUOTE" in prompt
        assert "PRICE" in prompt
        assert "OUTLOOK" in prompt
        assert "30d" in prompt
        assert "[item1,item2,item3]" in prompt

    def test_few_shot_examples_format(self):
        """Test that few-shot examples are properly formatted."""
        examples = build_few_shot_examples()

        assert "Q:" in examples
        assert "A:" in examples
        assert "QUOTE" in examples
        assert "PRICE" in examples
        assert "REGIME" in examples

    def test_few_shot_covers_all_verbs(self):
        """Test that examples cover all 7 DSL verbs."""
        examples = build_few_shot_examples()

        verbs = ["PRICE", "REGIME", "COVARIANCE", "TRANSITION", "WHAT_IF", "EXPLAIN", "SURFACE"]
        for verb in verbs:
            assert verb in examples, f"Verb {verb} not in examples"

    def test_error_recovery_prompt(self):
        """Test error recovery prompt construction."""
        prompt = build_error_recovery_prompt(
            user_query="Price a call",
            prev_dsl="PRICE option type=call",
            error_msg="Missing required parameter: S",
        )

        assert "PREVIOUS OUTPUT" in prompt
        assert "ERROR" in prompt
        assert "USER QUERY" in prompt
        assert "PRICE option type=call" in prompt

    def test_context_prompt_format(self):
        """Test context prompt structure."""
        system, user_msg = build_context_prompt("Price a call option")

        assert len(system) > 0
        assert len(user_msg) > 0
        assert "Price a call option" in user_msg


class TestOllamaClient:
    """Test Ollama client (with mocking)."""

    @patch("requests.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b"},
                {"name": "other:latest"},
            ]
        }
        mock_get.return_value = mock_response

        client = OllamaClient()
        assert client.health_check() is True

    @patch("requests.get")
    def test_health_check_no_model(self, mock_get):
        """Test health check when model not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "other:latest"}]}
        mock_get.return_value = mock_response

        client = OllamaClient(model="missing:model")
        assert client.health_check() is False

    @patch("requests.get")
    def test_health_check_connection_error(self, mock_get):
        """Test health check with connection error."""
        import requests
        mock_get.side_effect = requests.ConnectionError()

        client = OllamaClient()
        assert client.health_check() is False

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_success(self, mock_get, mock_post):
        """Test successful text generation."""
        # Mock health check
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {
            "models": [{"name": "qwen2.5-coder:7b"}]
        }
        mock_get.return_value = health_response

        # Mock generate
        gen_response = Mock()
        gen_response.json.return_value = {
            "response": "PRICE option type=call S=100 K=100 T=30d sigma=0.2"
        }
        mock_post.return_value = gen_response

        client = OllamaClient()
        result = client.generate("Price a call option")

        assert "PRICE" in result
        assert "type=call" in result

    @patch("requests.post")
    @patch("requests.get")
    def test_generate_unavailable_server(self, mock_get, mock_post):
        """Test error when Ollama server unavailable."""
        mock_get.return_value.status_code = 500

        client = OllamaClient()

        with pytest.raises(RuntimeError, match="not available"):
            client.generate("Test prompt")

    @patch("requests.post")
    @patch("requests.get")
    def test_chat_success(self, mock_get, mock_post):
        """Test successful chat."""
        health_response = Mock()
        health_response.status_code = 200
        health_response.json.return_value = {
            "models": [{"name": "qwen2.5-coder:7b"}]
        }
        mock_get.return_value = health_response

        chat_response = Mock()
        chat_response.json.return_value = {
            "message": {"content": "PRICE option type=call S=100 K=100 T=30d sigma=0.2"}
        }
        mock_post.return_value = chat_response

        client = OllamaClient()
        messages = [{"role": "user", "content": "Price a call"}]
        result = client.chat(messages)

        assert "PRICE" in result


class TestNLToDSL:
    """Test NL→DSL translator."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Ollama client."""
        client = Mock(spec=OllamaClient)
        client.health_check.return_value = True
        return client

    def test_translator_initialization(self, mock_client):
        """Test translator initialization."""
        translator = NLToDSL(client=mock_client)

        assert translator.client == mock_client
        assert translator.max_retries == 3
        assert translator.temperature == 0.5

    def test_translate_empty_query(self, mock_client):
        """Test error on empty query."""
        translator = NLToDSL(client=mock_client)

        with pytest.raises(ValueError, match="empty"):
            translator.translate("")

    def test_translate_price_query(self, mock_client):
        """Test translating price query."""
        mock_client.chat.return_value = (
            "PRICE option type=call S=100 K=100 T=30d sigma=0.2"
        )

        translator = NLToDSL(client=mock_client)
        result = translator.translate("Price a call option with spot 100, strike 100, 30 days, 20% vol")

        assert "PRICE" in result
        assert "type=call" in result
        assert "S=100" in result
        mock_client.chat.assert_called_once()

    def test_translate_regime_query(self, mock_client):
        """Test translating regime query."""
        mock_client.chat.return_value = (
            "REGIME current asset=silver"
        )

        translator = NLToDSL(client=mock_client)
        result = translator.translate("What's the current regime for silver?")

        assert "REGIME" in result
        assert "asset=silver" in result

    def test_translate_with_retry_on_validation_error(self, mock_client):
        """Test retry on validation failure."""
        # First attempt fails validation, second succeeds
        mock_client.chat.side_effect = [
            "invalid dsl @#$",  # Invalid
            "PRICE option type=call S=100 K=100 T=30d sigma=0.2",  # Valid
        ]

        translator = NLToDSL(client=mock_client, max_retries=2)
        result = translator.translate("Price a call", verbose=False)

        assert "PRICE" in result
        assert mock_client.chat.call_count == 2

    def test_translate_max_retries_exceeded(self, mock_client):
        """Test error when max retries exceeded."""
        # Always return invalid DSL
        mock_client.chat.return_value = "totally invalid @#$"

        translator = NLToDSL(client=mock_client, max_retries=2)

        with pytest.raises(RuntimeError, match="failed after"):
            translator.translate("Price a call", verbose=False)

        # Should have tried 2 times
        assert mock_client.chat.call_count == 2

    def test_translate_what_if_query(self, mock_client):
        """Test translating what-if query."""
        mock_client.chat.return_value = (
            "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]"
        )

        translator = NLToDSL(client=mock_client)
        result = translator.translate("What happens to delta and vega if silver shifts to stress regime?")

        assert "WHAT_IF" in result
        assert "STRESS" in result
        assert "show=[delta,vega]" in result

    def test_translate_surface_query(self, mock_client):
        """Test translating surface query."""
        mock_client.chat.return_value = (
            "SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d"
        )

        translator = NLToDSL(client=mock_client)
        result = translator.translate("Generate a vol surface for silver in stable regime")

        assert "SURFACE" in result
        assert "asset=silver" in result
        assert "strikes=[0.9,1.0,1.1]" in result

    def test_batch_translate(self, mock_client):
        """Test batch translation."""
        mock_client.chat.side_effect = [
            "PRICE option type=call S=100 K=100 T=30d sigma=0.2",
            "REGIME current asset=silver",
            "WHAT_IF regime_shift to=STRESS asset=silver show=[price]",
        ]

        translator = NLToDSL(client=mock_client)
        queries = [
            "Price a call",
            "Current regime",
            "What if stress regime",
        ]

        results = translator.batch_translate(queries, verbose=False)

        assert len(results) == 3
        assert "PRICE" in results[0]
        assert "REGIME" in results[1]
        assert "WHAT_IF" in results[2]

    def test_translate_removes_extra_whitespace(self, mock_client):
        """Test that extra whitespace is removed."""
        mock_client.chat.return_value = "  PRICE option type=call S=100 K=100 T=30d sigma=0.2  \n  "

        translator = NLToDSL(client=mock_client)
        result = translator.translate("Price a call")

        assert result == "PRICE option type=call S=100 K=100 T=30d sigma=0.2"

    def test_translate_removes_trailing_explanation(self, mock_client):
        """Test that trailing explanation is removed."""
        mock_client.chat.return_value = (
            "PRICE option type=call S=100 K=100 T=30d sigma=0.2\n"
            "This prices a call option with..."
        )

        translator = NLToDSL(client=mock_client)
        result = translator.translate("Price a call")

        assert result == "PRICE option type=call S=100 K=100 T=30d sigma=0.2"
        assert "prices a call" not in result


class TestIntegration:
    """Integration tests (with mocking)."""

    @pytest.fixture
    def mock_client(self):
        """Create mock Ollama client."""
        client = Mock(spec=OllamaClient)
        client.health_check.return_value = True
        return client

    def test_full_pipeline_nl_to_execution(self, mock_client):
        """Test full pipeline: NL → DSL → Parsing → Execution."""
        from src.options.dsl.parser import parse_dsl
        from src.options.dsl.executor import DSLExecutor

        # Mock LLM response
        mock_client.chat.return_value = (
            "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2"
        )

        # Translate
        translator = NLToDSL(client=mock_client)
        dsl = translator.translate("Price a call option with spot 100, strike 100, 90 days, 20% vol")

        # Parse
        node = parse_dsl(dsl)

        # Execute
        executor = DSLExecutor()
        result = executor.execute(node)

        # Verify
        assert result["query_type"] == "PRICE"
        assert result["price"] > 0
        assert result["option_type"] == "call"

    def test_translate_multiple_verb_types(self, mock_client):
        """Test translating all 7 DSL verbs."""
        dsl_outputs = [
            "PRICE option type=call S=100 K=100 T=30d sigma=0.2",
            "REGIME current asset=silver",
            "REGIME prob from=STABLE to=STRESS horizon=10d",
            "COVARIANCE assets=[silver,gold] regime=STABLE window=60d",
            "TRANSITION matrix asset=silver normalize=true",
            "WHAT_IF regime_shift to=STRESS asset=silver show=[delta]",
            "EXPLAIN regime=STABLE features=[ricci_curvature]",
            "SURFACE vol asset=silver regime=STABLE strikes=[1.0] T=30d",
        ]

        translator = NLToDSL(client=mock_client)

        for dsl in dsl_outputs:
            mock_client.chat.return_value = dsl

            # Should parse and validate without error
            result = translator.translate("Test query")
            assert len(result) > 0
