"""
Natural Language → DSL Translation with automatic error recovery.

Translates user queries to DSL with retry loop on validation failures.
"""

from typing import Optional, Tuple
from .client import OllamaClient
from .prompts import build_context_prompt, build_error_recovery_prompt
from ..dsl.parser import parse_dsl
from ..dsl.validator import DSLValidator, ValidationError


class NLToDSL:
    """Translates natural language to DSL with error recovery."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        max_retries: int = 3,
        temperature: float = 0.5,  # Lower temp for more consistent DSL output
    ):
        """
        Initialize translator.

        Args:
            client: OllamaClient (default: creates new one)
            max_retries: Maximum retry attempts on validation failure
            temperature: LLM temperature (lower = more deterministic)
        """
        self.client = client or OllamaClient()
        self.validator = DSLValidator()
        self.max_retries = max_retries
        self.temperature = temperature

    def translate(self, user_query: str, verbose: bool = False) -> str:
        """
        Translate natural language to DSL.

        Args:
            user_query: Natural language query
            verbose: If True, print translation attempts

        Returns:
            DSL string

        Raises:
            RuntimeError: If translation fails after max_retries
            ValueError: If user query is empty or invalid
        """
        if not user_query or not user_query.strip():
            raise ValueError("User query cannot be empty")

        dsl_output = None
        validation_error = None

        # Try to generate DSL
        for attempt in range(self.max_retries):
            try:
                # Generate DSL
                if attempt == 0:
                    # First attempt: normal prompt
                    system_prompt, user_message = build_context_prompt(user_query)
                    dsl_output = self._generate_dsl(system_prompt, user_message)
                else:
                    # Retry: error recovery prompt
                    recovery_prompt = build_error_recovery_prompt(
                        user_query,
                        dsl_output,
                        str(validation_error),
                    )
                    system_prompt, _ = build_context_prompt("")  # Get system prompt only
                    dsl_output = self._generate_dsl(system_prompt, recovery_prompt)

                if verbose:
                    print(f"[Attempt {attempt + 1}] Generated DSL: {dsl_output}")

                # Validate DSL
                try:
                    node = parse_dsl(dsl_output)
                    self.validator.validate(node)

                    if verbose:
                        print(f"[Attempt {attempt + 1}] ✓ Validation passed")

                    return dsl_output

                except (SyntaxError, ValidationError) as e:
                    validation_error = e
                    if verbose:
                        print(f"[Attempt {attempt + 1}] ✗ Validation failed: {e}")

                    # Continue to next retry
                    if attempt < self.max_retries - 1:
                        continue
                    else:
                        raise

            except Exception as e:
                if verbose:
                    print(f"[Attempt {attempt + 1}] Error: {e}")

                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"Translation failed after {self.max_retries} attempts: {e}"
                    ) from e

        raise RuntimeError(
            f"Failed to generate valid DSL after {self.max_retries} attempts. "
            f"Last validation error: {validation_error}"
        )

    def _generate_dsl(self, system_prompt: str, user_message: str) -> str:
        """
        Generate DSL using Ollama.

        Args:
            system_prompt: System prompt
            user_message: User message

        Returns:
            Generated DSL string

        Raises:
            RuntimeError: If Ollama API fails
        """
        try:
            # Use chat interface (more reliable than generate for structured output)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            response = self.client.chat(
                messages=messages,
                temperature=self.temperature,
                top_p=0.95,
                max_tokens=200,  # DSL commands are short
            )

            # Clean output (remove extra whitespace, comments, etc.)
            dsl = response.strip()

            # Remove any trailing explanation
            if "\n" in dsl:
                dsl = dsl.split("\n")[0].strip()

            return dsl

        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def batch_translate(
        self,
        queries: list,
        verbose: bool = False,
    ) -> list:
        """
        Translate multiple queries.

        Args:
            queries: List of natural language queries
            verbose: If True, print progress

        Returns:
            List of DSL strings

        Raises:
            RuntimeError: If any query fails to translate
        """
        results = []
        for i, query in enumerate(queries):
            if verbose:
                print(f"[{i+1}/{len(queries)}] Translating: {query[:50]}...")

            dsl = self.translate(query, verbose=False)
            results.append(dsl)

        return results

    def interactive_mode(self):
        """
        Start interactive translation mode (REPL-like).

        User can enter queries one at a time and see results.
        """
        print("Interactive NL→DSL Translator")
        print("Type 'quit' or 'exit' to stop")
        print("-" * 50)

        while True:
            try:
                user_input = input("\nQuery: ").strip()

                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                if not user_input:
                    print("Empty query. Please try again.")
                    continue

                dsl = self.translate(user_input, verbose=True)

                print(f"\nGenerated DSL:")
                print(f"  {dsl}")

            except (RuntimeError, ValueError) as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
