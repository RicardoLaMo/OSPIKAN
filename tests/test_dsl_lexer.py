"""
Tests for the options DSL lexer.
"""

import pytest
from src.options.dsl.lexer import Lexer, TokenKind


class TestLexerBasic:
    """Test basic tokenization."""

    def test_empty_string(self):
        """Test lexing empty string."""
        lexer = Lexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].kind == TokenKind.EOF

    def test_simple_price_command(self):
        """Test lexing simple PRICE command."""
        dsl = "PRICE option type=call S=30 K=32"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        kinds = [t.kind for t in tokens]
        assert kinds[0] == TokenKind.PRICE
        assert kinds[1] == TokenKind.OPTION
        assert kinds[2] == TokenKind.IDENTIFIER  # type
        assert kinds[3] == TokenKind.EQUALS
        assert kinds[4] == TokenKind.IDENTIFIER  # call
        assert kinds[-1] == TokenKind.EOF

    def test_numbers_and_identifiers(self):
        """Test lexing numbers and identifiers."""
        dsl = "S=30 K=32 sigma=0.25"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        # S
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[0].value == "S"
        # =
        assert tokens[1].kind == TokenKind.EQUALS
        # 30
        assert tokens[2].kind == TokenKind.NUMBER
        assert tokens[2].value == "30"
        # K
        assert tokens[3].kind == TokenKind.IDENTIFIER
        assert tokens[3].value == "K"
        # =
        assert tokens[4].kind == TokenKind.EQUALS
        # 32
        assert tokens[5].kind == TokenKind.NUMBER
        assert tokens[5].value == "32"
        # sigma
        assert tokens[6].kind == TokenKind.IDENTIFIER
        assert tokens[6].value == "sigma"
        # =
        assert tokens[7].kind == TokenKind.EQUALS
        # 0.25
        assert tokens[8].kind == TokenKind.NUMBER
        assert tokens[8].value == "0.25"

    def test_time_suffixes(self):
        """Test lexing time suffixes (d, w, m)."""
        dsl = "T=45d horizon=4w window=6m"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        # T=45d
        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[2].kind == TokenKind.TIME_SUFFIX
        assert tokens[2].value == "45d"

        # horizon=4w
        assert tokens[3].kind == TokenKind.IDENTIFIER
        assert tokens[5].kind == TokenKind.TIME_SUFFIX
        assert tokens[5].value == "4w"

        # window=6m
        assert tokens[6].kind == TokenKind.IDENTIFIER
        assert tokens[8].kind == TokenKind.TIME_SUFFIX
        assert tokens[8].value == "6m"

    def test_list_syntax(self):
        """Test lexing list syntax."""
        dsl = "assets=[silver,gold,dxy]"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        # assets is a keyword, so it will be TokenKind.ASSETS
        assert tokens[0].kind == TokenKind.ASSETS
        assert tokens[1].kind == TokenKind.EQUALS
        assert tokens[2].kind == TokenKind.LBRACKET
        assert tokens[3].kind == TokenKind.IDENTIFIER
        assert tokens[4].kind == TokenKind.COMMA
        assert tokens[5].kind == TokenKind.IDENTIFIER
        assert tokens[6].kind == TokenKind.COMMA
        assert tokens[7].kind == TokenKind.IDENTIFIER
        assert tokens[8].kind == TokenKind.RBRACKET

    def test_keywords(self):
        """Test that keywords are correctly identified."""
        dsl = "PRICE REGIME COVARIANCE TRANSITION WHAT_IF EXPLAIN SURFACE"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.PRICE
        assert tokens[1].kind == TokenKind.REGIME
        assert tokens[2].kind == TokenKind.COVARIANCE
        assert tokens[3].kind == TokenKind.TRANSITION
        assert tokens[4].kind == TokenKind.WHAT_IF
        assert tokens[5].kind == TokenKind.EXPLAIN
        assert tokens[6].kind == TokenKind.SURFACE

    def test_quoted_strings(self):
        """Test lexing quoted strings."""
        dsl = 'name="silver call"'
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.IDENTIFIER
        assert tokens[1].kind == TokenKind.EQUALS
        assert tokens[2].kind == TokenKind.STRING
        assert tokens[2].value == "silver call"

    def test_single_quoted_strings(self):
        """Test lexing single-quoted strings."""
        dsl = "name='test value'"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[2].kind == TokenKind.STRING
        assert tokens[2].value == "test value"

    def test_escaped_quotes(self):
        """Test lexing escaped quotes in strings."""
        dsl = r'text="He said \"Hello\""'
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[2].kind == TokenKind.STRING
        assert tokens[2].value == 'He said "Hello"'

    def test_invalid_character(self):
        """Test that invalid characters raise error."""
        dsl = "S=30 @ K=32"
        lexer = Lexer(dsl)

        with pytest.raises(SyntaxError, match="Unexpected character"):
            lexer.tokenize()

    def test_unterminated_string(self):
        """Test that unterminated strings raise error."""
        dsl = 'text="unterminated'
        lexer = Lexer(dsl)

        with pytest.raises(SyntaxError, match="Unterminated string"):
            lexer.tokenize()

    def test_whitespace_handling(self):
        """Test that various whitespace is handled correctly."""
        dsl = "  PRICE   option   type=call   S=30  \t\nK=32  "
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        # Should have same tokens as without extra whitespace
        assert tokens[0].kind == TokenKind.PRICE
        assert tokens[1].kind == TokenKind.OPTION
        assert tokens[2].kind == TokenKind.IDENTIFIER
        assert tokens[-1].kind == TokenKind.EOF


class TestLexerFullExamples:
    """Test lexing full DSL examples."""

    def test_price_query_full(self):
        """Test full PRICE query."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=kan_regime regime=STRESS r=0.05"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.PRICE
        assert tokens[0].value == "PRICE"
        assert tokens[1].kind == TokenKind.OPTION
        # All tokens should be valid
        assert tokens[-1].kind == TokenKind.EOF

    def test_regime_query(self):
        """Test REGIME query."""
        dsl = "REGIME current asset=silver"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.REGIME
        assert tokens[1].kind == TokenKind.CURRENT

    def test_covariance_query(self):
        """Test COVARIANCE query."""
        dsl = "COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.COVARIANCE
        # List should be properly tokenized
        bracket_idx = next(i for i, t in enumerate(tokens) if t.kind == TokenKind.LBRACKET)
        assert tokens[bracket_idx + 1].kind == TokenKind.IDENTIFIER
        assert tokens[bracket_idx + 1].value == "silver"

    def test_surface_query(self):
        """Test SURFACE query."""
        dsl = "SURFACE vol asset=silver regime=all strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].kind == TokenKind.SURFACE
        assert tokens[1].kind == TokenKind.VOL

    def test_position_tracking(self):
        """Test that token positions are tracked."""
        dsl = "PRICE option S=30"
        lexer = Lexer(dsl)
        tokens = lexer.tokenize()

        assert tokens[0].position == 0
        assert tokens[1].position > 0  # "option" starts after "PRICE "
        # Positions should be increasing or equal
        for i in range(len(tokens) - 1):
            assert tokens[i + 1].position >= tokens[i].position
