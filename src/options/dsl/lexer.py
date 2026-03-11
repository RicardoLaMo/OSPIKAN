"""
Lexer for the options pricing DSL.

Tokenizes DSL strings into a stream of tokens for parsing.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional


class TokenKind(Enum):
    """Token types for the DSL lexer."""
    # Verbs (commands)
    PRICE = auto()
    REGIME = auto()
    COVARIANCE = auto()
    TRANSITION = auto()
    WHAT_IF = auto()
    EXPLAIN = auto()
    SURFACE = auto()

    # Keywords
    OPTION = auto()
    CURRENT = auto()
    PROB = auto()
    MATRIX = auto()
    ASSETS = auto()
    REGIME_SHIFT = auto()
    VOL = auto()

    # Operators
    EQUALS = auto()  # =
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()  # ,

    # Values
    IDENTIFIER = auto()  # alphanumeric with underscores
    NUMBER = auto()  # integer or float
    STRING = auto()  # quoted string
    TIME_SUFFIX = auto()  # d, w, m (e.g., 30d)

    # Special
    EOF = auto()


@dataclass
class Token:
    """A single token with kind and value."""
    kind: TokenKind
    value: str
    position: int = 0  # byte position in input


class Lexer:
    """Pure-Python tokenizer for DSL strings."""

    KEYWORDS = {
        "PRICE": TokenKind.PRICE,
        "REGIME": TokenKind.REGIME,
        "COVARIANCE": TokenKind.COVARIANCE,
        "TRANSITION": TokenKind.TRANSITION,
        "WHAT_IF": TokenKind.WHAT_IF,
        "EXPLAIN": TokenKind.EXPLAIN,
        "SURFACE": TokenKind.SURFACE,
        "option": TokenKind.OPTION,
        "current": TokenKind.CURRENT,
        "prob": TokenKind.PROB,
        "matrix": TokenKind.MATRIX,
        "assets": TokenKind.ASSETS,
        "regime_shift": TokenKind.REGIME_SHIFT,
        "vol": TokenKind.VOL,
    }

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: List[Token] = []

    def error(self, msg: str) -> None:
        """Raise a lexer error with position info."""
        raise SyntaxError(f"Lexer error at position {self.pos}: {msg}")

    def peek(self, offset: int = 0) -> Optional[str]:
        """Peek at a character without consuming."""
        idx = self.pos + offset
        if idx < len(self.text):
            return self.text[idx]
        return None

    def advance(self) -> Optional[str]:
        """Consume and return the next character."""
        if self.pos < len(self.text):
            ch = self.text[self.pos]
            self.pos += 1
            return ch
        return None

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.peek() and self.peek() in ' \t\n\r':
            self.advance()

    def read_number(self) -> Token:
        """Read a number token (int or float)."""
        start_pos = self.pos
        num_str = ''
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            num_str += self.advance()
        return Token(TokenKind.NUMBER, num_str, start_pos)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword token."""
        start_pos = self.pos
        ident = ''
        while self.peek() and (self.peek().isalnum() or self.peek() in '_'):
            ident += self.advance()

        # Check if it's a keyword
        kind = self.KEYWORDS.get(ident, TokenKind.IDENTIFIER)
        return Token(kind, ident, start_pos)

    def read_time_suffix(self) -> Token:
        """Read a time suffix like '30d', '4w', '6m'."""
        start_pos = self.pos
        # We've already read digits; now read the suffix letter
        suffix = self.advance()  # 'd', 'w', or 'm'
        return Token(TokenKind.TIME_SUFFIX, suffix, start_pos)

    def read_string(self, quote: str) -> Token:
        """Read a quoted string."""
        start_pos = self.pos
        self.advance()  # consume opening quote
        s = ''
        while self.peek() and self.peek() != quote:
            if self.peek() == '\\':
                self.advance()
                escaped = self.advance()
                if escaped == 'n':
                    s += '\n'
                elif escaped == 't':
                    s += '\t'
                elif escaped == '\\':
                    s += '\\'
                elif escaped == quote:
                    s += quote
                else:
                    s += escaped
            else:
                s += self.advance()
        if self.peek() != quote:
            self.error(f"Unterminated string starting at position {start_pos}")
        self.advance()  # consume closing quote
        return Token(TokenKind.STRING, s, start_pos)

    def tokenize(self) -> List[Token]:
        """Tokenize the entire input string."""
        self.tokens = []
        while self.pos < len(self.text):
            self.skip_whitespace()
            if self.pos >= len(self.text):
                break

            ch = self.peek()

            # Single-character tokens
            if ch == '=':
                self.tokens.append(Token(TokenKind.EQUALS, '=', self.pos))
                self.advance()
            elif ch == '[':
                self.tokens.append(Token(TokenKind.LBRACKET, '[', self.pos))
                self.advance()
            elif ch == ']':
                self.tokens.append(Token(TokenKind.RBRACKET, ']', self.pos))
                self.advance()
            elif ch == ',':
                self.tokens.append(Token(TokenKind.COMMA, ',', self.pos))
                self.advance()
            # Quoted strings
            elif ch in '"\'':
                self.tokens.append(self.read_string(ch))
            # Numbers and time suffixes
            elif ch.isdigit():
                num_token = self.read_number()
                # Check if followed by time suffix letter (d, w, m)
                if self.peek() and self.peek() in 'dwm':
                    suffix_token = self.read_time_suffix()
                    # Combine number and suffix into one token value
                    num_token.value += suffix_token.value
                    num_token.kind = TokenKind.TIME_SUFFIX
                # Otherwise keep as NUMBER token (for decimals like 0.5)
                self.tokens.append(num_token)
            # Identifiers and keywords
            elif ch.isalpha() or ch == '_':
                self.tokens.append(self.read_identifier())
            else:
                self.error(f"Unexpected character: {ch}")

        self.tokens.append(Token(TokenKind.EOF, '', self.pos))
        return self.tokens
