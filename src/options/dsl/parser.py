"""
Recursive descent parser for the options pricing DSL.

Converts a token stream into an Abstract Syntax Tree (AST).
"""

from typing import List, Union, Optional
from .lexer import Lexer, Token, TokenKind
from .ast_nodes import (
    ASTNode,
    PriceQuery,
    RegimeCurrentQuery,
    RegimeProbQuery,
    CovarianceQuery,
    TransitionQuery,
    WhatIfQuery,
    ExplainQuery,
    SurfaceQuery,
    OutlookQuery,
)


def parse_dsl(text: str) -> ASTNode:
    """
    Convenience function to lex and parse a DSL string.

    Args:
        text: DSL command string

    Returns:
        Parsed AST node
    """
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


class Parser:
    """Recursive descent parser for DSL."""

    PARAM_ALIASES = {
        "S": ("spot",),
        "K": ("strike",),
        "T": ("expiry", "tenor", "maturity"),
        "sigma": ("vol", "volatility"),
        "r": ("rate",),
        "q": ("dividend", "yield"),
        "assets": ("basket",),
        "window": ("lookback",),
        "show": ("metrics",),
        "to": ("target", "toward"),
        "regime": ("backdrop", "state"),
        "asset": ("underlying",),
    }

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def error(self, msg: str) -> None:
        """Raise a parse error with position info."""
        token = self.current_token()
        raise SyntaxError(f"Parse error at position {token.position}: {msg}")

    def current_token(self) -> Token:
        """Get the current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek_token(self, offset: int = 1) -> Token:
        """Peek ahead."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF

    def advance(self) -> Token:
        """Consume and return the current token."""
        token = self.current_token()
        if token.kind != TokenKind.EOF:
            self.pos += 1
        return token

    def expect(self, kind: TokenKind) -> Token:
        """Consume a token of the expected kind or error."""
        token = self.current_token()
        if token.kind != kind:
            self.error(f"Expected {kind}, got {token.kind}")
        return self.advance()

    def parse(self) -> ASTNode:
        """Parse the top-level DSL command."""
        token = self.current_token()

        if token.kind == TokenKind.PRICE:
            return self.parse_price()
        elif token.kind == TokenKind.REGIME:
            return self.parse_regime()
        elif token.kind == TokenKind.COVARIANCE:
            return self.parse_covariance()
        elif token.kind == TokenKind.TRANSITION:
            return self.parse_transition()
        elif token.kind == TokenKind.WHAT_IF:
            return self.parse_what_if()
        elif token.kind == TokenKind.EXPLAIN:
            return self.parse_explain()
        elif token.kind == TokenKind.SURFACE:
            return self.parse_surface()
        elif token.kind == TokenKind.OUTLOOK:
            return self.parse_outlook()
        else:
            self.error(f"Unknown command: {token.value}")

    def parse_price(self) -> PriceQuery:
        """Parse: PRICE option type=call S=30 K=32 T=45d sigma=kan_regime [regime=STRESS] [r=0.05]"""
        self.expect(TokenKind.PRICE)

        inline_option_type = None
        if self.current_token().kind == TokenKind.OPTION:
            self.advance()
        elif self.current_token().kind == TokenKind.IDENTIFIER and self.current_token().value in ("call", "put"):
            inline_option_type = self.advance().value

        # Parse required parameters
        params = self.parse_params()

        # Extract parameters
        option_type = inline_option_type or self._get_param(params, "type", required=True)
        spot_price = float(self._get_param(params, "S", required=True))
        strike_price = float(self._get_param(params, "K", required=True))
        time_str = self._get_param(params, "T", required=True)
        time_to_expiry = self._parse_time(time_str)
        volatility = self._parse_volatility(self._get_param(params, "sigma", required=True))

        # Parse optional parameters
        regime = self._get_param(params, "regime", required=False)
        risk_free_rate = self._parse_ratio(self._get_param(params, "r", required=False, default="0.05"))
        dividend_yield = self._parse_ratio(self._get_param(params, "q", required=False, default="0.0"))

        return PriceQuery(
            option_type=option_type,
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            regime=regime,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

    def parse_regime(self) -> Union[RegimeCurrentQuery, RegimeProbQuery]:
        """Parse: REGIME current asset=silver  OR  REGIME prob from=STABLE to=STRESS horizon=10d"""
        self.expect(TokenKind.REGIME)

        token = self.current_token()
        if token.kind == TokenKind.CURRENT:
            self.advance()
            params = self.parse_params()
            asset = self._get_param(params, "asset", required=True)
            return RegimeCurrentQuery(asset=asset)
        elif token.kind == TokenKind.PROB:
            self.advance()
            params = self.parse_params()
            from_regime = self._get_param(params, "from", required=True)
            to_regime = self._get_param(params, "to", required=True)
            horizon_str = self._get_param(params, "horizon", required=True)
            horizon = self._parse_time(horizon_str)
            return RegimeProbQuery(from_regime=from_regime, to_regime=to_regime, horizon=horizon)
        else:
            self.error("REGIME expects 'current' or 'prob'")

    def parse_covariance(self) -> CovarianceQuery:
        """Parse: COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"""
        self.expect(TokenKind.COVARIANCE)
        params = self.parse_params()

        assets = self._parse_list(self._get_param(params, "assets", required=True))
        regime = self._get_param(params, "regime", required=True)
        window_str = self._get_param(params, "window", required=True)
        window = self._parse_time(window_str)

        return CovarianceQuery(assets=assets, regime=regime, window=window)

    def parse_transition(self) -> TransitionQuery:
        """Parse: TRANSITION matrix asset=silver normalize=true"""
        self.expect(TokenKind.TRANSITION)
        self.expect(TokenKind.MATRIX)
        params = self.parse_params()

        asset = self._get_param(params, "asset", required=True)
        normalize_str = self._get_param(params, "normalize", required=False, default="true")
        normalize = normalize_str.lower() in ("true", "yes", "1")

        return TransitionQuery(asset=asset, normalize=normalize)

    def parse_what_if(self) -> WhatIfQuery:
        """Parse: WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,vol,price]"""
        self.expect(TokenKind.WHAT_IF)
        if self.current_token().kind == TokenKind.REGIME_SHIFT:
            self.advance()
        params = self.parse_params()

        to_regime = self._get_param(params, "to", required=True)
        asset = self._get_param(params, "asset", required=True)
        spot_price = self._get_optional_float(params, "S")
        strike_price = self._get_optional_float(params, "K")
        time_to_expiry = self._get_optional_time(params, "T")
        show_str = self._get_param(params, "show", required=False, default="[delta,vega,vol,price]")
        show = self._parse_list(show_str)

        return WhatIfQuery(
            to_regime=to_regime,
            asset=asset,
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            show=show,
        )

    def parse_explain(self) -> ExplainQuery:
        """Parse: EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress,ga_rotor]"""
        self.expect(TokenKind.EXPLAIN)
        params = self.parse_params()

        regime = self._get_param(params, "regime", required=True)
        features_str = self._get_param(params, "features", required=True)
        features = self._parse_list(features_str)

        return ExplainQuery(regime=regime, features=features)

    def parse_surface(self) -> SurfaceQuery:
        """Parse: SURFACE vol asset=silver regime=all strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"""
        self.expect(TokenKind.SURFACE)
        self.expect(TokenKind.VOL)
        params = self.parse_params()

        asset = self._get_param(params, "asset", required=True)
        regime = self._get_param(params, "regime", required=True)
        strikes_str = self._get_param(params, "strikes", required=True)
        strikes = [float(s) for s in self._parse_list(strikes_str)]
        time_str = self._get_param(params, "T", required=True)
        time_to_expiry = self._parse_time(time_str)
        vol_type = self._get_param(params, "vol_type", required=False, default="implied")

        return SurfaceQuery(
            asset=asset,
            regime=regime,
            strikes=strikes,
            time_to_expiry=time_to_expiry,
            vol_type=vol_type,
        )

    def parse_outlook(self) -> OutlookQuery:
        """Parse: OUTLOOK asset=silver horizon=10d [regime=STRESS]."""
        self.expect(TokenKind.OUTLOOK)
        params = self.parse_params()

        asset = self._get_param(params, "asset", required=True)
        horizon_str = self._get_param(params, "horizon", required=True)
        horizon = self._parse_time(horizon_str)
        regime = self._get_param(params, "regime", required=False)

        return OutlookQuery(asset=asset, horizon=horizon, regime=regime)

    def parse_params(self) -> dict:
        """Parse key=value or key=[...] parameters until EOF."""
        params = {}
        while self.current_token().kind != TokenKind.EOF:
            # Parse key (allow keywords like 'assets' as parameter names)
            key_token = self.current_token()
            # Accept IDENTIFIER, OPTION, ASSETS, or other keywords that could be param names
            if key_token.kind not in (TokenKind.IDENTIFIER, TokenKind.OPTION, TokenKind.ASSETS,
                                     TokenKind.VOL, TokenKind.REGIME_SHIFT):
                break
            key = key_token.value
            self.advance()

            # Expect =
            self.expect(TokenKind.EQUALS)

            # Parse value
            value_token = self.current_token()
            if value_token.kind == TokenKind.LBRACKET:
                # List value: [item1,item2,...]
                self.advance()
                items = []
                while self.current_token().kind != TokenKind.RBRACKET:
                    items.append(self.current_token().value)
                    self.advance()
                    if self.current_token().kind == TokenKind.COMMA:
                        self.advance()
                self.expect(TokenKind.RBRACKET)
                params[key] = "[" + ",".join(items) + "]"
            else:
                # Scalar value (number, identifier, string, or time_suffix)
                params[key] = value_token.value
                self.advance()

        return params

    def _get_param(self, params: dict, key: str, required: bool = False, default: str = None) -> str:
        """Get a parameter value, with optional default."""
        if key in params:
            return params[key]
        for alias in self.PARAM_ALIASES.get(key, ()):
            if alias in params:
                return params[alias]
        if required and default is None:
            self.error(f"Missing required parameter: {key}")
        return default

    def _get_optional_float(self, params: dict, key: str) -> Optional[float]:
        """Get an optional float parameter."""
        val = self._get_param(params, key, required=False)
        return float(val) if val else None

    def _get_optional_time(self, params: dict, key: str) -> Optional[float]:
        """Get an optional time parameter."""
        val = self._get_param(params, key, required=False)
        return self._parse_time(val) if val else None

    def _parse_time(self, time_str: str) -> float:
        """
        Parse time string: 30d, 4w, 6m, 0.5 -> years.

        Args:
            time_str: Time string like "45d", "4w", "6m", or "0.5" (assumes years)

        Returns:
            Time in years as float
        """
        if not time_str:
            return 0.0
        time_str = time_str.strip()
        # Extract number and suffix
        i = 0
        while i < len(time_str) and (time_str[i].isdigit() or time_str[i] == '.'):
            i += 1
        if i == 0:
            self.error(f"Invalid time format: {time_str}")
        num = float(time_str[:i])
        suffix = time_str[i:].lower()

        if suffix == '' or suffix == 'y':
            # No suffix or 'y' means already in years
            return num
        elif suffix == 'd':
            return num / 365.0
        elif suffix == 'w':
            return num * 7 / 365.0
        elif suffix == 'm':
            return num * 30 / 365.0
        else:
            self.error(f"Unknown time suffix: {suffix}")

    def _parse_volatility(self, vol_str: str) -> Union[float, str]:
        """
        Parse volatility: float, 'kan_regime', or 'hist_Nd'.

        Args:
            vol_str: Volatility specification

        Returns:
            float or string identifier
        """
        try:
            return self._parse_ratio(vol_str)
        except ValueError:
            # It's a string like 'kan_regime' or 'hist_20d'
            return vol_str

    def _parse_ratio(self, value: str) -> float:
        """
        Parse a decimal or finance-style percentage.

        Examples:
            0.2  -> 0.2
            20%  -> 0.2
            5%   -> 0.05
        """
        if value.endswith("%"):
            return float(value[:-1]) / 100.0
        return float(value)

    def _parse_list(self, list_str: str) -> List[str]:
        """
        Parse list string: [item1,item2,...] -> [item1, item2, ...]

        Args:
            list_str: List string like "[silver,gold,dxy]"

        Returns:
            List of items (strings)
        """
        if not list_str.startswith('[') or not list_str.endswith(']'):
            self.error(f"Invalid list format: {list_str}")
        inner = list_str[1:-1]
        if not inner:
            return []
        return [item.strip() for item in inner.split(',')]
