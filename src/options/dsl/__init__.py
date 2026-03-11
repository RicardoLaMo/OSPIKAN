"""
DSL (Domain-Specific Language) for options pricing and regime queries.

Provides: Lexer, Parser, AST nodes, Validator, and Executor for intuitive
options pricing commands and regime/covariance analysis queries.
"""

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
)
from .lexer import Lexer, Token, TokenKind
from .parser import parse_dsl, Parser
from .validator import DSLValidator, ValidationError

__all__ = [
    "ASTNode",
    "PriceQuery",
    "RegimeCurrentQuery",
    "RegimeProbQuery",
    "CovarianceQuery",
    "TransitionQuery",
    "WhatIfQuery",
    "ExplainQuery",
    "SurfaceQuery",
    "Lexer",
    "Token",
    "TokenKind",
    "parse_dsl",
    "Parser",
    "DSLValidator",
    "ValidationError",
]
