"""
Validation module for out-of-sample testing.

IMPORTANT: Use the FIXED validation framework (out_of_sample_fixed.py) for all thesis work.
The original out_of_sample.py has critical data leakage issues.

Usage:
    from src.validation.out_of_sample_fixed import (
        ValidationConfig,
        compare_models,
        walk_forward_validation,
    )
"""

# Re-export from fixed module as the default
from src.validation.out_of_sample_fixed import (
    ValidationConfig,
    ValidationResults,
    compare_models,
    walk_forward_validation,
    print_comparison_summary,
    save_validation_report,
)

__all__ = [
    "ValidationConfig",
    "ValidationResults",
    "compare_models",
    "walk_forward_validation",
    "print_comparison_summary",
    "save_validation_report",
]
