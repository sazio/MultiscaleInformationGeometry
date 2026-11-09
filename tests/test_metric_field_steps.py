"""Step rules of src.metric_field.step_sizes (norm_fraction is the rule of Fig. 5)."""
import math
import pytest
from src.metric_field import STEP_RULES, step_sizes


def test_norm_fraction_is_fraction_of_reference_norm():
    assert step_sizes(238.0, 77.0, "norm_fraction", 0.01, ref_norm=32.9) == pytest.approx((0.329, 0.329))


def test_norm_fraction_needs_reference_norm():
    with pytest.raises(ValueError):
        step_sizes(1.0, 1.0, "norm_fraction", 0.01)


def test_other_rules_unchanged():
    assert step_sizes(4.0, 1.0, "sqrt_lambda", 0.5) == pytest.approx((1.0, 0.5))
    assert step_sizes(4.0, 1.0, "fixed", 0.25) == pytest.approx((0.25, 0.25))
    assert STEP_RULES[0] == "norm_fraction"
