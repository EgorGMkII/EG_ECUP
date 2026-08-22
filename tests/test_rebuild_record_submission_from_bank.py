"""Unit tests for the pure record-submission assembler contract."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_record_submission_from_bank.py"
SPEC = importlib.util.spec_from_file_location("record_assembler", MODULE_PATH)
assembler = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(assembler)


def meta():
    return {
        "model_order": ["CatBoost", "S1_GRU", "S2_GRU", "ETT"],
        "react_stack_weights": [1.0, 0.0, 0.0, 0.0],
        "churn_stack_weights": [1.0, 0.0, 0.0, 0.0],
        "amount_ridge_coefficients": [1.0, 0.0, 0.0, 0.0],
        "amount_ridge_intercept": -2.0,
        "ALPHA": 1.1,
    }


class RecordAssemblerTest(unittest.TestCase):
    def setUp(self):
        self.react = np.array([[0.0, 2.0, -1.0, 1.0], [2.0, 0.0, 1.0, -1.0]])
        self.churn = np.array([[0.0, 1.0, -1.0, 2.0], [2.0, -1.0, 1.0, 0.0]])
        self.amount = np.array([[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])

    def test_canonical_formula_uses_logits_alpha_and_conditional_clipping(self):
        values = assembler.compute_predictions([0, 1], self.react, self.churn, self.amount, meta())
        expected_first = 0.0  # amount 1 + intercept -2 clips before hurdle fusion
        # Active users use 1 - sigmoid(churn_logit), not sigmoid(churn_logit).
        expected_second = np.expm1((1.0 - 1.0 / (1.0 + np.exp(-2.0))) ** 1.1)
        np.testing.assert_allclose(values, [expected_first, expected_second])

    def test_rejects_invalid_model_order(self):
        bad = meta()
        bad["model_order"] = ["S1_GRU", "CatBoost", "S2_GRU", "ETT"]
        with self.assertRaisesRegex(assembler.ContractError, "model_order"):
            assembler.validate_meta(bad)

    def test_rejects_nan_and_non_binary_activity(self):
        with self.assertRaisesRegex(assembler.ContractError, "NaN or Inf"):
            assembler.compute_predictions([0, 1], np.full((2, 4), np.nan), self.churn, self.amount, meta())
        with self.assertRaisesRegex(assembler.ContractError, "was_active"):
            assembler.compute_predictions([0, 2], self.react, self.churn, self.amount, meta())

    def test_rejects_missing_logit_column_and_probability_contract_violation(self):
        with self.assertRaisesRegex(assembler.ContractError, "missing required"):
            assembler.validate_bank_columns(["user_id", "was_active"])
        with self.assertRaisesRegex(assembler.ContractError, "probability"):
            assembler.validate_bank_columns([*assembler.REQUIRED_BANK_COLUMNS, "cb_react_prob"])

    def test_rejects_duplicate_and_missing_users_before_join(self):
        with self.assertRaisesRegex(assembler.ContractError, "duplicate"):
            assembler.validate_user_ids(["2", "2"], ["1", "2"])
        with self.assertRaisesRegex(assembler.ContractError, "do not exactly match"):
            assembler.validate_user_ids(["1", "3"], ["1", "2"])

    def test_output_schema_contract_constants(self):
        self.assertEqual(assembler.MODEL_ORDER, ("CatBoost", "S1_GRU", "S2_GRU", "ETT"))
        self.assertEqual(assembler.REQUIRED_BANK_COLUMNS[0:2], ("user_id", "was_active"))
