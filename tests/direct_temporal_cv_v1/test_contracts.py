from datetime import date

from src.direct_temporal_cv_v1.contracts import FOUR_FOLD_250K_V1


def test_four_fold_contract_is_exact() -> None:
    assert [fold.inference_anchor for fold in FOUR_FOLD_250K_V1] == [date(2025, 10, 16), date(2025, 11, 15), date(2025, 12, 15), date(2026, 1, 14)]
    for fold in FOUR_FOLD_250K_V1:
        assert fold.train_target_end == fold.inference_anchor
        assert fold.validation_target_start == fold.inference_anchor.fromordinal(fold.inference_anchor.toordinal() + 1)
        assert fold.validation_target_end == fold.inference_anchor.fromordinal(fold.inference_anchor.toordinal() + 30)
