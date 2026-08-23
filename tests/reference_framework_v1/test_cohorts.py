from src.reference_framework_v1.cohorts import _largest_remainder


def test_largest_remainder_preserves_exact_total() -> None:
    allocation = _largest_remainder({"00": 11, "01": 3, "10": 7, "11": 19}, 25)
    assert sum(allocation.values()) == 25
    assert allocation == _largest_remainder({"00": 11, "01": 3, "10": 7, "11": 19}, 25)
