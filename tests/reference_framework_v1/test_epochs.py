from src.reference_framework_v1.epochs import _stage
from src.reference_framework_v1.registry import GRUAdapter
from src.ssl_temporal_stack_v1.training import round_robin_batches


def test_epoch_stage_resolves_full_loader_passes_to_steps() -> None:
    resolved = _stage(
        {"epochs": 3, "learning_rate": 3e-4, "scheduler": "cosine", "warmup_fraction": 0.1},
        samples=350_000,
        batches_per_epoch=14 * 49,
        effective_batch_size=512,
    )
    assert resolved["steps"] == 3 * 686
    assert resolved["warmup_steps"] == round(resolved["steps"] * 0.1)
    assert resolved["epoch_resolution"]["samples_per_epoch"] == 350_000


def test_gru_epoch_recipe_is_valid_before_runtime_resolution() -> None:
    stage = {"epochs": 2, "learning_rate": 1e-3, "scheduler": "cosine", "warmup_fraction": 0.1}
    raw = {
        "batch_size": 512,
        "ssl": stage,
        "base": stage,
        "specialists": {task: {"H": stage, "F": stage} for task in ("react", "churn", "amount")},
    }
    assert GRUAdapter("s1").validate_config(raw).model_id == "s1"


def test_synchronized_epoch_does_not_restart_short_anchor_early() -> None:
    def factory(values):
        return lambda _: iter(((value, 1) for value in values))

    stream = round_robin_batches({"short": factory([1]), "long": factory([10, 11, 12])}, synchronized_epochs=True)
    first_epoch = [next(stream).anchor for _ in range(4)]
    assert first_epoch == ["short", "long", "long", "long"]
    assert next(stream).anchor == "short"
