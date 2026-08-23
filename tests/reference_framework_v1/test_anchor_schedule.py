from src.ssl_temporal_stack_v1.training import round_robin_batches


def test_weighted_anchor_schedule_is_deterministic() -> None:
    def factory(_: int):
        yield "batch", 1

    stream = round_robin_batches({"old": factory, "new": factory}, {"old": 1, "new": 3})
    observed = [next(stream).anchor for _ in range(8)]
    assert observed[:4].count("new") == 3
    assert observed[:4].count("old") == 1
