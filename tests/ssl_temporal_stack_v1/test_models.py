from __future__ import annotations

import torch

from src.ssl_temporal_stack_v1.models import (
    EventTimeTransformer,
    GRUBackbone,
    S1MaskedPretrainer,
    S2MultiHorizonPretrainer,
    Specialist,
    TransitionBase,
)


def test_gru_ssl_and_transition_shapes() -> None:
    x = torch.zeros(3, 180, 15)
    s1 = S1MaskedPretrainer()
    assert s1(x).shape == (3, 180, 15)
    s2 = S2MultiHorizonPretrainer()
    assert set(s2(x)) == {"buy_7", "buy_14", "buy_30", "gmv_7", "gmv_14", "gmv_30"}
    encoder = GRUBackbone()
    base = TransitionBase(encoder, lambda values: encoder(values)[1])
    outputs = base(x)
    assert all(value.shape == (3,) for value in outputs.values())


def test_specialist_phase_f_unfreezes_only_last_gru_layer_and_attention() -> None:
    model = Specialist(GRUBackbone(), "react", "s1")
    model.freeze_phase_h()
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    model.unfreeze_phase_f()
    trainable = {name for name, parameter in model.encoder.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all("_l1" in name or name.startswith("attention.") for name in trainable)


def test_ett_empty_history_is_finite() -> None:
    model = EventTimeTransformer().eval()
    with torch.no_grad():
        outputs = model(
            torch.zeros(2, 180, 12), torch.zeros(2, 180, 12),
            torch.zeros(2, 180, dtype=torch.long), torch.ones(2, 180, dtype=torch.bool),
            torch.ones(2, dtype=torch.bool),
        )
    assert all(torch.isfinite(value).all() for value in outputs.values())
