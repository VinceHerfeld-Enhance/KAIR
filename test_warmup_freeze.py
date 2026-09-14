"""Regression test for ModelELVSR's warm-up freeze (``warmup_freeze_keys``).

Runnable directly (``python test_warmup_freeze.py``) or under pytest.

The trap this guards: the optical flow model lives INSIDE the encoder, so its
parameters are named ``encoder.optical_flow_model.*``. ``fix_keys`` matching is by
name substring, so a warm-up key of ``"encoder"`` also matches SpyNet -- and
releasing the warm-up blindly would unfreeze it, silently undoing the permanent
``fix_keys`` freeze that is supposed to hold for the whole run.
"""
import torch.nn as nn

from models.model_elvsr import ModelELVSR


class _FakeNet(nn.Module):
    """Parameter names mirroring STVSRNet's, which is all the freeze logic sees."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.optical_flow_model = nn.Linear(2, 2)
        self.encoder.trunk = nn.Linear(2, 2)
        self.prior = nn.Linear(2, 2)
        self.prior_adapter = nn.Linear(2, 2)
        self.bspline_flow = nn.Linear(2, 2)


def _model():
    m = object.__new__(ModelELVSR)  # skip the heavy constructor
    m.netG = _FakeNet()
    m.fix_keys = ["optical_flow_model"]
    m.warmup_freeze_keys = ["encoder", "prior"]
    m.warmup_freeze_iter = 5000
    m.warmup_frozen = False
    m.warmup_released = False
    return m


def _grads(m, prefix):
    return [p.requires_grad for n, p in m.netG.named_parameters() if n.startswith(prefix)]


def test_warmup_freezes_encoder_and_prior_but_not_the_head():
    m = _model()
    m._apply_warmup_freeze(0)
    assert not any(_grads(m, "encoder.")), "encoder must be frozen during warm-up"
    assert not any(_grads(m, "prior")), "prior/prior_adapter must be frozen during warm-up"
    assert all(_grads(m, "bspline_flow")), "the trajectory head must keep training"


def test_warmup_holds_until_the_threshold():
    m = _model()
    m._apply_warmup_freeze(0)
    m._apply_warmup_freeze(4999)
    assert not any(_grads(m, "encoder.trunk")), "released one step early"


def test_release_keeps_optical_flow_frozen():
    """The regression: releasing 'encoder' must not release SpyNet."""
    m = _model()
    m._apply_warmup_freeze(0)
    m._apply_warmup_freeze(5000)
    assert not any(_grads(m, "encoder.optical_flow_model")), "SpyNet was released by the warm-up"
    assert all(_grads(m, "encoder.trunk")), "encoder should train after the warm-up"
    assert all(_grads(m, "prior")), "prior should train after the warm-up"


def test_resume_past_the_threshold_lands_in_the_right_state():
    """A run resumed past warmup_freeze_iter never ran the freeze half.

    The release must therefore assert the fix_keys invariant itself rather than
    inherit it from the fix_iter block having run first.
    """
    m = _model()
    m._apply_warmup_freeze(9000)
    assert all(_grads(m, "encoder.trunk")), "resume should leave the encoder trainable"
    assert not any(_grads(m, "encoder.optical_flow_model")), "resume left SpyNet trainable"


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures.append(name)
                print(f"FAIL  {name}: {exc}")
    print("FAILURES:", failures or "none")
    raise SystemExit(1 if failures else 0)
