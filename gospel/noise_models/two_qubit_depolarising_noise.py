# gospel/noise_models/two_qubit_depolarising_noise.py
from __future__ import annotations
from dataclasses import dataclass

def _check_prob(p: float) -> None:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"prob must be in [0, 1], got {p!r}")

@dataclass(frozen=True)
class TwoQubitDepolarisingNoise:
    """Marker for a proper two-qubit depolarizing channel with total error prob `prob`.

    Interpreted by the Stim adapter as DEPOLARIZE2(q0, q1, prob),
    i.e., (1 - p) * I + (p/15) * sum_{Q != II} Q.
    """
    prob: float

    def __post_init__(self) -> None:
        _check_prob(self.prob)
