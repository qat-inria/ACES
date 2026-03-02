# gospel/noise_models/uniform_two_qubit_depolarising_noise_model.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, Literal, Optional

from graphix.command import BaseM, CommandKind
from graphix.noise_models.noise_model import (
    A,
    CommandOrNoise,
    NoiseCommands,
    NoiseModel,
)
from graphix.rng import ensure_rng
from gospel.noise_models.two_qubit_depolarising_noise import TwoQubitDepolarisingNoise

if TYPE_CHECKING:
    from numpy.random import Generator

logger = logging.getLogger(__name__)

Parametrization = Literal["two_qubit", "single_qubit_like"]


def _check_prob(name: str, p: float) -> None:
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {p!r}")


def expected_lambda_two_qubit(p: float) -> float:
    """Eigenvalue on any non-identity 2-qubit Pauli under proper 2-qubit depolarizing."""
    return 1.0 - (16.0 / 15.0) * p


class UniformTwoQubitDepolarisingNoiseModel(NoiseModel):
    """
    Apply the same 2-qubit depolarizing channel on every entangling command.

    Proper 2-qubit parameterization:
        E(ρ) = (1 - p) ρ + (p/15) * Σ_{Q ∈ P₂ \ {II}} Q ρ Q

    Parameters
    ----------
    entanglement_error_prob : float
        The 2-qubit depolarizing probability p.
    parametrization : {"two_qubit", "single_qubit_like"}, default "two_qubit"
        - "two_qubit": your backend interprets a 2-node noise as proper 2Q depol with p/15 weights.
        - "single_qubit_like": for backends that only implement 1Q semantics; we rescale p_eff = 5p
          so the realized 2Q channel has total non-identity weight p. (Clamped to 1 with a warning.)
    rng : numpy.random.Generator | None
        Included for interface parity; not used here.

    Notes
    -----
    * Only `E` (entangling) commands are noised; all others pass through.
    * Classical measurement outcomes are not flipped.
    * Expected Pauli eigenvalue: λ = 1 - 16 p / 15  (see `expected_lambda_two_qubit`).
    """

    def __init__(
        self,
        entanglement_error_prob: float = 0.0,
        parametrization: Parametrization = "two_qubit",
        rng: Optional[Generator] = None,
    ) -> None:
        _check_prob("entanglement_error_prob", entanglement_error_prob)
        if parametrization not in ("two_qubit", "single_qubit_like"):
            raise ValueError(
                f"parametrization must be 'two_qubit' or 'single_qubit_like', got {parametrization!r}"
            )

        self.p: Final[float] = entanglement_error_prob
        self.parametrization: Final[Parametrization] = parametrization
        self.rng = ensure_rng(rng)

        if self.p == 0.0:
            self._ent_noise: Optional[TwoQubitDepolarisingNoise] = None
        else:
            if self.parametrization == "two_qubit":
                p_eff = self.p
            else:  # "single_qubit_like"
                p_eff = 5.0 * self.p
                if p_eff > 1.0:
                    logger.warning(
                        "single_qubit_like rescaling produced p_eff=%.3f > 1. "
                        "Clamping to 1.0; realized channel will deviate from requested p=%.3f.",
                        p_eff, self.p,
                    )
                    p_eff = 1.0
            self._ent_noise = TwoQubitDepolarisingNoise(prob=p_eff)

    # ---------------- NoiseModel interface ----------------

    def input_nodes(self, nodes: list[int]) -> NoiseCommands:
        # No preparation noise in this model
        return []

    def command(self, cmd: CommandOrNoise) -> NoiseCommands:
        k = cmd.kind

        if k == CommandKind.E:
            u, v = cmd.nodes
            if self._ent_noise is None:
                return [cmd]
            # Use the 2Q noise marker so Stim adapter matches `case TwoQubitDepolarisingNoise(...)`.
            return [cmd, A(noise=self._ent_noise, nodes=[u, v])]

        if k in (CommandKind.N, CommandKind.M, CommandKind.X, CommandKind.Z,
                 CommandKind.C, CommandKind.T, CommandKind.A, CommandKind.S):
            return [cmd]

        raise RuntimeError(f"Unhandled CommandKind: {k!r}")

    def confuse_result(self, cmd: BaseM, result: bool) -> bool:
        # No classical measurement flips in this model
        return result
