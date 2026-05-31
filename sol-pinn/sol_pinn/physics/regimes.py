"""Regime presets for the 1D SOL parallel transport problem.

Provides two standard regimes:
  - Sheath-limited:  T_t/T_up ≈ 0.98,  T(s) nearly flat
  - Conduction-limited: T_t/T_up ≈ 0.6,  T(s) clearly nonlinear

Usage:
    from sol_pinn.physics.regimes import sheath_limited, conduction_limited

    config = sheath_limited()
    config = conduction_limited(T_up=80.0)
"""

from .params import SOLConfig
from .constants import L_DEFAULT, T_UP_DEFAULT


def sheath_limited(**overrides) -> SOLConfig:
    """Return SOLConfig for the sheath-limited regime.

    In this regime the parallel heat conduction is very efficient compared
    to the sheath exhaust capability, so T_t ≈ T_up and T(s) is nearly flat.

    Default parameters (Stangeby Ch.10, sheath-limited SOL):
        L = 10 m,  κ = 2000,  p₀ = 1e21,  T_up = 100 eV

    yields: T_t/T_up ≈ 0.98
    """
    kw = dict(L=10.0, kappa_parallel=2000.0, p0=1e21, T_up=100.0)
    kw.update(overrides)
    return SOLConfig(**kw)


def conduction_limited(**overrides) -> SOLConfig:
    """Return SOLConfig for the conduction-limited regime.

    In this regime the parallel conduction itself is the bottleneck,
    producing a clear temperature gradient along the field line and a
    strongly nonlinear T(s) profile.  This is the regime described in
    Stangeby Ch.11 (conduction-limited SOL).

    Default parameters:
        L = 20 m,  κ = 1000,  p₀ = 2e21,  T_up = 80 eV

    yields: T_t/T_up ≈ 0.61  (visible curvature in T(s))
    """
    kw = dict(L=20.0, kappa_parallel=1000.0, p0=2e21, T_up=80.0)
    kw.update(overrides)
    return SOLConfig(**kw)


# Convenience alias so callers can pass regime="sheath-limited" etc.
_REGIMES = {
    "sheath-limited": sheath_limited,
    "conduction-limited": conduction_limited,
    "sheath": sheath_limited,
    "conduction": conduction_limited,
}


def from_name(name: str, **overrides) -> SOLConfig:
    """Look up a regime by short name."""
    name = name.lower().replace("_", "-")
    factory = _REGIMES.get(name)
    if factory is None:
        msg = f"Unknown regime {name!r}.  Choices: {list(_REGIMES)}"
        raise ValueError(msg)
    return factory(**overrides)
