"""Parameter containers for SOL transport and solver settings."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SOLConfig:
    """Physical configuration for the 1D SOL parallel transport model.

    Attributes:
        L: Field-line length in meters.
        kappa_parallel: Parallel conductivity prefactor.
        gamma_sheath: Dimensionless sheath heat transmission coefficient.
        p0: Constant pressure closure value.
        T_up: Upstream electron temperature in eV.
        S_E: Volumetric energy source term as a function of position.
    """

    L: float = 20.0
    kappa_parallel: float = 1000.0
    gamma_sheath: float = 7.0
    p0: float = 2.0e21
    T_up: float = 80.0
    S_E: Callable = field(default=lambda s: 0.0)

    @property
    def alpha(self) -> float:
        """Return the sheath heat-flux coefficient alpha.

        Stangeby Eq.(11.16) gives: n_u T_u = 2 n_t T_t (含靶板动压项).
        So n_t = p0 / (2 * T_t), and the target sheath heat flux is:

            q_t = gamma * e * n_t * T_t * sqrt(T_t / m_i)
                = gamma * e^(3/2) * p0 / (2 * sqrt(m_i)) * T_t^(1/2)

        therefore:

            alpha = gamma * e^(3/2) * p0 / (2 * sqrt(m_i)).
        """
        import numpy as np
        from .constants import E_CHARGE, M_I

        return self.gamma_sheath * (E_CHARGE ** 1.5) * self.p0 / (2.0 * np.sqrt(M_I))

    def clone_with(self, **kwargs) -> "SOLConfig":
        """Return a copy with selected fields replaced."""
        d = dict(self.__dict__)
        d.update(kwargs)
        return SOLConfig(**d)


@dataclass
class SolverConfig:
    """Numerical solver settings shared by FD and PINN workflows."""

    n_points: int = 200
    max_iter: int = 1000
    tol: float = 1e-8
    omega: float = 0.5
