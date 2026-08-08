"""Low-level numerical primitives for LiouScope."""

from .adjoint import alicki_adjoint, hs_adjoint
from .kronecker import unvec, vec
from .linalg import (
    EigenDecomposition,
    eig_nonhermitian,
    is_density_matrix,
    is_hermitian,
    support_check,
)
from .pseudospec import pseudospectral_radius, pseudospectrum_extent
from .resolvent import resolvent_apply_superlu, resolvent_norm
from .scale import rate_scale

__all__ = [
    "EigenDecomposition",
    "alicki_adjoint",
    "eig_nonhermitian",
    "hs_adjoint",
    "is_density_matrix",
    "is_hermitian",
    "pseudospectral_radius",
    "pseudospectrum_extent",
    "rate_scale",
    "resolvent_apply_superlu",
    "resolvent_norm",
    "support_check",
    "unvec",
    "vec",
]
