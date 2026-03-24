"""
U7 MOPSO Dispatcher – compatibility shim.

MOPSOPlanner lives in U11_mopso_dispatcher; this module re-exports it
so that U10_candidate_generator can import from ``U7_mopso_dispatcher``
as documented.
"""
from U11_mopso_dispatcher import MOPSOPlanner  # noqa: F401

__all__ = ["MOPSOPlanner"]
