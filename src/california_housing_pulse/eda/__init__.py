"""Decision-oriented exploratory analysis (Milestone 2).

``analysis`` measures, ``figures`` draws, ``report`` renders. Nothing here is
imported by the data pipeline, so a missing plotting backend can never break a
data rebuild.
"""

from __future__ import annotations

from .report import build_eda

__all__ = ["build_eda"]
