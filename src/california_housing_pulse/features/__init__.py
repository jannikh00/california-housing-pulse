"""Target construction and leakage-safe feature engineering.

``target`` executes the frozen Milestone 2 contract in ``configs/target.yaml``.
``spec``, ``transforms`` and ``build`` execute the Milestone 3 feature
specification in ``configs/features.yaml`` — declaration, vocabulary, and
assembly respectively.
"""

from __future__ import annotations

from .build import build_features, feature_names, write_availability_report
from .spec import load_feature_contract
from .target import add_target, load_contract, modeling_rows

__all__ = [
    "add_target",
    "build_features",
    "feature_names",
    "load_contract",
    "load_feature_contract",
    "modeling_rows",
    "write_availability_report",
]
