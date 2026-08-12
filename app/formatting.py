"""
Shared display-formatting helpers for the Streamlit frontend.

Deliberately kept outside app/pages/: any .py file placed directly inside
app/pages/ is auto-registered by Streamlit as a sidebar page, and numeric-
prefixed page filenames (e.g. "2_Modeles.py") aren't valid Python
identifiers anyway, so they can't be imported for unit testing. Living next
to app/streamlit_app.py, this module is importable both by Streamlit (which
inserts that directory into sys.path for the main script and every page) and
by pytest (via the `app` namespace package from the repo root).
"""

from __future__ import annotations

import pandas as pd


def format_metric(value, decimals: int = 3) -> str:
    """
    Format a single scalar metric for display.

    Returns "—" when `value` is None, pandas NA/NaN, or otherwise not
    convertible to a float (e.g. left over from a mixed classification /
    regression pandas.DataFrame where the metric doesn't apply to this run).
    """
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_f1_pair(train_value, test_value) -> str:
    """
    Format an "F1 train / test" pair for display.

    Returns "—" as soon as either side is None or pandas NA/NaN, instead of
    the "nan / nan" that pandas produces for regression runs (which don't
    log F1).
    """
    if (
        train_value is None
        or test_value is None
        or pd.isna(train_value)
        or pd.isna(test_value)
    ):
        return "—"
    return f"{float(train_value):.3f} / {float(test_value):.3f}"
