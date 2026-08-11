"""
Data contract for Kitchener-Waterloo water-main break risk classification.

Phase 1 scope: breaks-only physical attributes (material, diameter/size, install year).
Length from Water_Mains is deferred to a later phase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Raw break-incident columns (Open Kitchener / ArcGIS extract)
# ---------------------------------------------------------------------------
RAW_REQUIRED_COLUMNS = [
    "ASSETID",
    "INCIDENT_DATE",
    "ASSET_MATERIAL",
    "ASSET_SIZE",
    "ASSET_YEAR_INSTALLED",
]

RAW_OPTIONAL_COLUMNS = [
    "OBJECTID",
    "WATBREAKINCIDENTID",
    "STREET",
    "ROADSEGMENTID",
    "X",
    "Y",
    "BREAK_TYPE",
    "BREAK_NATURE",
    "BREAK_APPARENT_CAUSE",
    "BREAK_CATEGORIZATION",
    "ASSET_EXISTS",
    "ASSET_DEPTH",
    "FROST_DEPTH",
    "NEW_SECTION_LENGTH",
    "GLOBALID",
]

# Columns that encode post-break / repair knowledge and must NEVER be used as
# model features for a snapshot dated at prediction time t.
LEAKAGE_FORBIDDEN_COLUMNS = [
    "BREAK_TYPE",
    "BREAK_NATURE",
    "BREAK_APPARENT_CAUSE",
    "BREAK_CATEGORIZATION",
    "REPAIR_TYPE",
    "NEW_SECTION_LENGTH",
    "POSITIVE_PRESSURE_MAINTANED",
    "AIR_GAP_MAINTANED",
    "DISINFECTED",
    "MECHANICAL_REMOVAL",
    "FLUSHING_EXCAVATION",
    "HIGHER_VELOCITY_FLUSHING",
    "ANODE_INSTALLED",
    "STATUS",
    "STATUS_DATE",
    "RETURN_TO_NORMAL",
    "HOUR_IMPACTED",
    "UNITS_IMPACTED",
    "ROAD_CLOSED",
    "SIDEWALK_CLOSED",
]

# ---------------------------------------------------------------------------
# Snapshot / modeling schema (generated rows)
# ---------------------------------------------------------------------------
SNAPSHOT_ID_COLUMNS = [
    "asset_id",
    "snapshot_date",
]

CATEGORICAL_COLUMNS = [
    "material",
]

# Features present on Phase-1 snapshot rows (before preprocessing enrichment).
SNAPSHOT_NUMERIC_COLUMNS = [
    "diameter_mm",
    "install_year",
    "age_years",
    "prior_break_count",
]

# Features added in Phase-2 preprocessing from past-only break history.
ENGINEERED_NUMERIC_COLUMNS = [
    "years_since_last_break",
]

NUMERIC_COLUMNS = SNAPSHOT_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS

# Length deferred. Kept here as a documented future field.
DEFERRED_NUMERIC_COLUMNS = [
    "length_m",
]

# Non-feature columns retained for auditing / temporal splits only.
META_COLUMNS = [
    "asset_id",
    "snapshot_date",
    "horizon_years",
    "street",
    "snapshot_origin",
]

TARGET_COLUMN = "break_within_horizon"
TARGET_VALUES = (0, 1)

# Regression target: years from snapshot_date until the next recorded break.
# Only defined for non-censored rows (a future break exists); NaN otherwise.
REGRESSION_TARGET_COLUMN = "years_until_break"

# Model input features only (no IDs, no post-break fields, no target).
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

SNAPSHOT_FEATURE_COLUMNS = CATEGORICAL_COLUMNS + SNAPSHOT_NUMERIC_COLUMNS

SNAPSHOT_OUTPUT_COLUMNS = (
    SNAPSHOT_ID_COLUMNS
    + SNAPSHOT_FEATURE_COLUMNS
    + [
        TARGET_COLUMN,
        "horizon_years",
        "street",
    ]
)

# Inference-time request fields (API / Streamlit). Physical + time-aware attrs.
INFERENCE_INPUT_COLUMNS = [
    "material",
    "diameter_mm",
    "install_year",
    "age_years",
    "prior_break_count",
    "years_since_last_break",
]
