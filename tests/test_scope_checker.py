"""Tests for bid_engine.scope_checker.

Simulates Tyler's actual v1 → v2 evolution on Park Avenue Elementary School:
v1 missed scaffolding, fencing, and overhead shed protection; v2 added all
three. The checker should flag all 3 on v1 and pass clean on v2.
"""

from __future__ import annotations

import pytest

from bid_engine.pricing import DEFAULT_WAGES, PrevailingWage, ScopeItem
from bid_engine.scope_checker import (
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    ProjectConfig,
    ScopeAlert,
    check_scope,
)


# ---------------------------------------------------------------------------
# Fixtures — Tyler v1 / v2 simulations
# ---------------------------------------------------------------------------


def make_v1_scope() -> list[ScopeItem]:
    """Tyler's v1: extracted WS/R scope only — no scaffolding/fence/shed."""
    return [
        ScopeItem("WS1", 170.3, "LF", description="Clean Cast Stone Bands"),
        ScopeItem("WS5", 23.0, "EA", description="Refasten Loose Grille"),
        ScopeItem("WS8", 28.1, "LF", description="Rake Out and Re-point Brick Joints"),
        ScopeItem("WS15", 126.7, "SF", description="Remove Loose Brick around Lintel"),
        ScopeItem("R1", 2.0, "EA", description="Remove Door"),
    ]


def make_v2_scope() -> list[ScopeItem]:
    """Tyler's v2: v1 + the three items he was told to add."""
    return make_v1_scope() + [
        ScopeItem(
            "SCAFFOLD",
            1.0,
            "LS",
            description="Supported pipe scaffolding around building perimeter",
        ),
        ScopeItem(
            "FENCE",
            1.0,
            "LS",
            description="6' chain link construction fence, 15' setback",
        ),
        ScopeItem(
            "SHED",
            1.0,
            "EA",
            description="Overhead shed protection at all exit doors",
        ),
    ]


# ---------------------------------------------------------------------------
# Core v1 / v2 behavior
# ---------------------------------------------------------------------------


def test_v1_catches_missing_scaffolding_fence_shed():
    alerts = check_scope(make_v1_scope())
    ids = {a.item_id for a in alerts}
    assert "scaffolding" in ids
    assert "fencing" in ids
    assert "shed_protection" in ids
    assert all(a.severity == SEVERITY_CRITICAL for a in alerts)
    assert len(alerts) == 3


def test_v2_passes_clean():
    alerts = check_scope(make_v2_scope())
    assert alerts == []


def test_v2_individual_items_satisfy_each_check():
    """Removing any one of the three v2 items should re-fire just that alert."""
    base_v2 = make_v2_scope()
    for missing_code, expected_id in [
        ("SCAFFOLD", "scaffolding"),
        ("FENCE", "fencing"),
        ("SHED", "shed_protection"),
    ]:
        partial = [si for si in base_v2 if si.code != missing_code]
        alerts = check_scope(partial)
        fired = {a.item_id for a in alerts}
        assert fired == {expected_id}, f"removed {missing_code}: got {fired}"


# ---------------------------------------------------------------------------
# Keyword matching variants
# ---------------------------------------------------------------------------


def test_keyword_match_is_case_insensitive():
    items = [
        ScopeItem("X", 1, "LS", description="SCAFFOLDING — pipe, full perimeter"),
        ScopeItem("Y", 1, "LS", description="CHAIN LINK FENCING"),
        ScopeItem("Z", 1, "EA", description="OVERHEAD SHED at egress"),
    ]
    assert check_scope(items) == []


def test_keyword_match_on_code_alone():
    """Description may be empty — code itself should be searched."""
    items = [
        ScopeItem("SCAFFOLDING_LS", 1, "LS"),
        ScopeItem("FENCE_PERIMETER", 1, "LS"),
        ScopeItem("SHED_AT_DOORS", 1, "EA"),
    ]
    assert check_scope(items) == []


# ---------------------------------------------------------------------------
# Boom lift check (multi-story)
# ---------------------------------------------------------------------------


def test_boom_lift_flagged_on_multi_story_without_lift():
    alerts = check_scope(make_v2_scope(), ProjectConfig(stories=3))
    boom = [a for a in alerts if a.item_id == "boom_lift"]
    assert len(boom) == 1
    assert boom[0].severity == SEVERITY_WARNING


def test_boom_lift_not_flagged_on_single_story():
    alerts = check_scope(make_v2_scope(), ProjectConfig(stories=1))
    assert not any(a.item_id == "boom_lift" for a in alerts)


def test_boom_lift_satisfied_when_present():
    scope = make_v2_scope() + [
        ScopeItem("LIFT", 1.0, "LS", description="Boom lift alternate to scaffolding")
    ]
    alerts = check_scope(scope, ProjectConfig(stories=4))
    assert not any(a.item_id == "boom_lift" for a in alerts)


def test_boom_lift_invalid_stories_logs_and_skips(caplog):
    with caplog.at_level("WARNING"):
        alerts = check_scope(make_v2_scope(), ProjectConfig(stories=-1))
    assert not any(a.item_id == "boom_lift" for a in alerts)
    assert any("invalid stories" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Prevailing wage check (NY)
# ---------------------------------------------------------------------------


def test_ny_project_without_wage_rates_alerts():
    alerts = check_scope(make_v2_scope(), ProjectConfig(state="NY"))
    fired = [a for a in alerts if a.item_id == "prevailing_wage"]
    assert len(fired) == 1
    assert fired[0].severity == SEVERITY_CRITICAL


def test_ny_project_with_correct_wages_passes():
    alerts = check_scope(
        make_v2_scope(),
        ProjectConfig(state="NY", wage_rates=DEFAULT_WAGES),
    )
    assert not any(a.item_id.startswith("prevailing_wage") for a in alerts)


def test_ny_project_with_wrong_wage_flagged():
    wages = dict(DEFAULT_WAGES)
    wages["bricklayer"] = PrevailingWage("bricklayer", 30.00, 20.00)  # $50 — way under
    alerts = check_scope(
        make_v2_scope(),
        ProjectConfig(state="NY", wage_rates=wages),
    )
    fired = [a for a in alerts if a.item_id == "prevailing_wage:bricklayer"]
    assert len(fired) == 1
    assert "does not match" in fired[0].description


def test_ny_project_missing_one_trade_flagged():
    wages = {k: v for k, v in DEFAULT_WAGES.items() if k != "ironworker"}
    alerts = check_scope(
        make_v2_scope(),
        ProjectConfig(state="NY", wage_rates=wages),
    )
    fired = [a for a in alerts if a.item_id == "prevailing_wage:ironworker"]
    assert len(fired) == 1


def test_non_ny_project_skips_wage_check():
    alerts = check_scope(make_v2_scope(), ProjectConfig(state="CA"))
    assert not any(a.item_id.startswith("prevailing_wage") for a in alerts)


def test_state_check_is_case_and_whitespace_insensitive():
    alerts = check_scope(make_v2_scope(), ProjectConfig(state="  ny  "))
    assert any(a.item_id == "prevailing_wage" for a in alerts)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_scope_still_returns_required_alerts(caplog):
    with caplog.at_level("WARNING"):
        alerts = check_scope([])
    fired_ids = {a.item_id for a in alerts}
    assert fired_ids == {"scaffolding", "fencing", "shed_protection"}
    assert any("empty scope" in r.message for r in caplog.records)


def test_non_iterable_input_raises():
    with pytest.raises(TypeError, match="must be iterable"):
        check_scope(42)  # type: ignore[arg-type]


def test_scope_alert_fields_populated():
    alert = check_scope(make_v1_scope())[0]
    assert isinstance(alert, ScopeAlert)
    assert alert.item_id
    assert alert.severity in (SEVERITY_CRITICAL, SEVERITY_WARNING, "info")
    assert alert.description
    assert alert.suggested_action


def test_check_combines_required_and_boom_and_wage():
    """v1 + multi-story NY w/o wages = 3 critical + 1 warning + 1 critical = 5."""
    alerts = check_scope(make_v1_scope(), ProjectConfig(state="NY", stories=3))
    ids = {a.item_id for a in alerts}
    assert "scaffolding" in ids
    assert "fencing" in ids
    assert "shed_protection" in ids
    assert "boom_lift" in ids
    assert "prevailing_wage" in ids
    assert len(alerts) == 5
