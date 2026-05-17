"""Scope completeness checker — flag missing required items in a bid scope.

Calibrated against Reference's actual v1 → v2 corrections on the Park Avenue
Elementary School project, where the v1 markup missed scaffolding, fencing,
and overhead shed protection. The checker accepts a list of ScopeItems plus
optional ProjectConfig (state, stories, wage rates) and returns a list of
ScopeAlerts the contractor should address before submitting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from bid_engine.pricing import DEFAULT_WAGES, PrevailingWage, ScopeItem

logger = logging.getLogger(__name__)

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

WAGE_TOLERANCE = 0.01  # dollars — wage rates considered equal within 1¢


@dataclass(frozen=True)
class ScopeAlert:
    item_id: str
    severity: str
    description: str
    suggested_action: str


@dataclass(frozen=True)
class ProjectConfig:
    state: str = ""
    stories: int = 1
    wage_rates: Mapping[str, PrevailingWage] | None = None


@dataclass(frozen=True)
class _RequiredItem:
    item_id: str
    keywords: tuple[str, ...]
    description: str
    suggested_action: str


# Required on every exterior renovation bid (per spec + Reference's email thread).
_REQUIRED_ITEMS: tuple[_RequiredItem, ...] = (
    _RequiredItem(
        item_id="scaffolding",
        keywords=("scaffold",),
        description="Supported scaffolding (pipe scaffolding) for elevation work",
        suggested_action=(
            "Add a scaffolding line item — pipe scaffolding per elevation, "
            "priced LS or by sq ft of facade"
        ),
    ),
    _RequiredItem(
        item_id="fencing",
        keywords=("fence", "fencing", "chain link"),
        description="Chain link fence around construction area (6' height, 15' from building)",
        suggested_action=(
            "Add chain link fence line item — 6' height, 15' setback from building"
        ),
    ),
    _RequiredItem(
        item_id="shed_protection",
        keywords=("shed", "overhead protection", "canopy"),
        description="Overhead shed protection at all exit doors",
        suggested_action=(
            "Add overhead shed protection — required at all exit doors per DOB"
        ),
    ),
)

_BOOM_KEYWORDS: tuple[str, ...] = ("boom", "aerial lift", "aerial")


def _searchable_text(items: Sequence[ScopeItem]) -> str:
    """Concatenate codes + descriptions into one lowercased blob for substring search."""
    return " ".join(f"{si.code} {si.description}" for si in items).lower()


def _check_required_keywords(items: Sequence[ScopeItem]) -> list[ScopeAlert]:
    haystack = _searchable_text(items)
    alerts: list[ScopeAlert] = []
    for required in _REQUIRED_ITEMS:
        if any(kw in haystack for kw in required.keywords):
            continue
        logger.warning("scope check: missing %s", required.item_id)
        alerts.append(
            ScopeAlert(
                item_id=required.item_id,
                severity=SEVERITY_CRITICAL,
                description=required.description,
                suggested_action=required.suggested_action,
            )
        )
    return alerts


def _check_boom_lift(
    items: Sequence[ScopeItem], config: ProjectConfig
) -> list[ScopeAlert]:
    if config.stories < 1:
        logger.warning(
            "scope check: invalid stories=%d on ProjectConfig — skipping boom lift check",
            config.stories,
        )
        return []
    if config.stories <= 1:
        return []
    haystack = _searchable_text(items)
    if any(kw in haystack for kw in _BOOM_KEYWORDS):
        return []
    logger.warning(
        "scope check: multi-story building (%d stories) but no boom lift item",
        config.stories,
    )
    return [
        ScopeAlert(
            item_id="boom_lift",
            severity=SEVERITY_WARNING,
            description=(
                f"Multi-story building ({config.stories} stories) — boom/aerial "
                "lift recommended"
            ),
            suggested_action=(
                "Add a boom lift / aerial lift line item as an alternate to "
                "pipe scaffolding for upper-story work"
            ),
        )
    ]


def _check_prevailing_wage(config: ProjectConfig) -> list[ScopeAlert]:
    if config.state.strip().upper() != "NY":
        return []
    if config.wage_rates is None:
        logger.warning(
            "scope check: NY project but no wage_rates supplied on ProjectConfig"
        )
        return [
            ScopeAlert(
                item_id="prevailing_wage",
                severity=SEVERITY_CRITICAL,
                description=(
                    "NY State project: prevailing wage rates required but "
                    "no wage_rates supplied"
                ),
                suggested_action=(
                    "Provide NY DOL prevailing wage rates on ProjectConfig.wage_rates"
                ),
            )
        ]
    alerts: list[ScopeAlert] = []
    for trade, expected in DEFAULT_WAGES.items():
        actual = config.wage_rates.get(trade)
        if actual is None:
            alerts.append(
                ScopeAlert(
                    item_id=f"prevailing_wage:{trade}",
                    severity=SEVERITY_CRITICAL,
                    description=(
                        f"NY State project: no wage rate provided for {trade}"
                    ),
                    suggested_action=(
                        f"Add NY DOL prevailing wage rate for {trade} "
                        f"(expected ${expected.total:.2f}/hr)"
                    ),
                )
            )
            continue
        if abs(actual.total - expected.total) > WAGE_TOLERANCE:
            alerts.append(
                ScopeAlert(
                    item_id=f"prevailing_wage:{trade}",
                    severity=SEVERITY_CRITICAL,
                    description=(
                        f"NY State project: {trade} wage ${actual.total:.2f}/hr "
                        f"does not match NY DOL rate ${expected.total:.2f}/hr"
                    ),
                    suggested_action=(
                        f"Update {trade} wage to NY DOL prevailing rate "
                        f"${expected.total:.2f}/hr"
                    ),
                )
            )
    return alerts


def check_scope(
    scope_items: Iterable[ScopeItem],
    config: ProjectConfig | None = None,
) -> list[ScopeAlert]:
    """Run all scope-completeness checks; return the alerts that fired."""
    try:
        items = list(scope_items)
    except TypeError as exc:
        raise TypeError(
            f"scope_items must be iterable of ScopeItem, got {type(scope_items).__name__}"
        ) from exc

    if not items:
        logger.warning("check_scope: empty scope item list")

    cfg = config or ProjectConfig()

    alerts: list[ScopeAlert] = []
    alerts.extend(_check_required_keywords(items))
    alerts.extend(_check_boom_lift(items, cfg))
    alerts.extend(_check_prevailing_wage(cfg))

    logger.info(
        "scope check complete: %d alert(s) across %d scope item(s)",
        len(alerts),
        len(items),
    )
    return alerts
