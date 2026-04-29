from __future__ import annotations

from src.utils.logger import logger


def select_best_model(cv_results: dict[str, dict[str, float]]) -> str:
    """
    Given a mapping of model_name -> {mape, rmse, ...}, return the champion model name.
    Primary sort: avg_mape ascending. Tiebreaker: avg_rmse ascending.
    """
    if not cv_results:
        raise ValueError("cv_results is empty — nothing to select from")

    ranked = sorted(
        cv_results.items(),
        key=lambda kv: (
            kv[1].get("mape", float("inf")),
            kv[1].get("rmse", float("inf")),
        ),
    )

    champion = ranked[0][0]
    logger.info(
        "Champion selected",
        champion=champion,
        mape=ranked[0][1].get("mape"),
        rmse=ranked[0][1].get("rmse"),
        runners_up=[r[0] for r in ranked[1:]],
    )
    return champion


def rank_models(cv_results: dict[str, dict[str, float]]) -> list[dict]:
    """Return models ranked by mape ascending, with rank field added."""
    ranked = sorted(
        cv_results.items(),
        key=lambda kv: (
            kv[1].get("mape", float("inf")),
            kv[1].get("rmse", float("inf")),
        ),
    )
    return [
        {"rank": i + 1, "model": name, **metrics, "is_champion": i == 0}
        for i, (name, metrics) in enumerate(ranked)
    ]
