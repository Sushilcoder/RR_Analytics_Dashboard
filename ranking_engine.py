"""Provisional leaderboard engine for SupeRR Selector Season 2.

The scores are exploratory only. Pressure is a context proxy (final 5 overs or
late innings with <=4 wickets remaining), not a validated win-probability or
required-run-rate model. Bowling conceded runs use a provisional extras
interpretation and must not be treated as official scorecard figures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CREDITED_WICKET_TYPES = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def _to_bool(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    ).fillna(False).astype(bool)


def _pct_by_format(df: pd.DataFrame, col: str, higher_is_better: bool) -> pd.Series:
    return df.groupby("format", dropna=False)[col].rank(pct=True, ascending=higher_is_better)


def _aggregate_players(df: pd.DataFrame, score_col: str, metrics: dict[str, str]) -> pd.DataFrame:
    rows = []
    for (pid, name), group in df.groupby(["player_id", "player"], dropna=False):
        weights = np.sqrt(group["pressure_balls"].clip(lower=1).to_numpy(dtype=float))
        row = {"player_id": str(pid), "player": str(name), "score": float(np.average(group[score_col], weights=weights))}
        for output_col, source_col in metrics.items():
            if source_col in group.columns:
                if source_col in {"pressure_balls", "total_balls", "pressure_runs", "pressure_runs_proxy", "wickets", "pressure_matches"}:
                    row[output_col] = float(group[source_col].sum())
                else:
                    vals = pd.to_numeric(group[source_col], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(vals)
                    row[output_col] = float(np.average(vals[valid], weights=weights[valid])) if valid.any() else np.nan
        row["formats"] = ", ".join(sorted(group["format"].dropna().astype(str).unique()))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def calculate_provisional_rankings(
    balls: pd.DataFrame,
    matches: pd.DataFrame,
    min_pressure_balls: int = 60,
    min_total_balls: int = 200,
    min_pressure_matches: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return provisional top-10 and full leaderboard frames plus summary."""
    required_ball = {
        "match_id", "innings", "overid", "innings_ball_num", "extras_type", "runs",
        "is_wicket", "dismissed_player_id", "striker_id", "striker_name", "bowler_id", "bowler_name",
        "is_four", "is_six", "dismissal_type",
    }
    required_match = {"match_id", "first_inn_revised_overs", "second_inn_revised_overs"}
    if balls.empty or matches.empty or not required_ball.issubset(balls.columns) or not required_match.issubset(matches.columns):
        return pd.DataFrame(), pd.DataFrame(), {"error": "Required columns are missing. Check that ball_by_ball.csv and matches.csv are loaded."}

    b = balls.copy()
    m = matches.copy()
    for col in ["innings", "overid", "innings_ball_num", "runs"]:
        b[col] = pd.to_numeric(b[col], errors="coerce")
    b["is_wicket_bool"] = _to_bool(b["is_wicket"])
    b["is_four_bool"] = _to_bool(b["is_four"])
    b["is_six_bool"] = _to_bool(b["is_six"])
    b["extras_norm"] = b["extras_type"].astype("string").str.strip().str.lower()
    b["illegal"] = b["extras_norm"].isin(["wide", "overstep"])
    b["legal_ball"] = ~b["illegal"]
    b["runs_num"] = b["runs"].fillna(0.0)

    # Attach competition format and revised-over metadata to every delivery.
    keep_match_cols = [c for c in ["match_id", "comp_id", "match_format", "first_inn_revised_overs", "second_inn_revised_overs"] if c in m.columns]
    b = b.merge(m[keep_match_cols].drop_duplicates("match_id"), on="match_id", how="left", suffixes=("", "_match"))
    if "match_format" not in b.columns:
        b["match_format"] = "Unknown"
    b["format"] = b["match_format"].fillna("Unknown").astype(str)

    b["scheduled_overs"] = 20.0
    for innings_no, revised_col in [(1, "first_inn_revised_overs"), (2, "second_inn_revised_overs")]:
        if revised_col in b.columns:
            revised = pd.to_numeric(b[revised_col], errors="coerce")
            use = b["innings"].eq(innings_no) & revised.notna() & revised.gt(0)
            b.loc[use, "scheduled_overs"] = revised.loc[use]
    b["scheduled_balls"] = (b["scheduled_overs"] * 6).round().astype(int)

    b["sort_ball"] = b["innings_ball_num"]
    missing_seq = b["sort_ball"].isna()
    b.loc[missing_seq, "sort_ball"] = b.loc[missing_seq, "overid"] * 10
    b = b.sort_values(["match_id", "innings", "sort_ball"], kind="stable").copy()
    keys = ["match_id", "innings"]
    b["legal_before"] = b.groupby(keys, dropna=False)["legal_ball"].transform(
        lambda s: s.astype(int).cumsum().shift(fill_value=0)
    )
    b["wickets_before"] = b.groupby(keys, dropna=False)["is_wicket_bool"].transform(
        lambda s: s.astype(int).cumsum().shift(fill_value=0)
    )
    b["balls_remaining"] = (b["scheduled_balls"] - b["legal_before"]).clip(lower=0)
    b["wickets_remaining"] = (10 - b["wickets_before"]).clip(lower=0)

    # Provisional pressure proxy: final 5 scheduled overs OR (after 6 overs)
    # the side has four or fewer wickets remaining. This avoids unverified
    # score reconstruction / required-run-rate inputs, but is not a final model.
    b["pressure"] = (b["balls_remaining"].le(30) | (b["wickets_remaining"].le(4) & b["legal_before"].ge(36)))
    b = b[b["innings"].isin([1, 2])].copy()  # Exclude Super Overs from primary leaderboard.
    b["bat_ball"] = ~b["extras_norm"].eq("wide").fillna(False)  # No-balls count as faced; wides do not.
    b["boundary"] = b["is_four_bool"] | b["is_six_bool"]
    b["dot_bat"] = b["runs_num"].eq(0) & b["bat_ball"]
    b["bat_dismissal"] = b["is_wicket_bool"] & b["dismissed_player_id"].astype("string").eq(b["striker_id"].astype("string")).fillna(False)

    # ------------------------------- Batting ---------------------------------
    bat_rows = []
    for (pid, pname, fmt), g in b.groupby(["striker_id", "striker_name", "format"], dropna=False):
        if pd.isna(pid) or pd.isna(pname):
            continue
        p = g[g["pressure"]]
        n = g[~g["pressure"]]
        pb = int(p["bat_ball"].sum())
        total_balls = int(g["bat_ball"].sum())
        if pb <= 0:
            continue
        pressure_runs = float(p["runs_num"].sum())
        normal_balls = int(n["bat_ball"].sum())
        normal_runs = float(n["runs_num"].sum())
        bat_rows.append({
            "player_id": str(pid), "player": str(pname), "format": fmt,
            "pressure_balls": pb, "total_balls": total_balls,
            "pressure_runs": pressure_runs,
            "pressure_sr": 100 * pressure_runs / pb,
            "normal_sr": 100 * normal_runs / normal_balls if normal_balls else np.nan,
            "boundary_rate": 100 * int((p.loc[p["bat_ball"], "boundary"]).sum()) / pb,
            "dot_rate": 100 * int(p["dot_bat"].sum()) / pb,
            "dismissals": int(p["bat_dismissal"].sum()),
            "dismissal_rate": 100 * int(p["bat_dismissal"].sum()) / pb,
            "pressure_matches": int(p.loc[p["bat_ball"], "match_id"].nunique()),
        })
    bat = pd.DataFrame(bat_rows)
    if not bat.empty:
        bat = bat[(bat.pressure_balls >= 30) & (bat.total_balls >= 100) & (bat.pressure_matches >= 3)].copy()
        if not bat.empty:
            bat["uplift"] = bat["pressure_sr"] - bat["normal_sr"]
            bat["sr_pct"] = _pct_by_format(bat, "pressure_sr", True)
            bat["boundary_pct"] = _pct_by_format(bat, "boundary_rate", True)
            bat["dot_pct"] = _pct_by_format(bat, "dot_rate", False)
            bat["dismissal_pct"] = _pct_by_format(bat, "dismissal_rate", False)
            bat["uplift_pct"] = _pct_by_format(bat, "uplift", True)
            bat["score"] = 100 * (
                0.35 * bat["sr_pct"] + 0.20 * bat["boundary_pct"] + 0.15 * bat["dot_pct"]
                + 0.15 * bat["dismissal_pct"] + 0.15 * bat["uplift_pct"]
            )
            bat_all = _aggregate_players(bat, "score", {
                "pressure_balls": "pressure_balls", "total_balls": "total_balls",
                "pressure_runs": "pressure_runs", "pressure_sr": "pressure_sr",
                "pressure_matches": "pressure_matches", "avg_uplift": "uplift",
            })
            bat_all = bat_all[(bat_all.pressure_balls >= min_pressure_balls)
                              & (bat_all.total_balls >= min_total_balls)
                              & (bat_all.pressure_matches >= min_pressure_matches)].copy()
            bat_all = bat_all.sort_values(["score", "pressure_balls"], ascending=[False, False]).reset_index(drop=True)
        else:
            bat_all = pd.DataFrame()
    else:
        bat_all = pd.DataFrame()

    # ------------------------------- Bowling ---------------------------------
    bowl_rows = []
    b["bowl_conceded_proxy"] = b["runs_num"] + b["extras_norm"].isin(["wide", "overstep"]).astype(int)
    b["credited_wicket"] = b["dismissal_type"].isin(CREDITED_WICKET_TYPES)
    b["boundary"] = b["is_four_bool"] | b["is_six_bool"]
    for (pid, pname, fmt), g in b.groupby(["bowler_id", "bowler_name", "format"], dropna=False):
        if pd.isna(pid) or pd.isna(pname):
            continue
        p = g[g["pressure"] & g["legal_ball"]]
        n = g[(~g["pressure"]) & g["legal_ball"]]
        pb = int(len(p))
        total_balls = int(g["legal_ball"].sum())
        if pb <= 0:
            continue
        pressure_runs_proxy = float(p["bowl_conceded_proxy"].sum())
        normal_runs_proxy = float(n["bowl_conceded_proxy"].sum())
        normal_balls = int(len(n))
        wickets = int(p["credited_wicket"].sum())
        boundaries = int(p["boundary"].sum())
        bowl_rows.append({
            "player_id": str(pid), "player": str(pname), "format": fmt,
            "pressure_balls": pb, "total_balls": total_balls,
            "pressure_runs_proxy": pressure_runs_proxy,
            "pressure_econ_proxy": 6 * pressure_runs_proxy / pb,
            "normal_econ_proxy": 6 * normal_runs_proxy / normal_balls if normal_balls else np.nan,
            "wickets": wickets, "wicket_rate_per_6": 6 * wickets / pb,
            "boundary_rate": 100 * boundaries / pb,
            "pressure_matches": int(p["match_id"].nunique()),
        })
    bowl = pd.DataFrame(bowl_rows)
    if not bowl.empty:
        bowl = bowl[(bowl.pressure_balls >= 30) & (bowl.total_balls >= 100) & (bowl.pressure_matches >= 3)].copy()
        if not bowl.empty:
            bowl["economy_improvement"] = bowl["normal_econ_proxy"] - bowl["pressure_econ_proxy"]
            bowl["econ_pct"] = _pct_by_format(bowl, "pressure_econ_proxy", False)
            bowl["wicket_pct"] = _pct_by_format(bowl, "wicket_rate_per_6", True)
            bowl["boundary_pct"] = _pct_by_format(bowl, "boundary_rate", False)
            bowl["improvement_pct"] = _pct_by_format(bowl, "economy_improvement", True)
            bowl["score"] = 100 * (
                0.40 * bowl["econ_pct"] + 0.30 * bowl["wicket_pct"]
                + 0.15 * bowl["boundary_pct"] + 0.15 * bowl["improvement_pct"]
            )
            bowl_all = _aggregate_players(bowl, "score", {
                "pressure_balls": "pressure_balls", "total_balls": "total_balls",
                "pressure_runs_proxy": "pressure_runs_proxy", "pressure_econ_proxy": "pressure_econ_proxy",
                "wickets": "wickets", "pressure_matches": "pressure_matches",
                "economy_improvement": "economy_improvement",
            })
            bowl_all = bowl_all[(bowl_all.pressure_balls >= min_pressure_balls)
                                & (bowl_all.total_balls >= min_total_balls)
                                & (bowl_all.pressure_matches >= min_pressure_matches)].copy()
            bowl_all = bowl_all.sort_values(["score", "pressure_balls"], ascending=[False, False]).reset_index(drop=True)
        else:
            bowl_all = pd.DataFrame()
    else:
        bowl_all = pd.DataFrame()

    summary = {
        "matches": int(b["match_id"].nunique()),
        "delivery_rows": int(len(b)),
        "pressure_delivery_rows": int(b["pressure"].sum()),
        "pressure_share_pct": 100 * float(b["pressure"].mean()) if len(b) else 0.0,
        "eligible_batters": int(len(bat_all)),
        "eligible_bowlers": int(len(bowl_all)),
        "min_pressure_balls": int(min_pressure_balls),
        "min_total_balls": int(min_total_balls),
        "min_pressure_matches": int(min_pressure_matches),
        "error": None,
    }
    return bat_all.head(10), bowl_all.head(10), summary
