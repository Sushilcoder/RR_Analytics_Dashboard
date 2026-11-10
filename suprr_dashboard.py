from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ranking_engine import calculate_provisional_rankings


# -----------------------------------------------------------------------------
# SupeRR Selector Season 2 – Round 1 | Local CSV analysis dashboard
# The app calculates clearly labelled provisional leaderboards. Final rankings
# remain blocked until organiser scoring definitions are verified.
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="SupeRR Selector | Data Analysis Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXPECTED_FILES = {
    "ball_by_ball": "ball_by_ball.csv",
    "matches": "matches.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "playing_eleven": "playing_eleven.csv",
    "player_mapping": "competition_team_player_mapping.csv",
}

REQUIRED_COLUMNS = {
    "ball_by_ball": ["match_id", "innings", "striker_id", "striker_name", "bowler_id", "bowler_name", "runs", "is_wicket"],
    "matches": ["match_id", "comp_id", "match_date", "result"],
    "competitions": ["comp_id", "comp_name", "match_format", "season"],
    "teams": ["comp_id", "team_id", "team_name"],
    "playing_eleven": ["match_id", "team_id", "player_id", "is_played"],
    "player_mapping": ["comp_id", "team_id", "player_id", "player_name"],
}

ID_COLUMNS = {
    "ball_by_ball": ["match_id", "batting_team_id", "bowling_team_id", "striker_id", "non_striker_id", "bowler_id", "dismissed_player_id"],
    "matches": ["match_id", "comp_id", "home_team_id", "away_team_id", "win_team_id"],
    "competitions": ["comp_id"],
    "teams": ["comp_id", "team_id"],
    "playing_eleven": ["match_id", "team_id", "player_id"],
    "player_mapping": ["comp_id", "team_id", "player_id"],
}


def _normalise_name(name: str) -> str:
    name = Path(name).name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def identify_dataset(filename: str) -> Optional[str]:
    n = _normalise_name(filename)
    if "ball_by_ball" in n or "ballbyball" in n or "delivery" in n:
        return "ball_by_ball"
    if "playing_eleven" in n or "playing_xi" in n or "playing11" in n:
        return "playing_eleven"
    if "competition_team_player" in n or "player_mapping" in n or "competition_player" in n:
        return "player_mapping"
    if "competitions" in n or "competition" in n:
        return "competitions"
    if "matches" in n or "match_data" in n:
        return "matches"
    if "teams" in n or "team_data" in n:
        return "teams"
    return None


def read_csv_bytes(data: bytes, filename: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False, dtype_backend="numpy_nullable")
        except TypeError:
            # Compatibility with older pandas versions.
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False)
            except Exception as exc:  # pragma: no cover - fallback
                last_error = exc
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read {filename}: {last_error}")


def add_dataset(datasets: Dict[str, pd.DataFrame], key: str, df: pd.DataFrame) -> None:
    # Keep identifiers as strings to avoid accidental numeric conversion.
    for col in ID_COLUMNS.get(key, []):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    datasets[key] = df


def load_uploaded_files(zip_upload, csv_uploads) -> Tuple[Dict[str, pd.DataFrame], list[str], list[str]]:
    datasets: Dict[str, pd.DataFrame] = {}
    messages: list[str] = []
    errors: list[str] = []

    if zip_upload is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(zip_upload.getvalue())) as zf:
                for member in zf.namelist():
                    if member.lower().endswith(".csv") and not member.startswith("__MACOSX/"):
                        key = identify_dataset(member)
                        if key:
                            try:
                                df = read_csv_bytes(zf.read(member), member)
                                add_dataset(datasets, key, df)
                                messages.append(f"Loaded **{member}** → {key.replace('_', ' ')} ({len(df):,} rows)")
                            except Exception as exc:
                                errors.append(f"{member}: {exc}")
        except zipfile.BadZipFile:
            errors.append("The uploaded ZIP file could not be opened. Please upload a valid ZIP archive.")

    # Individual CSV uploads take precedence over files with the same purpose in ZIP.
    for upload in csv_uploads or []:
        key = identify_dataset(upload.name)
        if not key:
            messages.append(f"Skipped unrecognised CSV filename: {upload.name}")
            continue
        try:
            df = read_csv_bytes(upload.getvalue(), upload.name)
            add_dataset(datasets, key, df)
            messages.append(f"Loaded **{upload.name}** → {key.replace('_', ' ')} ({len(df):,} rows)")
        except Exception as exc:
            errors.append(f"{upload.name}: {exc}")

    return datasets, messages, errors


def as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def as_bool(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="boolean")
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return series.astype("string").str.strip().str.lower().map(mapping).astype("boolean")


def fmt_num(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{int(value):,}"
    except Exception:
        return str(value)


def file_audit(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, expected in EXPECTED_FILES.items():
        df = datasets.get(key)
        if df is None:
            rows.append({"Dataset": expected, "Status": "Missing", "Rows": 0, "Columns": 0, "Missing cells": "—", "Missing required columns": ", ".join(REQUIRED_COLUMNS[key])})
            continue
        missing_required = [c for c in REQUIRED_COLUMNS[key] if c not in df.columns]
        rows.append({
            "Dataset": expected,
            "Status": "Loaded" if not missing_required else "Needs review",
            "Rows": len(df),
            "Columns": len(df.columns),
            "Missing cells": int(df.isna().sum().sum()),
            "Missing required columns": ", ".join(missing_required) if missing_required else "None",
        })
    return pd.DataFrame(rows)


def selected_scope(datasets: Dict[str, pd.DataFrame], chosen_comp_names: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = datasets.get("matches", pd.DataFrame()).copy()
    comps = datasets.get("competitions", pd.DataFrame()).copy()
    balls = datasets.get("ball_by_ball", pd.DataFrame()).copy()
    if matches.empty or comps.empty:
        return matches, balls, comps

    if "comp_id" in comps.columns and "comp_id" in matches.columns:
        comp_info = comps[[c for c in ["comp_id", "comp_name", "match_format", "season"] if c in comps.columns]].drop_duplicates("comp_id")
        matches = matches.merge(comp_info, on="comp_id", how="left", suffixes=("", "_competition"))

    if chosen_comp_names and "comp_name" in matches.columns:
        ids = matches.loc[matches["comp_name"].isin(chosen_comp_names), "comp_id"].dropna().astype(str).unique()
        matches = matches[matches["comp_id"].astype(str).isin(ids)].copy()

    if "match_id" in balls.columns and "match_id" in matches.columns:
        balls = balls[balls["match_id"].astype(str).isin(matches["match_id"].dropna().astype(str))].copy()
    return matches, balls, comps


def make_duplicate_report(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = {
        "matches": ["match_id"],
        "competitions": ["comp_id"],
        "teams": ["comp_id", "team_id"],
        "playing_eleven": ["match_id", "team_id", "player_id"],
        "player_mapping": ["comp_id", "team_id", "player_id"],
        "ball_by_ball": ["match_id", "innings", "overid", "over_ball_num"],
    }
    rows = []
    for name, cols in keys.items():
        df = datasets.get(name)
        if df is None:
            continue
        present = [c for c in cols if c in df.columns]
        if len(present) != len(cols):
            rows.append({"Dataset": EXPECTED_FILES[name], "Key checked": ", ".join(cols), "Duplicate rows": "Not checked – key column missing"})
        else:
            dup = int(df.duplicated(present, keep="first").sum())
            rows.append({"Dataset": EXPECTED_FILES[name], "Key checked": ", ".join(present), "Duplicate rows": dup})
    return pd.DataFrame(rows)


def metric_card(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label, value, help=help_text)


# ------------------------------- Header --------------------------------------
st.title("🏏 SupeRR Selector Season 2")
st.subheader("Round 1 · Cricket Pressure Performance Analysis Workspace")
st.caption("Upload the organiser's CSV files or one ZIP archive. Explore data quality, competition coverage and player records before building the pressure model.")

with st.sidebar:
    st.header("1 · Upload organiser data")
    zip_upload = st.file_uploader("Upload dataset ZIP", type=["zip"], help="Recommended: upload the ZIP containing all six CSV files.")
    csv_uploads = st.file_uploader("Or upload individual CSV files", type=["csv"], accept_multiple_files=True)
    st.divider()
    st.header("2 · Analysis scope")
    st.caption("After upload, select the competitions to include in the analysis.")


datasets, load_messages, load_errors = load_uploaded_files(zip_upload, csv_uploads)

if not datasets:
    st.info("Start by uploading the organiser's ZIP file or the six CSV files using the sidebar.")
    st.markdown("### Expected files")
    st.dataframe(pd.DataFrame({"Expected CSV": list(EXPECTED_FILES.values()), "Purpose": ["Delivery-level records", "Match context and results", "Competition names, formats and seasons", "Team identifiers and names", "Match-day player participation", "Competition/team/player name mapping"]}), hide_index=True, use_container_width=True)
    st.markdown("### How this dashboard will be used")
    st.markdown("1. Verify that all six datasets are present.\n2. Inspect columns, missing values, duplicates and relationships.\n3. Select IPL and T20I competitions.\n4. Explore match, batter and bowler records.\n5. Build pressure scores only after the organiser's runs/extras/legal-ball definitions are confirmed.")
    st.stop()

for message in load_messages:
    st.sidebar.success(message)
for error in load_errors:
    st.sidebar.error(error)

# Competition selector defaults to T20I + IPL-named competitions when possible.
comp_df = datasets.get("competitions", pd.DataFrame()).copy()
all_comp_names: list[str] = []
default_comp_names: list[str] = []
if not comp_df.empty and {"comp_id", "comp_name"}.issubset(comp_df.columns):
    comp_df["comp_name"] = comp_df["comp_name"].astype("string")
    all_comp_names = sorted(comp_df["comp_name"].dropna().astype(str).unique().tolist())
    fmt = comp_df.get("match_format", pd.Series("", index=comp_df.index)).astype("string").str.upper()
    nm = comp_df["comp_name"].astype("string").str.upper()
    preferred = comp_df.loc[fmt.eq("T20I") | nm.str.contains(r"\bIPL\b|INDIAN PREMIER LEAGUE", regex=True, na=False), "comp_name"]
    default_comp_names = sorted(preferred.dropna().astype(str).unique().tolist())

with st.sidebar:
    chosen_comp_names = st.multiselect(
        "Competitions",
        options=all_comp_names,
        default=default_comp_names if default_comp_names else all_comp_names,
        help="Default selection aims to include T20I and IPL competitions. Adjust this after checking competition names.",
    ) if all_comp_names else []

matches_scoped, balls_scoped, comps_scoped = selected_scope(datasets, chosen_comp_names)

# Apply selected match IDs to playing XI for scoped analysis.
playing = datasets.get("playing_eleven", pd.DataFrame()).copy()
if not playing.empty and not matches_scoped.empty and "match_id" in playing.columns and "match_id" in matches_scoped.columns:
    playing_scoped = playing[playing["match_id"].astype(str).isin(matches_scoped["match_id"].dropna().astype(str))].copy()
else:
    playing_scoped = playing

# Basic context and field warnings.
ball = balls_scoped.copy()
if not ball.empty:
    for col in ["runs", "innings", "overid", "over_ball_num", "innings_ball_num"]:
        if col in ball.columns:
            ball[f"_{col}_numeric"] = pd.to_numeric(ball[col], errors="coerce")
    if "is_four" in ball.columns:
        ball["_is_four_bool"] = as_bool(ball["is_four"])
    if "is_six" in ball.columns:
        ball["_is_six_bool"] = as_bool(ball["is_six"])
    if "is_wicket" in ball.columns:
        ball["_is_wicket_bool"] = as_bool(ball["is_wicket"])

# ------------------------------- Tabs ----------------------------------------
tabs = st.tabs(["Overview", "Data quality", "Competition & matches", "Batting explorer", "Bowling explorer", "Raw data & exports", "Methodology workspace", "Provisional top 10"])

with tabs[0]:
    st.subheader("Dataset overview")
    audit = file_audit(datasets)
    missing_count = int((audit["Status"] == "Missing").sum())
    st.caption("Counts below use the current competition selection where applicable.")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("CSV datasets loaded", f"{len(datasets)}/6")
    with c2: metric_card("Matches in scope", fmt_num(len(matches_scoped)))
    with c3: metric_card("Delivery records in scope", fmt_num(len(balls_scoped)))
    with c4: metric_card("Missing datasets", str(missing_count))

    st.markdown("#### Uploaded file status")
    st.dataframe(audit, hide_index=True, use_container_width=True)

    if not matches_scoped.empty:
        left, right = st.columns(2)
        with left:
            if "match_format" in matches_scoped.columns:
                fmt_counts = matches_scoped["match_format"].fillna("Unknown").value_counts().rename_axis("Format").reset_index(name="Matches")
                fig = px.bar(fmt_counts, x="Format", y="Matches", title="Matches by format", text="Matches")
                st.plotly_chart(fig, use_container_width=True)
        with right:
            date_col = pd.to_datetime(matches_scoped.get("match_date"), errors="coerce") if "match_date" in matches_scoped.columns else pd.Series(dtype="datetime64[ns]")
            if not date_col.empty:
                years = date_col.dt.year.value_counts().sort_index().rename_axis("Year").reset_index(name="Matches")
                fig = px.bar(years, x="Year", y="Matches", title="Matches by year", text="Matches")
                st.plotly_chart(fig, use_container_width=True)

    if not comps_scoped.empty and "match_format" in comps_scoped.columns:
        st.markdown("#### Competition catalogue")
        summary = comps_scoped.copy()
        if "comp_name" in summary.columns:
            summary = summary[summary["comp_name"].isin(chosen_comp_names)] if chosen_comp_names else summary
        show_cols = [c for c in ["comp_id", "comp_name", "match_format", "season"] if c in summary.columns]
        st.dataframe(summary[show_cols].drop_duplicates().sort_values(show_cols[:2] if len(show_cols) > 1 else show_cols), hide_index=True, use_container_width=True, height=280)

    st.info("A provisional, clearly labelled Top 10 is available in the **Provisional top 10** tab. It uses a limited pressure proxy and is not a final Unstop ranking.")

with tabs[1]:
    st.subheader("Data quality and relationship checks")
    st.markdown("#### Required dataset and column audit")
    st.dataframe(file_audit(datasets), hide_index=True, use_container_width=True)

    st.markdown("#### Duplicate-key checks")
    st.dataframe(make_duplicate_report(datasets), hide_index=True, use_container_width=True)

    quality_rows = []
    for key, df in datasets.items():
        quality_rows.append({"Dataset": EXPECTED_FILES[key], "Rows": len(df), "Columns": len(df.columns), "Missing cells": int(df.isna().sum().sum()), "Rows with any missing value": int(df.isna().any(axis=1).sum())})
    st.markdown("#### Missing-value summary")
    st.dataframe(pd.DataFrame(quality_rows), hide_index=True, use_container_width=True)

    if not ball.empty:
        st.markdown("#### Ball-by-ball field diagnostics")
        diag = []
        for col in ["runs", "extras_type", "extras1_runs", "extras_2_runs", "over_ball_num", "innings_ball_num", "is_wicket", "dismissal_type"]:
            if col in ball.columns:
                diag.append({"Field": col, "Non-null values": int(ball[col].notna().sum()), "Unique values": int(ball[col].nunique(dropna=True)), "Example values": ", ".join(ball[col].dropna().astype(str).unique()[:8])})
        st.dataframe(pd.DataFrame(diag), hide_index=True, use_container_width=True)

        if {"extras_type", "extras1_runs"}.issubset(ball.columns):
            st.markdown("#### Extras field relationship – inspect before using in formulas")
            cross = pd.crosstab(ball["extras_type"].fillna("<missing>"), ball["extras1_runs"].fillna("<missing>").astype(str)).reset_index()
            st.dataframe(cross, hide_index=True, use_container_width=True)
            st.caption("This is a frequency view only. The dashboard does not assume that extras1_runs contains actual extra runs.")

        if "over_ball_num" in ball.columns:
            ob = pd.to_numeric(ball["over_ball_num"], errors="coerce")
            beyond_six = int((ob > 6).sum())
            st.metric("Records where over_ball_num > 6", fmt_num(beyond_six), help="A diagnostic count only; confirm the column's exact definition before using it to calculate legal balls.")

    # Relationship/orphan checks.
    rel_rows = []
    if not datasets.get("ball_by_ball", pd.DataFrame()).empty and not datasets.get("matches", pd.DataFrame()).empty:
        bdf = datasets["ball_by_ball"]
        mdf = datasets["matches"]
        if {"match_id"}.issubset(bdf.columns) and {"match_id"}.issubset(mdf.columns):
            valid = set(mdf["match_id"].dropna().astype(str))
            orphan = int((~bdf["match_id"].dropna().astype(str).isin(valid)).sum())
            rel_rows.append({"Relationship": "ball_by_ball.match_id → matches.match_id", "Unmatched records": orphan})
    if not datasets.get("matches", pd.DataFrame()).empty and not datasets.get("competitions", pd.DataFrame()).empty:
        mdf, cdf = datasets["matches"], datasets["competitions"]
        if "comp_id" in mdf.columns and "comp_id" in cdf.columns:
            valid = set(cdf["comp_id"].dropna().astype(str))
            orphan = int((~mdf["comp_id"].dropna().astype(str).isin(valid)).sum())
            rel_rows.append({"Relationship": "matches.comp_id → competitions.comp_id", "Unmatched records": orphan})
    if rel_rows:
        st.markdown("#### Foreign-key relationship checks")
        st.dataframe(pd.DataFrame(rel_rows), hide_index=True, use_container_width=True)

with tabs[2]:
    st.subheader("Competition and match explorer")
    if matches_scoped.empty:
        st.info("Upload matches.csv and competitions.csv to explore match coverage.")
    else:
        date_series = pd.to_datetime(matches_scoped.get("match_date"), errors="coerce") if "match_date" in matches_scoped.columns else pd.Series(dtype="datetime64[ns]")
        if not date_series.empty and date_series.notna().any():
            dmin, dmax = date_series.min().date(), date_series.max().date()
            date_range = st.date_input("Match date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                parsed = pd.to_datetime(matches_scoped["match_date"], errors="coerce")
                display_matches = matches_scoped[(parsed >= start) & (parsed <= end)].copy()
            else:
                display_matches = matches_scoped.copy()
        else:
            display_matches = matches_scoped.copy()

        show_cols = [c for c in ["match_id", "match_date", "comp_name", "match_format", "season", "stage", "home_team", "away_team", "result"] if c in display_matches.columns]
        st.dataframe(display_matches[show_cols].sort_values("match_date", ascending=False) if "match_date" in show_cols else display_matches[show_cols], hide_index=True, use_container_width=True, height=360)

        if "match_id" in display_matches.columns and not display_matches.empty:
            match_options = display_matches["match_id"].dropna().astype(str).unique().tolist()
            selected_match = st.selectbox("Inspect a match", match_options)
            match_row = display_matches[display_matches["match_id"].astype(str) == selected_match].head(1)
            if not match_row.empty:
                st.markdown("#### Selected match metadata")
                st.dataframe(match_row[show_cols], hide_index=True, use_container_width=True)
            match_deliveries = balls_scoped[balls_scoped["match_id"].astype(str) == str(selected_match)].copy() if "match_id" in balls_scoped.columns else pd.DataFrame()
            if not match_deliveries.empty:
                innings_options = sorted(match_deliveries["innings"].dropna().astype(str).unique().tolist()) if "innings" in match_deliveries.columns else []
                chosen_innings = st.selectbox("Innings", ["All"] + innings_options)
                if chosen_innings != "All":
                    match_deliveries = match_deliveries[match_deliveries["innings"].astype(str) == chosen_innings]
                st.caption(f"{len(match_deliveries):,} recorded delivery rows")
                st.dataframe(match_deliveries, hide_index=True, use_container_width=True, height=320)
                st.download_button("Download selected match deliveries", match_deliveries.to_csv(index=False).encode("utf-8"), file_name=f"match_{selected_match}_deliveries.csv", mime="text/csv")
            else:
                st.info("No ball-by-ball records are available for this match in the uploaded dataset.")

with tabs[3]:
    st.subheader("Batting explorer")
    if ball.empty or "striker_name" not in ball.columns:
        st.info("Upload ball_by_ball.csv to explore batter records.")
    else:
        batter_names = sorted(ball["striker_name"].dropna().astype(str).unique().tolist())
        selected_batter = st.selectbox("Choose a batter", batter_names)
        batter_rows = ball[ball["striker_name"].astype(str) == selected_batter].copy()
        num_runs = pd.to_numeric(batter_rows["runs"], errors="coerce") if "runs" in batter_rows.columns else pd.Series(dtype=float)
        b1, b2, b3, b4 = st.columns(4)
        with b1: metric_card("Recorded delivery rows", fmt_num(len(batter_rows)))
        with b2: metric_card("Sum of runs field", fmt_num(num_runs.sum()), "Raw sum of the supplied runs column; its scoring definition is awaiting confirmation.")
        with b3: metric_card("Four flags", fmt_num(as_bool(batter_rows["is_four"]).fillna(False).sum()) if "is_four" in batter_rows.columns else "—")
        with b4: metric_card("Six flags", fmt_num(as_bool(batter_rows["is_six"]).fillna(False).sum()) if "is_six" in batter_rows.columns else "—")

        if "match_id" in batter_rows.columns:
            by_match = batter_rows.groupby("match_id", dropna=False).agg(recorded_deliveries=("match_id", "size"))
            if "runs" in batter_rows.columns:
                by_match["runs_field_sum"] = pd.to_numeric(batter_rows["runs"], errors="coerce").groupby(batter_rows["match_id"]).sum()
            by_match = by_match.reset_index()
            if "runs_field_sum" in by_match.columns:
                fig = px.bar(by_match, x="match_id", y="runs_field_sum", title="Raw runs-field sum by match", labels={"match_id": "Match ID", "runs_field_sum": "Sum of runs field"})
                st.plotly_chart(fig, use_container_width=True)
                st.caption("This chart is exploratory only; it is not labelled as official batter runs until the field definition is confirmed.")
        display_cols = [c for c in ["match_id", "innings", "overid", "over_ball_num", "runs", "is_four", "is_six", "extras_type", "dismissal_type"] if c in batter_rows.columns]
        st.markdown("#### Selected batter's delivery records")
        st.dataframe(batter_rows[display_cols], hide_index=True, use_container_width=True, height=320)
        st.download_button("Download batter records", batter_rows.to_csv(index=False).encode("utf-8"), file_name="selected_batter_records.csv", mime="text/csv")
        st.warning("Strike rate, balls faced and pressure batting score are intentionally not calculated yet because runs/extras and legal-delivery definitions need confirmation.")

with tabs[4]:
    st.subheader("Bowling explorer")
    if ball.empty or "bowler_name" not in ball.columns:
        st.info("Upload ball_by_ball.csv to explore bowler records.")
    else:
        bowler_names = sorted(ball["bowler_name"].dropna().astype(str).unique().tolist())
        selected_bowler = st.selectbox("Choose a bowler", bowler_names)
        bowler_rows = ball[ball["bowler_name"].astype(str) == selected_bowler].copy()
        wickets_flagged = as_bool(bowler_rows["is_wicket"]).fillna(False).sum() if "is_wicket" in bowler_rows.columns else np.nan
        w1, w2, w3 = st.columns(3)
        with w1: metric_card("Recorded delivery rows", fmt_num(len(bowler_rows)))
        with w2: metric_card("Delivery rows flagged is_wicket", fmt_num(wickets_flagged), "This is a delivery-level flag count, not confirmed bowler-credit wickets.")
        with w3: metric_card("Matches represented", fmt_num(bowler_rows["match_id"].nunique()) if "match_id" in bowler_rows.columns else "—")
        if "match_id" in bowler_rows.columns:
            per_match = bowler_rows.groupby("match_id").size().rename("recorded_delivery_rows").reset_index()
            fig = px.bar(per_match, x="match_id", y="recorded_delivery_rows", title="Recorded bowling delivery rows by match")
            st.plotly_chart(fig, use_container_width=True)
        display_cols = [c for c in ["match_id", "innings", "overid", "over_ball_num", "runs", "extras_type", "is_wicket", "dismissal_type", "bowl_speed", "pace_or_spin"] if c in bowler_rows.columns]
        st.markdown("#### Selected bowler's delivery records")
        st.dataframe(bowler_rows[display_cols], hide_index=True, use_container_width=True, height=320)
        st.download_button("Download bowler records", bowler_rows.to_csv(index=False).encode("utf-8"), file_name="selected_bowler_records.csv", mime="text/csv")
        st.warning("Economy rate, legal overs, bowler-credit wickets and pressure bowling score are intentionally not calculated until scoring definitions are confirmed.")

with tabs[5]:
    st.subheader("Raw data browser and exports")
    dataset_label_to_key = {EXPECTED_FILES[k]: k for k in datasets.keys()}
    selected_dataset_label = st.selectbox("Choose dataset", list(dataset_label_to_key.keys()))
    selected_key = dataset_label_to_key[selected_dataset_label]
    raw_df = datasets[selected_key]
    search = st.text_input("Optional text filter", placeholder="Search visible rows across columns")
    view_df = raw_df.copy()
    if search.strip():
        mask = view_df.astype("string").apply(lambda col: col.str.contains(re.escape(search), case=False, na=False)).any(axis=1)
        view_df = view_df[mask]
    st.caption(f"Showing {min(len(view_df), 2_000):,} of {len(view_df):,} matching rows · {len(raw_df.columns)} columns")
    st.dataframe(view_df.head(2_000), use_container_width=True, height=420)
    st.download_button("Download selected dataset as CSV", raw_df.to_csv(index=False).encode("utf-8"), file_name=EXPECTED_FILES[selected_key], mime="text/csv")
    st.info("The original uploaded CSV data is retained in-memory for this session. Downloaded files contain the loaded source table, not a modified version.")

with tabs[6]:
    st.subheader("Methodology workspace")
    st.markdown("This workspace separates what can be analysed safely now from calculations that need organiser confirmation.")
    st.markdown("#### Proposed analysis sequence")
    st.markdown("1. **Dataset audit:** confirm all six CSVs, schemas, missing values, duplicate keys and joins.\n2. **Score reconstruction:** validate runs, extras, legal deliveries and wickets against official scorecards.\n3. **Match-state reconstruction:** calculate score, wickets in hand, legal balls remaining and chase requirement before each delivery.\n4. **Pressure definition:** define and test pressure indicators consistently for batting and bowling.\n5. **Role-specific metrics:** build separate batting and bowling baselines and pressure scores.\n6. **Fair comparison:** apply exposure thresholds, reliability adjustment and IPL/T20I context normalisation.\n7. **Provisional leaderboard:** inspect the exploratory Top 10 tab and exposure filters.\n8. **Validation:** test match-to-match consistency and sensitivity to pressure thresholds and weights.\n9. **Submission:** regenerate and freeze the final Top 10 lists only after organiser definitions and scorecard checks.")

    st.markdown("#### Calculations currently held back")
    held = pd.DataFrame([
        {"Calculation": "Official team score / required runs", "Reason": "Runs and extras field definitions not confirmed"},
        {"Calculation": "Legal balls remaining", "Reason": "Legal-delivery and ball-number definitions not confirmed"},
        {"Calculation": "Bowler economy and overs", "Reason": "Runs conceded and legal-ball rules not confirmed"},
        {"Calculation": "Bowler-credit wickets", "Reason": "Dismissal and bowler-credit rules not confirmed"},
        {"Calculation": "Pressure scores and final rankings", "Reason": "Depend on validated match state and scoring definitions"},
    ])
    st.dataframe(held, hide_index=True, use_container_width=True)

    st.markdown("#### Export a data audit snapshot")
    audit_parts = [file_audit(datasets).to_csv(index=False)]
    audit_parts.append("\n\nDUPLICATE KEY CHECKS\n")
    audit_parts.append(make_duplicate_report(datasets).to_csv(index=False))
    audit_parts.append("\n\nSCOPE\n")
    audit_parts.append(f"Selected competitions: {', '.join(chosen_comp_names) if chosen_comp_names else 'All competitions'}\n")
    audit_parts.append(f"Scoped matches: {len(matches_scoped)}\nScoped delivery rows: {len(balls_scoped)}\n")
    st.download_button("Download audit snapshot (TXT)", "".join(audit_parts).encode("utf-8"), file_name="suprr_dataset_audit.txt", mime="text/plain")


with tabs[7]:
    st.subheader("🏆 Provisional Top 10 — Batters and Bowlers")
    st.warning(
        "These are exploratory rankings from the currently uploaded data, not final submission rankings. "
        "Pressure is provisionally defined as the final 5 scheduled overs OR a situation with 4 or fewer wickets "
        "remaining after at least 6 legal overs. Required run rate and match closeness are not used because "
        "the supplied run/extras fields have not yet been confirmed."
    )
    st.markdown("#### Eligibility filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        min_pb = st.number_input("Minimum pressure balls", min_value=10, max_value=300, value=60, step=5)
    with f2:
        min_tb = st.number_input("Minimum total balls", min_value=50, max_value=1000, value=200, step=25)
    with f3:
        min_pm = st.number_input("Minimum pressure matches", min_value=2, max_value=30, value=5, step=1)

    batter_top, bowler_top, ranking_summary = calculate_provisional_rankings(
        balls_scoped, matches_scoped, int(min_pb), int(min_tb), int(min_pm)
    )
    if ranking_summary.get("error"):
        st.error(ranking_summary["error"])
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Matches analysed", fmt_num(ranking_summary["matches"]))
        with k2:
            metric_card("Delivery rows", fmt_num(ranking_summary["delivery_rows"]))
        with k3:
            metric_card("Pressure delivery rows", fmt_num(ranking_summary["pressure_delivery_rows"]))
        with k4:
            metric_card("Pressure share", f"{ranking_summary['pressure_share_pct']:.1f}%")

        st.caption(
            f"Eligible pool after filters: {ranking_summary['eligible_batters']:,} batters · "
            f"{ranking_summary['eligible_bowlers']:,} bowlers. The selected competition filter in the sidebar applies."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("### Top 10 batters")
            if batter_top.empty:
                st.info("No batters meet the current exposure filters. Lower the thresholds or broaden the competition selection.")
            else:
                display_bat = batter_top.copy()
                display_bat.insert(0, "Rank", range(1, len(display_bat) + 1))
                cols = [c for c in ["Rank", "player", "score", "pressure_sr", "avg_uplift", "pressure_balls", "pressure_matches", "formats"] if c in display_bat.columns]
                st.dataframe(display_bat[cols].rename(columns={
                    "player": "Player", "score": "Score / 100", "pressure_sr": "Pressure SR",
                    "avg_uplift": "SR uplift", "pressure_balls": "Pressure balls",
                    "pressure_matches": "Pressure matches", "formats": "Format(s)"
                }).round(2), hide_index=True, use_container_width=True)
                st.download_button(
                    "Download batter Top 10 CSV", display_bat.to_csv(index=False).encode("utf-8"),
                    file_name="provisional_top10_batters.csv", mime="text/csv", key="dl_prov_bat"
                )
        with right:
            st.markdown("### Top 10 bowlers")
            if bowler_top.empty:
                st.info("No bowlers meet the current exposure filters. Lower the thresholds or broaden the competition selection.")
            else:
                display_bowl = bowler_top.copy()
                display_bowl.insert(0, "Rank", range(1, len(display_bowl) + 1))
                cols = [c for c in ["Rank", "player", "score", "pressure_econ_proxy", "wickets", "pressure_balls", "pressure_matches", "formats"] if c in display_bowl.columns]
                st.dataframe(display_bowl[cols].rename(columns={
                    "player": "Player", "score": "Score / 100", "pressure_econ_proxy": "Pressure economy proxy",
                    "wickets": "Credited wickets", "pressure_balls": "Pressure legal balls",
                    "pressure_matches": "Pressure matches", "formats": "Format(s)"
                }).round(2), hide_index=True, use_container_width=True)
                st.download_button(
                    "Download bowler Top 10 CSV", display_bowl.to_csv(index=False).encode("utf-8"),
                    file_name="provisional_top10_bowlers.csv", mime="text/csv", key="dl_prov_bowl"
                )

        st.markdown("#### Scoring logic used for this provisional run")
        st.markdown(
            "**Batters:** 35% pressure strike-rate percentile + 20% boundary-rate percentile + "
            "15% inverse dot-ball-rate percentile + 15% inverse dismissal-rate percentile + "
            "15% pressure-versus-normal strike-rate uplift percentile.\n\n"
            "**Bowlers:** 40% inverse pressure-economy-proxy percentile + 30% credited-wicket-rate percentile + "
            "15% inverse boundary-conceded-rate percentile + 15% economy-improvement percentile. "
            "Percentiles are calculated within format (IPL T20 versus T20I); player format scores are combined "
            "using square-root pressure-ball exposure weights."
        )
        st.error(
            "Important limitations: `runs` and extras fields are not confirmed by an organiser data dictionary; "
            "the bowling economy is therefore only a proxy (runs field + one run for a wide/no-ball), and "
            "required run rate / closeness are not included. Legal-ball identification also relies on the observed "
            "wide and Overstep labels. Do not copy these lists into Unstop as final rankings yet."
        )


st.divider()
st.caption("SupeRR Selector Season 2 · Round 1 · Local analysis workspace | The provisional Top 10 tab is exploratory; final rankings require organiser scoring definitions and scorecard validation.")
