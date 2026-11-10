# SupeRR Selector Season 2 – Round 1 Dashboard

A local Streamlit dashboard for uploading the organiser's six CSV datasets (or one ZIP containing them), auditing data quality, filtering competitions, exploring matches and viewing **provisional** batter and bowler Top 10 lists.

## What is new

The **Provisional top 10** tab calculates exploratory leaderboards from the currently selected competition set and the uploaded ball-by-ball data. It also lets you change the exposure thresholds and download each Top 10 list as CSV.

Pre-calculated example outputs from the ZIP supplied in this conversation are in `outputs/`:

- `provisional_top10_batters.csv`
- `provisional_top10_bowlers.csv`
- `analysis_notes.txt`

These example CSVs use the IPL and T20I competitions in the supplied archive. When you upload another dataset or change the competition selection, the dashboard recalculates the provisional lists.

## Important: rankings are provisional, not final Unstop selections

The supplied archive does not include a data dictionary. In particular, the exact meanings of `runs`, `extras_type`, `extras1_runs`, `extras_2_runs`, ball numbering and wicket fields have not been confirmed by the organisers.

The current exploratory pressure proxy is:

- A delivery in the final 5 scheduled overs; **or**
- A delivery after at least 6 legal overs when the batting team has 4 or fewer wickets remaining.

This intentionally avoids using required run rate while the scoreboard/extras fields remain unverified. It does not yet measure how close the match is or how difficult a chase is.

Eligibility defaults:

- At least 60 pressure balls
- At least 200 total balls
- At least 5 matches containing pressure deliveries

Batting score (percentiles calculated within IPL T20 vs T20I):

- 35% pressure strike-rate percentile
- 20% boundary-rate percentile
- 15% inverse dot-ball-rate percentile
- 15% inverse dismissal-rate percentile
- 15% pressure-vs-normal strike-rate uplift percentile

Bowling score:

- 40% inverse pressure-economy-proxy percentile
- 30% bowler-credit wicket-rate percentile
- 15% inverse boundary-conceded-rate percentile
- 15% normal-vs-pressure economy improvement percentile

Format scores for a player appearing in both IPL and T20I are combined using square-root pressure-ball exposure weights.

The current bowling economy proxy uses the `runs` field plus one run for each wide or `Overstep` delivery, and excludes byes/legbyes from bowler-charged runs. This is an assumption for exploration only, not an organiser-confirmed scoring interpretation. The dashboard deliberately flags this limitation. **Do not submit these provisional lists on Unstop as final rankings.** First obtain the organiser's data definitions and validate innings totals, legal balls and wickets against scorecards, then revise/freeze the model.

## Requirements

- Python 3.10 or newer recommended
- Internet connection for first-time package installation
- Windows, macOS or Linux

## Windows setup

1. Extract this package to a folder.
2. Open PowerShell in that folder.
3. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the dashboard:

   ```powershell
   streamlit run suprr_dashboard.py
   ```

6. A browser tab will open. Upload `ZIP.zip` in the left sidebar, or upload the six CSV files individually.

You can also use `run_dashboard.bat` on Windows after extracting the package. It installs requirements if needed and launches Streamlit.

## macOS / Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run suprr_dashboard.py
```

## Supported dataset filenames

- `ball_by_ball.csv`
- `matches.csv`
- `competitions.csv`
- `teams.csv`
- `playing_eleven.csv`
- `competition_team_player_mapping.csv`

The dashboard recognises common filename variations containing these dataset names. Individual CSV uploads override the corresponding file found inside the ZIP.

## Dashboard sections

- Overview: loaded datasets, scope counts and competition coverage.
- Data quality: required columns, missing values, duplicate-key checks, extras-field diagnostics and selected joins.
- Competition & matches: competition filtering, date filtering and match-level delivery inspection.
- Batting explorer: raw batter delivery records and source-field summaries.
- Bowling explorer: raw bowler delivery records and wicket-field diagnostics.
- Raw data & exports: browse and download the loaded source tables.
- Methodology workspace: analysis sequence and held-back calculations.
- Provisional top 10: calculated exploratory batter and bowler rankings, exposure filters and CSV downloads.

## Privacy

The dashboard runs locally on your computer. CSV contents are read into the local Streamlit process and are not uploaded by this app to a third-party data service. The app itself does not contain external data API calls.
