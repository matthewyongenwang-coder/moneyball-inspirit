"""
Moneyball data pipeline for the Undervalued Player Finder.

Builds one tidy table of player-seasons (1985-2016, the years salary data exists),
with hitting metrics (OBP, SLG, OPS), primary position, salary, and a per-season
"value" model that flags players who are underpaid relative to their production.

The CSVs come from the public Lahman baseball databank. They are bundled in ./data
so the live demo needs no network. If a file is missing locally, we fall back to
downloading it from GitHub.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GITHUB_BASE = "https://raw.githubusercontent.com/xorq-labs/baseballdatabank/master/"
REMOTE = {
    "Batting": "core/Batting.csv",
    "Salaries": "contrib/Salaries.csv",
    "People": "core/People.csv",
    "Appearances": "core/Appearances.csv",
}

# Games-by-position columns in Appearances -> readable position labels.
POSITION_COLUMNS = {
    "G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B", "G_ss": "SS",
    "G_lf": "LF", "G_cf": "CF", "G_rf": "RF", "G_dh": "DH", "G_p": "P",
}
# The nine lineup slots a team fills with hitters (pitchers excluded).
LINEUP_SLOTS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]


def _read(name):
    """Read a Lahman table from the local data dir, or GitHub as a fallback."""
    local = os.path.join(DATA_DIR, f"{name}.csv")
    if os.path.exists(local):
        return pd.read_csv(local)
    return pd.read_csv(GITHUB_BASE + REMOTE[name])


def build_player_seasons(min_ab=250):
    """Return one row per qualified hitter-season with metrics, salary and position."""
    batting = _read("Batting")
    salaries = _read("Salaries")
    people = _read("People")
    appearances = _read("Appearances")

    # A player traded mid-season has several Batting rows (stints); sum them.
    counting = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI",
                "BB", "SO", "IBB", "HBP", "SH", "SF"]
    b = batting.groupby(["playerID", "yearID"], as_index=False)[counting].sum()

    # Rate stats. Singles = hits that were not doubles, triples or home runs.
    b["1B"] = b["H"] - b["2B"] - b["3B"] - b["HR"]
    obp_denom = (b["AB"] + b["BB"] + b["HBP"] + b["SF"]).replace(0, np.nan)
    b["OBP"] = (b["H"] + b["BB"] + b["HBP"]) / obp_denom
    b["SLG"] = (b["1B"] + 2 * b["2B"] + 3 * b["3B"] + 4 * b["HR"]) / b["AB"].replace(0, np.nan)
    b["OPS"] = b["OBP"] + b["SLG"]

    # One salary per player-season (sum across teams if traded).
    s = salaries.groupby(["playerID", "yearID"], as_index=False)["salary"].sum()

    # Primary position = the spot where the player logged the most games.
    pos_cols = list(POSITION_COLUMNS)
    ap = appearances.groupby(["playerID", "yearID"], as_index=False)[pos_cols].sum()
    ap["POS"] = ap[pos_cols].idxmax(axis=1).map(POSITION_COLUMNS)
    ap = ap[["playerID", "yearID", "POS"]]

    df = (b.merge(s, on=["playerID", "yearID"], how="inner")
            .merge(ap, on=["playerID", "yearID"], how="left")
            .merge(people[["playerID", "nameFirst", "nameLast"]], on="playerID", how="left"))
    df["Name"] = (df["nameFirst"].fillna("") + " " + df["nameLast"].fillna("")).str.strip()

    # Keep qualified hitters with a real salary; drop pitchers.
    df = df[(df["POS"] != "P") & (df["POS"].notna())
            & (df["AB"] >= min_ab) & (df["salary"] > 0)].copy()
    df = df.dropna(subset=["OPS"]).reset_index(drop=True)
    return df


def add_value(df):
    """Add a per-season value model: predicted salary from OPS, and how underpaid.

    For each season we fit log(salary) = m * OPS + c across the league. A player's
    'value' is predicted salary minus actual salary: a large positive value means
    the player produced like someone paid far more, i.e. a bargain.
    """
    out = []
    for year, g in df.groupby("yearID"):
        g = g.copy()
        slope, intercept = np.polyfit(g["OPS"], np.log(g["salary"]), 1)
        g["pred_salary"] = np.exp(slope * g["OPS"] + intercept)
        g["value"] = g["pred_salary"] - g["salary"]
        # Rank 1 = most underpaid hitter that season.
        g["value_rank"] = g["value"].rank(ascending=False, method="min").astype(int)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_roster(season_df, budget, slots=None):
    """Assemble the best full lineup (one hitter per slot) within a salary budget.

    Strategy that both fills the lineup and spends smartly:
      1. Field a legal team first: take the cheapest hitter at every slot. If that
         floor already busts the budget, drop the priciest slots until it fits.
      2. Then repeatedly make the single affordable upgrade that adds the most OPS,
         until no upgrade fits the remaining money. This chases stars without ever
         leaving a position empty when the budget could cover one.
    """
    slots = list(slots or LINEUP_SLOTS)
    by_pos = {p: g.sort_values("OPS", ascending=False)
              for p, g in season_df[season_df["POS"].isin(slots)].groupby("POS")}

    # 1. Cheapest legal starter per slot.
    current = {}
    for p in slots:
        if p in by_pos:
            current[p] = by_pos[p].loc[by_pos[p]["salary"].idxmin()]
    # If the cheap floor busts the budget, drop the most expensive slots.
    while current and sum(r["salary"] for r in current.values()) > budget:
        drop = max(current, key=lambda p: current[p]["salary"])
        del current[drop]

    # 2. Upgrade by the biggest affordable OPS gain, repeatedly.
    for _ in range(500):  # generous bound; each step strictly raises total OPS
        spent = sum(r["salary"] for r in current.values())
        best = None  # (gain, pos, candidate_row)
        for p, cur in current.items():
            for _, cand in by_pos[p].iterrows():
                extra = cand["salary"] - cur["salary"]
                gain = cand["OPS"] - cur["OPS"]
                if gain > 0 and extra > 0 and spent + extra <= budget:
                    if best is None or gain > best[0]:
                        best = (gain, p, cand)
        if best is None:
            break
        current[best[1]] = best[2]

    roster = (pd.DataFrame(current.values())
              .sort_values("OPS", ascending=False) if current else pd.DataFrame())
    spent = sum(r["salary"] for r in current.values())
    missing = [s for s in slots if s not in current]
    return roster, spent, missing


# Cheap self-test when run directly.
if __name__ == "__main__":
    d = add_value(build_player_seasons())
    print(f"player-seasons: {len(d):,}  seasons: {d.yearID.min()}-{d.yearID.max()}")
    top = d[d.yearID == 2001].nsmallest(5, "value_rank")
    print(top[["Name", "POS", "OPS", "salary", "pred_salary", "value"]].to_string(index=False))
    r, spent, missing = build_roster(d[d.yearID == 2001], budget=40_000_000)
    print(f"\n$40M roster: {len(r)} players, spent ${spent:,.0f}, missing {missing}")
    print(r[["Name", "POS", "OPS", "salary"]].to_string(index=False))
