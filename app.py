"""
Moneyball: Undervalued Player Finder
An interactive demo for the Inspirit AI project.

Run locally with:  streamlit run app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from moneyball import build_player_seasons, add_value, build_roster, LINEUP_SLOTS

st.set_page_config(page_title="Moneyball: Undervalued Player Finder",
                   page_icon=None, layout="wide")


@st.cache_data
def load(min_ab):
    return add_value(build_player_seasons(min_ab=min_ab))


def money(x):
    return f"${x:,.0f}"


st.title("Moneyball: Undervalued Player Finder")
st.markdown(
    "It is the offseason and you are a general manager with a small budget. "
    "Every team can see the same stats, but the market overpays for some players "
    "and underpays for others. This tool fits a model of what the league pays for "
    "on-base and slugging production, then flags the hitters producing far more "
    "than their paycheck would suggest."
)

# ---- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    min_ab = st.slider("Minimum at-bats (qualified hitters)", 100, 502, 250, step=25)
    data = load(min_ab)
    seasons = sorted(data["yearID"].unique())
    default_year = 2001 if 2001 in seasons else seasons[-1]
    season = st.select_slider("Season", options=seasons, value=default_year)
    positions = st.multiselect("Positions", LINEUP_SLOTS, default=LINEUP_SLOTS)
    top_n = st.slider("How many bargains to show", 5, 40, 15, step=5)

season_df = data[(data["yearID"] == season) & (data["POS"].isin(positions))].copy()

# ---- Headline numbers -----------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Season", str(season))
c2.metric("Qualified hitters", f"{len(season_df):,}")
c3.metric("League avg OPS", f"{season_df['OPS'].mean():.3f}")
c4.metric("Median salary", money(season_df["salary"].median()))

tab_find, tab_build, tab_data = st.tabs(
    ["Find bargains", "Build a roster", "Full data"])

# ---- Tab 1: scatter + bargain table --------------------------------------
with tab_find:
    st.subheader(f"Production vs. pay, {season}")
    st.caption(
        "Each dot is a hitter. The line is what the league pays for a given OPS. "
        "Dots well above the line are bargains (a lot of production, little money); "
        "dots below are overpaid.")

    plot_df = season_df.assign(
        Bargain=season_df["value"].gt(0).map({True: "Underpaid", False: "Overpaid"}),
        SalaryM=season_df["salary"] / 1e6,
        PredM=season_df["pred_salary"] / 1e6,
    )
    base = alt.Chart(plot_df)
    dots = base.mark_circle(size=70, opacity=0.65).encode(
        x=alt.X("SalaryM:Q", title="Salary ($M)"),
        y=alt.Y("OPS:Q", title="OPS (On-base + Slugging)",
                scale=alt.Scale(zero=False)),
        color=alt.Color("Bargain:N",
                        scale=alt.Scale(domain=["Underpaid", "Overpaid"],
                                        range=["#d6002a", "#8a8f98"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("Name:N"), alt.Tooltip("POS:N", title="Pos"),
                 alt.Tooltip("OPS:Q", format=".3f"),
                 alt.Tooltip("SalaryM:Q", title="Salary $M", format=".2f"),
                 alt.Tooltip("PredM:Q", title="Model value $M", format=".2f")],
    )
    # The market curve: what the league pays for each OPS level (model value vs OPS).
    fit = base.mark_line(color="#111", strokeDash=[5, 4]).encode(
        x=alt.X("PredM:Q", title="Salary ($M)"),
        y=alt.Y("OPS:Q", scale=alt.Scale(zero=False)),
        order=alt.Order("OPS:Q"),
    )
    st.altair_chart(dots + fit, use_container_width=True)

    st.subheader(f"Top {top_n} most underpaid hitters, {season}")
    table = (season_df.nsmallest(top_n, "value_rank")
             [["Name", "POS", "OPS", "HR", "RBI", "salary", "pred_salary", "value"]]
             .rename(columns={"POS": "Pos", "salary": "Salary",
                              "pred_salary": "Model value", "value": "Underpaid by"}))
    st.dataframe(
        table.style.format({"OPS": "{:.3f}", "Salary": money,
                            "Model value": money, "Underpaid by": money}),
        use_container_width=True, hide_index=True)

# ---- Tab 2: roster builder ------------------------------------------------
with tab_build:
    st.subheader(f"Build a starting nine under budget, {season}")
    st.caption("Greedily takes the best available OPS at each position without "
               "blowing the budget: one C, 1B, 2B, 3B, SS, three outfielders, and a DH.")
    budget_m = st.slider("Payroll budget ($M)", 5, 120, 40, step=5)
    roster, spent, missing = build_roster(
        data[data["yearID"] == season], budget=budget_m * 1e6)

    if roster.empty:
        st.warning("No roster could be built at this budget. Try raising it.")
    else:
        r = (roster[["POS", "Name", "OPS", "HR", "RBI", "salary"]]
             .rename(columns={"POS": "Pos", "salary": "Salary"}))
        m1, m2, m3 = st.columns(3)
        m1.metric("Payroll used", money(spent))
        m2.metric("Team OPS (avg)", f"{roster['OPS'].mean():.3f}")
        m3.metric("Slots filled", f"{len(roster)}/{len(LINEUP_SLOTS)}")
        st.dataframe(r.style.format({"OPS": "{:.3f}", "Salary": money}),
                     use_container_width=True, hide_index=True)
        if missing:
            st.info("Could not afford a starter at: " + ", ".join(missing))

# ---- Tab 3: raw table -----------------------------------------------------
with tab_data:
    st.caption("Every qualified hitter-season in the current filters.")
    st.dataframe(
        season_df[["Name", "POS", "yearID", "AB", "OBP", "SLG", "OPS",
                   "HR", "RBI", "salary", "value_rank"]]
        .sort_values("OPS", ascending=False)
        .style.format({"OBP": "{:.3f}", "SLG": "{:.3f}", "OPS": "{:.3f}",
                       "salary": money}),
        use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data: Lahman baseball databank (Batting, Salaries, People, Appearances), "
    "salary years 1985-2016. Value model: per-season fit of log(salary) on OPS; "
    "'underpaid by' is predicted salary minus actual. Correlation, not causation, "
    "and this ignores defense and park effects.")
