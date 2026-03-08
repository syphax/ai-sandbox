"""
Velogames Fantasy Cycling Optimizer
=====================================
Generic ILP optimizer for any Velogames stage race contest.

Usage
-----
1. Set RACE_CSV to the riders CSV for the race you want to optimise.
2. Optionally adjust BUDGET, N_RIDERS, and N_ALTERNATIVES below.
3. Run:  python velogames_optimizer.py

The CSV must have these columns:
    name            – rider name
    team            – team name
    cost            – velogames velocredit cost
    expected_score  – your estimated fantasy points (edit this column to tune)
    notes           – optional free-text rationale (may be empty)

Dependencies
------------
    pip install pulp pandas
"""

import sys
from pathlib import Path

import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# CONFIG – change these to switch races
# ---------------------------------------------------------------------------
RACE_CSV     = "../data/2026-tirreno-adriatico-riders.csv"
#RACE_CSV     = "../data/2026-paris-nice-riders.csv"
BUDGET       = 100          # velocredit cap
N_RIDERS     = 9            # riders per team
N_ALTERNATIVES = 5          # how many near-optimal swaps to show
# ---------------------------------------------------------------------------


def load_riders(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"ERROR: CSV not found: {path.resolve()}")
    df = pd.read_csv(path)
    required = {"name", "team", "cost", "expected_score"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}")
    df["notes"] = df.get("notes", "").fillna("")
    df = df.dropna(subset=["name", "team", "cost", "expected_score"])
    df["cost"]           = df["cost"].astype(int)
    df["expected_score"] = df["expected_score"].astype(float)
    return df.reset_index(drop=True)


def solve(df: pd.DataFrame,
          budget: int = 100,
          n_riders: int = 9,
          exclude_names: list[str] | None = None) -> tuple[pd.DataFrame, float]:
    """
    Solve the ILP and return (selected_df, total_score).
    exclude_names forces those riders out of the solution (used for alternatives).
    """
    prob = pulp.LpProblem("Velogames", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(df))]

    # Objective
    prob += pulp.lpSum(df.loc[i, "expected_score"] * x[i] for i in range(len(df)))

    # Budget
    prob += pulp.lpSum(df.loc[i, "cost"] * x[i] for i in range(len(df))) <= budget, "Budget"

    # Team size
    prob += pulp.lpSum(x) == n_riders, "TeamSize"

    # Force-exclude riders (for alternative teams)
    if exclude_names:
        for name in exclude_names:
            idxs = df.index[df["name"] == name].tolist()
            for idx in idxs:
                prob += x[idx] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        return pd.DataFrame(), 0.0

    selected = df[[pulp.value(x[i]) > 0.5 for i in range(len(df))]].copy()
    selected = selected.sort_values("expected_score", ascending=False).reset_index(drop=True)
    return selected, selected["expected_score"].sum()


def print_team(team: pd.DataFrame, budget_used: int, total_score: float,
               budget: int, label: str = "OPTIMAL TEAM") -> None:
    w = 72
    print("=" * w)
    print(f"  {label}  |  Budget: {budget_used}/{budget}  |  Expected score: {total_score:.0f}")
    print("=" * w)
    print(f"{'#':<3} {'Rider':<32} {'Team':<28} {'Cost':>4}  {'Score':>6}")
    print("-" * w)
    for i, row in team.iterrows():
        print(f"{i+1:<3} {row['name']:<32} {row['team']:<28} "
              f"{int(row['cost']):>4}  {row['expected_score']:>6.0f}")
    print("-" * w)
    print(f"{'TOTAL':<66} {budget_used:>4}  {total_score:>6.0f}")


def print_notes(team: pd.DataFrame) -> None:
    notes = [(row["name"], row["notes"]) for _, row in team.iterrows() if row["notes"]]
    if notes:
        print("\nSelection rationale:")
        for name, note in notes:
            print(f"  {name}: {note}")


def print_value_table(df: pd.DataFrame, selected: pd.DataFrame, top_n: int = 10) -> None:
    unselected = df[~df["name"].isin(selected["name"])].copy()
    unselected["value"] = unselected["expected_score"] / unselected["cost"]
    unselected = unselected.sort_values("value", ascending=False).head(top_n)
    print(f"\n{'─'*72}")
    print(f"Top-{top_n} unselected riders by score/credit:")
    print(f"  {'Rider':<32} {'Cost':>4}  {'Score':>6}  {'Val':>5}")
    for _, row in unselected.iterrows():
        print(f"  {row['name']:<32} {int(row['cost']):>4}  "
              f"{row['expected_score']:>6.0f}  {row['value']:>5.1f}")


def print_alternatives(df: pd.DataFrame, base_team: pd.DataFrame,
                       base_score: float, budget: int,
                       n_riders: int, n_alternatives: int) -> None:
    print(f"\n{'─'*72}")
    print("Near-optimal alternative teams (swap one rider from the base team):")
    results = []
    for _, row in base_team.iterrows():
        alt, score = solve(df, budget, n_riders, exclude_names=[row["name"]])
        if not alt.empty:
            cost = int(alt["cost"].sum())
            results.append((score, cost, alt, row["name"]))
    results.sort(key=lambda t: t[0], reverse=True)
    seen: set[frozenset] = set()
    shown = 0
    for score, cost, team, excluded in results:
        key = frozenset(team["name"])
        if key in seen:
            continue
        seen.add(key)
        delta = score - base_score
        incoming = set(team["name"]) - set(base_team["name"])
        incoming_str = ", ".join(sorted(incoming)) if incoming else "(same riders)"
        print(f"  Exclude {excluded:<30} → score {score:.0f} ({delta:+.0f})  "
              f"adds: {incoming_str}")
        shown += 1
        if shown >= n_alternatives:
            break


def main() -> None:
    df = load_riders(RACE_CSV)
    race_label = Path(RACE_CSV).stem.replace("-riders", "").replace("-", " ").title()
    print(f"\nRace: {race_label}")
    print(f"Riders loaded: {len(df)}  |  Budget: {BUDGET}  |  Team size: {N_RIDERS}\n")

    best_team, total_score = solve(df, BUDGET, N_RIDERS)
    if best_team.empty:
        sys.exit("Solver found no feasible solution – check CSV and budget.")

    budget_used = int(best_team["cost"].sum())
    print_team(best_team, budget_used, total_score, BUDGET)
    print_notes(best_team)
    print_value_table(df, best_team)
    print_alternatives(df, best_team, total_score, BUDGET, N_RIDERS, N_ALTERNATIVES)
    print()


if __name__ == "__main__":
    main()
