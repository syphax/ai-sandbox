"""
Paris-Nice 2026 Velogames Fantasy Optimizer
============================================
Selects 9 riders under 100 velocredits to maximise expected fantasy score.

Scoring model
-------------
The Velogames system rewards:
  - Stage result (up to 150 pts per stage)
  - Daily GC position (up to 25 pts per stage x 8 stages)
  - Final GC (up to 300 pts)
  - Mountains / Points classifications (up to 60+10/stage each)
  - Intermediate sprints / KOM summits
  - Breakaway bonus (10 pts per qualifying stage in break)
  - Assists (teammate GC/stage/team top-3, 2-6 pts per stage)
  - TTT (up to 30 pts, stage 3)

Race profile
------------
Stage 1  (08 Mar) – Punchy / puncheur   – Côte Chanteloup 1.1km @8.3%, 11km out
Stage 2  (09 Mar) – Flat sprint         – Montargis
Stage 3  (10 Mar) – TTT 23.5 km        – Cosne → Pouilly
Stage 4  (11 Mar) – Mountain finish     – Uchon, brutal 2km @11%+
Stage 5  (12 Mar) – Longest / hardest   – Colombier-le-Vieux, 3020m gain
Stage 6  (13 Mar) – Hilly finish        – Apt, 4km @5%, 4.5km out
Stage 7  (14 Mar) – Summit finish       – Auron, ski station
Stage 8  (15 Mar) – GC decider in Nice  – 3 Cat-1 climbs, Linguador @8.8%

6 of 8 stages heavily favour climbers/GC riders.

Expected score model
--------------------
Estimated using:
  - PCS ranking / known rider ability at this race profile
  - Paris-Nice history (Onley 2nd 2025, Vingegaard won 2021-2022, etc.)
  - Stage win probability  → stage pts
  - GC finish probability  → GC pts (daily + end-of-tour)
  - TTT team strength      → TTT pts
  - Breakaway probability  → breakaway pts
  - Assist probability     → assist pts (riding alongside GC leaders)
All estimates are conservative mid-case scenarios.
"""

import pulp
from dataclasses import dataclass, field
from typing import List


@dataclass
class Rider:
    name: str
    team: str
    cost: int
    expected_score: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Rider database  (name, team, cost, expected_score)
# Expected scores are model estimates – see methodology above.
# ---------------------------------------------------------------------------
riders: List[Rider] = [
    # ---- Tier 1: GC favourites (likely top-5 overall) -----------------
    Rider("Jonas Vingegaard",       "Visma|Lease a Bike",          22, 680,
          "GC favourite. Won P-N 2021-22. Dominant climber. High stage win prob."),
    Rider("Juan Ayuso",             "UAE Team Emirates-XRG",       24, 590,
          "P-N specialist, 2nd 2023. UAE TTT strong. Multiple stage wins likely."),
    Rider("Oscar Onley",            "INEOS Grenadiers",            18, 420,
          "2nd overall P-N 2025. Excels on punchy/mountain stages."),
    Rider("Brandon McNulty",        "UAE Team Emirates-XRG",       18, 320,
          "UAE TTT ace + GC top-10. Strong TT specialist supports Ayuso."),
    Rider("Lenny Martinez",         "Bahrain-Victorious",          16, 310,
          "Young talent, top-5 GC potential. Good climber."),

    # ---- Tier 2: GC contenders / stage hunters (10-14 credits) --------
    Rider("Kévin Vauquelin",        "INEOS Grenadiers",            14, 270,
          "French climber, INEOS. Attacks on mountain stages. GC top-8."),
    Rider("Daniel Felipe Martínez", "Red Bull-BORA-hansgrohe",     14, 245,
          "Strong climber, GC top-10 target. Breakaway threat."),
    Rider("Aleksandr Vlasov",       "Red Bull-BORA-hansgrohe",     14, 220,
          "Good GC climber. P-N top-10 potential."),
    Rider("Davide Piganzoli",       "Visma|Lease a Bike",          12, 230,
          "Visma team. Earns massive assist pts alongside Vingegaard (GC+stage+team)."),
    Rider("Pavel Sivakov",          "UAE Team Emirates-XRG",       12, 215,
          "UAE team. Earns assist pts with Ayuso. Solid climber, GC support."),
    Rider("Iván Romeo",             "Movistar Team",               12, 175,
          "Climber, top-15 GC potential."),
    Rider("Harold Tejada",          "XDS Astana Team",             12, 150,
          "Climber, top-20 GC, steady scorer."),
    Rider("Biniam Girmay",          "NSN Cycling Team",            12, 210,
          "Sprinter/puncheur. Stage 1 & 2 wins possible. Points jersey contender."),
    Rider("Axel Zingle",            "Visma|Lease a Bike",          12, 155,
          "Sprinter/classics. Stage 1 podium possible. Visma assist pts."),

    # ---- Tier 2b: 10-credit riders ------------------------------------
    Rider("Carlos Rodríguez",       "INEOS Grenadiers",            10, 195,
          "INEOS. GC top-10 capable. Earns assists via Onley/Vauquelin."),
    Rider("David Gaudu",            "Groupama-FDJ United",         10, 165,
          "Climber, former P-N top-10. Good mountain stages."),
    Rider("Eddie Dunbar",           "Pinarello-Q36.5",             10, 150,
          "Climber. GC top-20 realistic. Breakaway value."),
    Rider("Dorian Godon",           "INEOS Grenadiers",            10, 130,
          "INEOS, classics/puncheur. Stage 1 threat. Assist pts."),
    Rider("Marc Soler",             "UAE Team Emirates-XRG",       10, 140,
          "UAE. Earns assist pts with Ayuso. Punchy stages."),
    Rider("Valentin Paret-Peintre", "Soudal Quick-Step",           10, 145,
          "Climber, breakaway specialist. Good value on mountain stages."),
    Rider("Pablo Castrillo",        "Movistar Team",               10, 120,
          "Young climber. Breakaway potential."),
    Rider("Mathias Vacek",          "Lidl-Trek",                   10, 110,
          "Sprinter/classics. Stage 1 podium possible."),
    Rider("Nicolas Prodhomme",      "Decathlon CMA CGM Team",      10, 100,
          "Climber, breakaway value."),
    Rider("Samuel Watson",          "INEOS Grenadiers",            10, 100,
          "INEOS domestique/climber. Assist pts."),
    Rider("Max Kanter",             "XDS Astana Team",             10, 95,
          "Sprinter. Stage 2 value."),

    # ---- Tier 3: 8-credit riders --------------------------------------
    Rider("Wilco Kelderman",        "Visma|Lease a Bike",          8,  155,
          "Visma. Earns assist pts alongside Vingegaard every stage. Solid climber."),
    Rider("Joshua Tarling",         "INEOS Grenadiers",            8,  140,
          "World TT champion. TTT stage 3 ace. INEOS assist pts."),
    Rider("Laurence Pithie",        "Red Bull-BORA-hansgrohe",     8,  110,
          "Sprinter/classics. Stage 1-2 potential."),
    Rider("Georg Steinhauser",      "EF Education-EasyPost",       8,  105,
          "Young climber. Breakaway specialist."),
    Rider("Pascal Ackermann",       "Team Jayco AlUla",            8,  95,
          "Sprinter. Stage 2 value."),
    Rider("Casper Van Uden",        "Team Picnic PostNL",          8,  90,
          "Sprinter. Stage 2 value."),
    Rider("Luke Lamperti",          "EF Education-EasyPost",       8,  85,
          "Sprinter/classics. Stage 1-2 value."),
    Rider("Sandy Dujardin",         "TotalEnergies",               8,  80,
          "Classics/punchy rider."),
    Rider("Alexandre Delettre",     "TotalEnergies",               8,  75,
          "Breakaway specialist."),
    Rider("Rick Pluimers",          "Tudor Pro Cycling Team",      8,  70, ""),
    Rider("Mathys Rondel",          "Tudor Pro Cycling Team",      8,  75,
          "Sprinter."),
    Rider("Matteo Trentin",         "Tudor Pro Cycling Team",      8,  80,
          "Experienced puncheur. Stage 1 value."),
    Rider("Torstein Træen",         "Uno-X Mobility",              8,  75,
          "Breakaway specialist."),
    Rider("Stian Fredheim",         "Uno-X Mobility",              8,  65, ""),
    Rider("Igor Arrieta",           "UAE Team Emirates-XRG",       8,  90,
          "UAE – earns assist pts with Ayuso."),
    Rider("Yevgeniy Fedorov",       "XDS Astana Team",             8,  70, ""),
    Rider("Mike Teunissen",         "XDS Astana Team",             8,  70, ""),
    Rider("Alex Baudin",            "EF Education-EasyPost",       8,  80,
          "Breakaway climber."),
    Rider("Søren Kragh Andersen",   "Lidl-Trek",                   8,  85,
          "Classics/puncheur."),
    Rider("Toms Skujiņš",           "Lidl-Trek",                   8,  80,
          "Breakaway specialist."),
    Rider("Mick Van Dijke",         "Red Bull-BORA-hansgrohe",     8,  75, ""),
    Rider("Milan Menten",           "Lotto Intermarché",           8,  80,
          "Sprinter. Stage 2 value."),
    Rider("Raúl García Pierna",     "Movistar Team",               8,  70, ""),
    Rider("Orluis Aular",           "Movistar Team",               8,  75, ""),
    Rider("Lewis Askey",            "NSN Cycling Team",            8,  80,
          "NSN – earns assist pts with Girmay."),
    Rider("Riley Sheehan",          "NSN Cycling Team",            8,  75, ""),
    Rider("Fabio Christen",         "Pinarello-Q36.5",             8,  70, ""),
    Rider("Phil Bauhaus",           "Bahrain-Victorious",          8,  100,
          "Sprinter. Stage 2 favourite. Sprint pts."),
    Rider("Damiano Caruso",         "Bahrain-Victorious",          8,  85,
          "Climber. GC top-15. Bahrain assists with Martinez."),
    Rider("Bryan Coquard",          "Cofidis",                     8,  90,
          "Sprinter. Stage 2 value."),

    # ---- Tier 4: 6-credit riders --------------------------------------
    Rider("Victor Campenaerts",     "Visma|Lease a Bike",          6,  135,
          "Elite TTT rider. Visma teammate → big assist pts over 8 stages."),
    Rider("Edoardo Affini",         "Visma|Lease a Bike",          6,  120,
          "Visma TTT specialist. Assist pts alongside Vingegaard."),
    Rider("Bruno Armirail",         "Visma|Lease a Bike",          6,  100,
          "Visma – earns assist pts."),
    Rider("Lennard Kämna",          "Lidl-Trek",                   6,  110,
          "Breakaway specialist. Wins from breaks on climbing stages."),
    Rider("Anthony Turgis",         "TotalEnergies",               6,  100,
          "Puncheur. Stage 1 & 6 breakaway threat."),
    Rider("Rémi Cavagna",           "Groupama-FDJ United",         6,  90,
          "Strong TTT, breakaway rider."),
    Rider("Quentin Pacher",         "Groupama-FDJ United",         6,  85,
          "Climber breakaway specialist."),
    Rider("Stefan Bissegger",       "Decathlon CMA CGM Team",      6,  90,
          "TTT specialist. Strong stage 3 scorer."),
    Rider("Daan Hoole",             "Decathlon CMA CGM Team",      6,  80, ""),
    Rider("Michał Kwiatkowski",     "INEOS Grenadiers",            6,  90,
          "Classics veteran. Stage 1 podium threat. INEOS assist pts."),
    Rider("Nico Denz",              "Red Bull-BORA-hansgrohe",     6,  80,
          "Breakaway specialist."),
    Rider("Fabio Van Den Bossche",  "Soudal Quick-Step",           6,  80, ""),
    Rider("Jasper Stuyven",         "Soudal Quick-Step",           6,  80,
          "Puncheur. Stage 1 threat."),
    Rider("Casper Pedersen",        "Soudal Quick-Step",           6,  75, ""),
    Rider("Pascal Eenkhoorn",       "Soudal Quick-Step",           6,  75, ""),
    Rider("Ion Izagirre",           "Cofidis",                     6,  75, ""),
    Rider("Benjamin Thomas",        "Cofidis",                     6,  80,
          "Breakaway specialist."),
    Rider("Rudy Molard",            "Groupama-FDJ United",         6,  65, ""),
    Rider("Clément Russo",          "Groupama-FDJ United",         6,  60, ""),
    Rider("Ewen Costiou",           "Groupama-FDJ United",         6,  75,
          "Sprinter."),
    Rider("Callum Scotson",         "Decathlon CMA CGM Team",      6,  65, ""),
    Rider("Cees Bol",               "Decathlon CMA CGM Team",      6,  80,
          "Sprinter. Stage 2 value."),
    Rider("Kasper Asgreen",         "EF Education-EasyPost",       6,  75, ""),
    Rider("Andreas Leknessund",     "Uno-X Mobility",              6,  80,
          "Climber, GC top-20 potential."),
    Rider("Rasmus Tiller",          "Uno-X Mobility",              6,  75,
          "Puncheur/sprinter."),
    Rider("Sven Erik Bystrøm",      "Uno-X Mobility",              6,  65, ""),
    Rider("Jefferson Cepeda",       "Movistar Team",               6,  80,
          "Climber. Mountain breakaway specialist."),
    Rider("Lorenzo Milesi",         "Movistar Team",               6,  75, ""),
    Rider("Matis Louvel",           "NSN Cycling Team",            6,  70, ""),
    Rider("Guillaume Boivin",       "NSN Cycling Team",            6,  60, ""),
    Rider("Aimé De Gendt",          "Pinarello-Q36.5",             6,  75,
          "Breakaway specialist."),
    Rider("David De La Cruz",       "Pinarello-Q36.5",             6,  70, ""),
    Rider("Julien Bernard",         "Lidl-Trek",                   6,  70,
          "Breakaway. Lidl assist pts if Kämna excels."),
    Rider("Jakob Söderqvist",       "Lidl-Trek",                   6,  65, ""),
    Rider("Vito Braet",             "Lotto Intermarché",           6,  70, ""),
    Rider("Jonas Rutsch",           "Lotto Intermarché",           6,  65, ""),
    Rider("Luca Van Boven",         "Lotto Intermarché",           6,  65, ""),
    Rider("Rune Herregodts",        "UAE Team Emirates-XRG",       6,  80,
          "UAE – earns assist pts with Ayuso."),
    Rider("Ivo Oliveira",           "UAE Team Emirates-XRG",       6,  80,
          "UAE TTT, earns assists with Ayuso."),
    Rider("Nils Politt",            "UAE Team Emirates-XRG",       6,  75, ""),
    Rider("Tim Van Dijke",          "Red Bull-BORA-hansgrohe",     6,  65, ""),
    Rider("Callum Thornley",        "Red Bull-BORA-hansgrohe",     6,  65, ""),
    Rider("Steff Cras",             "Soudal Quick-Step",           6,  75, ""),
    Rider("Robert Donaldson",       "Team Jayco AlUla",            6,  65, ""),
    Rider("Luka Mezgec",            "Team Jayco AlUla",            6,  70,
          "Sprinter."),
    Rider("Kelland O'Brien",        "Team Jayco AlUla",            6,  65, ""),
    Rider("Timo De Jong",           "Team Picnic PostNL",          6,  60, ""),
    Rider("Chris Hamilton",         "Team Picnic PostNL",          6,  65, ""),
    Rider("Juan G. Martinez",       "Team Picnic PostNL",          6,  75,
          "Climber."),
    Rider("Florian Dauphin",        "TotalEnergies",               6,  65, ""),
    Rider("Joris Delbove",          "TotalEnergies",               6,  60, ""),
    Rider("Mathis Le Berre",        "TotalEnergies",               6,  75, ""),
    Rider("Mattéo Vercher",         "TotalEnergies",               6,  65, ""),
    Rider("Will Barta",             "Tudor Pro Cycling Team",      6,  65, ""),
    Rider("Marco Haller",           "Tudor Pro Cycling Team",      6,  60, ""),
    Rider("Nicola Conci",           "XDS Astana Team",             6,  65, ""),
    Rider("Darren Van Bekkum",      "XDS Astana Team",             6,  60, ""),
    Rider("Nicolas Vinokurov",      "XDS Astana Team",             6,  65, ""),
    Rider("Jensen Plowright",       "Alpecin-Premier Tech",        6,  65, ""),
    Rider("Ramses Debruyne",        "Alpecin-Premier Tech",        6,  60, ""),
    Rider("Simon Dehairs",          "Alpecin-Premier Tech",        6,  55, ""),
    Rider("Jonas Geens",            "Alpecin-Premier Tech",        6,  55, ""),

    # ---- Tier 5: 4-credit riders --------------------------------------
    Rider("Jasha Sütterlin",        "Team Jayco AlUla",            4,  55, ""),
    Rider("Luke Durbridge",         "Team Jayco AlUla",            4,  60,
          "TTT specialist. Stage 3 value."),
    Rider("Patrick Gamper",         "Team Jayco AlUla",            4,  55, ""),
    Rider("Sébastien Grignard",     "Lotto Intermarché",           4,  55,
          "Punchy climber, breakaway."),
    Rider("Joshua Giddings",        "Lotto Intermarché",           4,  50, ""),
    Rider("Roel Van Sintmaartensdijk","Lotto Intermarché",         4,  45, ""),
    Rider("Alex Kirsch",            "Cofidis",                     4,  45, ""),
    Rider("Robert Stannard",        "Bahrain-Victorious",          4,  50,
          "Bahrain – assist pts with Martinez."),
    Rider("Mathijs Paasschens",     "Bahrain-Victorious",          4,  45, ""),
    Rider("Kamil Gradek",           "Bahrain-Victorious",          4,  40, ""),
    Rider("Nikias Arndt",           "Bahrain-Victorious",          4,  45, ""),
    Rider("Bert Van Lerberghe",     "Soudal Quick-Step",           4,  45, ""),
    Rider("Michel Hessmann",        "Movistar Team",               4,  50, ""),
    Rider("Ryan Mullen",            "NSN Cycling Team",            4,  45, ""),
    Rider("Nickolas Zukowsky",      "Pinarello-Q36.5",             4,  45, ""),
    Rider("Frederik Frison",        "Pinarello-Q36.5",             4,  40, ""),
    Rider("Max Walker",             "EF Education-EasyPost",       4,  45, ""),
    Rider("Alastair MacKellar",     "EF Education-EasyPost",       4,  45, ""),
    Rider("Johan Jacobs",           "Groupama-FDJ United",         4,  50, ""),
    Rider("Oscar Chamberlain",      "Decathlon CMA CGM Team",      4,  45, ""),
    Rider("Sander De Pestel",       "Decathlon CMA CGM Team",      4,  40, ""),
    Rider("Tim Marsman",            "Alpecin-Premier Tech",        4,  40, ""),
    Rider("Lindsay De Vylder",      "Alpecin-Premier Tech",        4,  45, ""),
    Rider("Maurice Ballerstedt",    "Alpecin-Premier Tech",        4,  40, ""),
    Rider("Niklas Märkl",           "Team Picnic PostNL",          4,  40, ""),
    Rider("Tim Naberman",           "Team Picnic PostNL",          4,  45, ""),
    Rider("Timo Roosen",            "Team Picnic PostNL",          4,  40, ""),
    Rider("Petr Kelemen",           "Tudor Pro Cycling Team",      4,  45, ""),
    Rider("Arthur Kluckers",        "Tudor Pro Cycling Team",      4,  40, ""),
    Rider("William Blume Levy",     "Uno-X Mobility",              4,  45, ""),
    Rider("Sakarias Koller Løland", "Uno-X Mobility",              4,  40, ""),
]


# ---------------------------------------------------------------------------
# Integer Linear Programming solver (PuLP)
# ---------------------------------------------------------------------------
def solve(budget: int = 100, n_riders: int = 9, top_n: int = 5) -> None:
    prob = pulp.LpProblem("VelogamesPN2026", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(riders))]

    # Objective
    prob += pulp.lpSum(r.expected_score * x[i] for i, r in enumerate(riders))

    # Budget constraint
    prob += pulp.lpSum(r.cost * x[i] for i, r in enumerate(riders)) <= budget, "Budget"

    # Team size
    prob += pulp.lpSum(x) == n_riders, "TeamSize"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected = [riders[i] for i in range(len(riders)) if pulp.value(x[i]) > 0.5]
    selected.sort(key=lambda r: r.expected_score, reverse=True)

    total_cost  = sum(r.cost           for r in selected)
    total_score = sum(r.expected_score for r in selected)

    print("=" * 70)
    print(f"  OPTIMAL TEAM  |  Budget used: {total_cost}/{budget}  |  "
          f"Expected score: {total_score:.0f}")
    print("=" * 70)
    print(f"{'#':<3} {'Rider':<30} {'Team':<28} {'Cost':>4}  {'Exp.Score':>9}")
    print("-" * 70)
    for idx, r in enumerate(selected, 1):
        print(f"{idx:<3} {r.name:<30} {r.team:<28} {r.cost:>4}  {r.expected_score:>9.0f}")
    print("-" * 70)
    print(f"{'TOTAL':<62} {total_cost:>4}  {total_score:>9.0f}")
    print()
    print("Notes:")
    for r in selected:
        if r.notes:
            print(f"  {r.name}: {r.notes}")
    print()

    # Sensitivity: show best value (score/cost) riders not selected
    print("─" * 70)
    print("Top-10 unselected riders by score/credit (value picks you missed):")
    unselected = [(riders[i], riders[i].expected_score / riders[i].cost)
                  for i in range(len(riders)) if pulp.value(x[i]) < 0.5]
    unselected.sort(key=lambda t: t[1], reverse=True)
    for r, v in unselected[:10]:
        print(f"  {r.name:<30} cost={r.cost:>2}  score={r.expected_score:>6.0f}  "
              f"val={v:>5.1f}")

    # Also show top-n alternative teams by permuting budget tightly
    print()
    print("─" * 70)
    print("Alternative near-optimal teams (small score trade-offs):")
    _show_alternatives(selected, budget, n_riders, top_n)


def _show_alternatives(base_team, budget, n_riders, top_n):
    """
    Brute-force alternatives by excluding each rider in the base team one at a
    time and re-solving.  Shows top_n swaps.
    """
    alternatives = []
    for exclude_idx, exclude_rider in enumerate(base_team):
        prob2 = pulp.LpProblem("alt", pulp.LpMaximize)
        x2 = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(len(riders))]
        # Force-exclude this rider
        exclude_global = next(i for i, r in enumerate(riders) if r.name == exclude_rider.name)
        prob2 += x2[exclude_global] == 0

        prob2 += pulp.lpSum(r.expected_score * x2[i] for i, r in enumerate(riders))
        prob2 += pulp.lpSum(r.cost           * x2[i] for i, r in enumerate(riders)) <= budget
        prob2 += pulp.lpSum(x2) == n_riders

        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        alt = [riders[i] for i in range(len(riders)) if pulp.value(x2[i]) > 0.5]
        alt.sort(key=lambda r: r.expected_score, reverse=True)
        score = sum(r.expected_score for r in alt)
        cost  = sum(r.cost           for r in alt)
        alternatives.append((score, cost, alt, exclude_rider))

    alternatives.sort(key=lambda t: t[0], reverse=True)
    seen = set()
    shown = 0
    for score, cost, team, excluded in alternatives:
        key = frozenset(r.name for r in team)
        if key in seen:
            continue
        seen.add(key)
        names = ", ".join(r.name for r in team)
        base_score = sum(r.expected_score for r in base_team)
        delta = score - base_score
        print(f"  -swap {excluded.name} → score {score:.0f} ({delta:+.0f})")
        shown += 1
        if shown >= top_n:
            break


if __name__ == "__main__":
    solve(budget=100, n_riders=9, top_n=5)