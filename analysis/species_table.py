#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base table by species: recall of SAT, AMS3D and watershed at 3.5 m.

What is new next to species_recall.py is the bridge between sites. Chamrousse names the
species with four letter codes and Pellizzano with English names, but several of
them are the same species. Translating both to scientific name lets you look at
the same species in two sites whose flight densities differ by a factor of two.

Resampling unit: the plot. Bootstrap of 2000 resamples with
numpy.random.default_rng(0), seed restarted on each aggregate.
The comparisons between methods are matched: inside each resample the difference
is computed over the same trees, not two separate intervals.
For the aggregates that join the two sites the resampling is stratified by site,
so as not to lose the mix of plots of each one.

Reads   viewer/data/trees.json
Writes  analysis/species_table.json
Does not modify any existing file.
"""

import json
import os
import collections
import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)          # the repository, where the viewer data is built
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
SOURCE = os.path.join(BASE, "viewer", "data", "trees.json")
OUTPUT = os.path.join(HERE, "species_table.json")

METHODS = ["SAT", "AMS3D", "WS"]
DIST = "3.5"
NBOOT = 2000
MIN_SEPARATED = 30      # below this the species is not listed on its own in the table
MIN_CONCLUSIVE = 30   # below this the interval is marked as not conclusive
MIN_COUNT = 10        # below this only the count is given

SITE_NAME = {"cham": "Chamrousse", "pell": "Pellizzano"}

# inventory code -> (scientific name, common name, leaf)
# leaf: conifer, deciduous conifer, broadleaf
SPECIES = {
    # Chamrousse, four letter codes
    "PIAB": ("Picea abies", "Norway spruce", "conifer"),
    "ABAL": ("Abies alba", "silver fir", "conifer"),
    "FASY": ("Fagus sylvatica", "European beech", "broadleaf"),
    "PIUN": ("Pinus uncinata", "mountain pine", "conifer"),
    "BEPE": ("Betula pendula", "silver birch", "broadleaf"),
    "ACPS": ("Acer pseudoplatanus", "sycamore maple", "broadleaf"),
    "SOAR": ("Sorbus aria", "whitebeam", "broadleaf"),
    "SOAU": ("Sorbus aucuparia", "rowan", "broadleaf"),
    "COAV": ("Corylus avellana", "common hazel", "broadleaf"),
    "POTR": ("Populus tremula", "aspen", "broadleaf"),
    "BEsp": ("Betula sp.", "undetermined birch", "broadleaf"),
    "FREX": ("Fraxinus excelsior", "European ash", "broadleaf"),
    "PICE": ("Picea sp.", "undetermined spruce", "conifer"),
    "ND": ("undetermined", "undetermined", "undetermined"),
    # Pellizzano, English names
    "Norway_spruce": ("Picea abies", "Norway spruce", "conifer"),
    "Silver_fir": ("Abies alba", "silver fir", "conifer"),
    "European_beech": ("Fagus sylvatica", "European beech", "broadleaf"),
    "Larch": ("Larix decidua", "European larch", "deciduous conifer"),
    "Silver_birch": ("Betula pendula", "silver birch", "broadleaf"),
    "Sycamore_maple": ("Acer pseudoplatanus", "sycamore maple", "broadleaf"),
    "Rowan": ("Sorbus aucuparia", "rowan", "broadleaf"),
    "Common_hazel": ("Corylus avellana", "common hazel", "broadleaf"),
    "Common_alder": ("Alnus sp.", "alder", "broadleaf"),
    "Aspen": ("Populus tremula", "aspen", "broadleaf"),
    "Willows": ("Salix sp.", "willow", "broadleaf"),
    "European_barberry": ("Berberis vulgaris", "European barberry", "broadleaf"),
}

# global counter of all the comparisons that get looked at
COUNTER = {"recall_intervals": 0, "difference_intervals": 0}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load():
    """Returns a list of rows, one per inventory tree."""
    with open(SOURCE) as f:
        data = json.load(f)

    rows = []
    for key, block in data.items():
        site, _ = key.split("/")
        inv = block["inv"]
        n = len(inv["x"])
        det = {}
        for m in METHODS:
            det[m] = set(int(p[1]) for p in block["pairs"][m][DIST])
        for i in range(n):
            sp_code = inv["species"][i]
            sci_of, common_name, leaf = SPECIES.get(sp_code, (sp_code, sp_code, "undetermined"))
            rows.append({
                "site": site,
                "plot": key,
                "code": sp_code,
                "sci_name": sci_of,
                "common_name": common_name,
                "leaf": leaf,
                "stratum": inv["stratum"][i],
                "npts": inv["npts"][i],
                "h": inv["h"][i],
                "det": {m: (i in det[m]) for m in METHODS},
            })
    return rows


# ---------------------------------------------------------------------------
# bootstrap of plots
# ---------------------------------------------------------------------------
def per_plot(rows, method):
    """Groups: plot -> (hits, total) for one method."""
    acc = collections.defaultdict(lambda: [0, 0])
    for f in rows:
        a = acc[f["plot"]]
        a[0] += 1 if f["det"][method] else 0
        a[1] += 1
    return acc


def strata_of(rows):
    """plots grouped by site, in stable order."""
    d = collections.defaultdict(list)
    for f in rows:
        if f["plot"] not in d[f["site"]]:
            d[f["site"]].append(f["plot"])
    return d


def bootstrap_indices(rows, nboot=NBOOT, seed=0):
    """
    Matrix nboot x k with the resampled plot indices.
    If the group touches two sites the resampling is stratified by site:
    inside each site as many plots are drawn as it had.
    Returns also the list of plots in the order of the columns.
    """
    rng = np.random.default_rng(seed)
    stratum_of = strata_of(rows)
    plots = []
    for s in sorted(stratum_of):
        plots.extend(stratum_of[s])
    pos = {p: j for j, p in enumerate(plots)}
    blocks = []
    for s in sorted(stratum_of):
        idx = np.array([pos[p] for p in stratum_of[s]])
        blocks.append(rng.choice(idx, size=(nboot, len(idx)), replace=True))
    return np.hstack(blocks), plots


def ci_recall(rows, method, nboot=NBOOT):
    """Point recall and percentiles 2.5 and 97.5 of the bootstrap of plots."""
    COUNTER["recall_intervals"] += 1
    acc = per_plot(rows, method)
    reps, plots = bootstrap_indices(rows, nboot)
    TP = np.array([acc[p][0] for p in plots], dtype=float)
    totals = np.array([acc[p][1] for p in plots], dtype=float)
    point_estimate = TP.sum() / totals.sum() if totals.sum() else float("nan")
    a = TP[reps].sum(axis=1)
    t = totals[reps].sum(axis=1)
    ok = t > 0
    v = a[ok] / t[ok]
    return {
        "recall": round(float(point_estimate), 4),
        "ci_low": round(float(np.percentile(v, 2.5)), 4),
        "ci_high": round(float(np.percentile(v, 97.5)), 4),
        "hits": int(TP.sum()),
        "n": int(totals.sum()),
    }


def ci_difference(rows, ma, mb, nboot=NBOOT):
    """
    Matched recall difference ma minus mb over the SAME trees.
    What is resampled is the difference, not two loose intervals.
    """
    COUNTER["difference_intervals"] += 1
    delta = collections.defaultdict(lambda: [0.0, 0.0])
    for f in rows:
        d = delta[f["plot"]]
        d[0] += (1 if f["det"][ma] else 0) - (1 if f["det"][mb] else 0)
        d[1] += 1
    reps, plots = bootstrap_indices(rows, nboot)
    num = np.array([delta[p][0] for p in plots], dtype=float)
    den = np.array([delta[p][1] for p in plots], dtype=float)
    point_estimate = num.sum() / den.sum() if den.sum() else float("nan")
    a = num[reps].sum(axis=1)
    t = den[reps].sum(axis=1)
    ok = t > 0
    v = a[ok] / t[ok]
    lo = float(np.percentile(v, 2.5))
    hi = float(np.percentile(v, 97.5))
    return {
        "diff": round(float(point_estimate), 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "n": int(den.sum()),
    }


# ---------------------------------------------------------------------------
# description of how the trees fall across plots
# ---------------------------------------------------------------------------
def distribution(rows):
    c = collections.Counter(f["plot"] for f in rows)
    n = sum(c.values())
    dom, ndom = c.most_common(1)[0]
    est = collections.Counter(f["stratum"] for f in rows)
    if len(c) == 1:
        warning = ("a single plot: with a bootstrap over plots the interval "
                 "comes out zero wide and says nothing")
    elif len(c) == 2:
        warning = "two plots: the interval is almost blind"
    else:
        warning = ""
    return {
        "n": n,
        "n_plots": len(c),
        "dominant_plot": dom,
        "frac_dominant_plot": round(ndom / n, 3),
        "per_plot": dict(sorted(c.items())),
        "stratum": {k: est.get(k, 0) for k in
                    ["dominant", "intermediate", "suppressed"]},
        "frac_suppressed": round(est.get("suppressed", 0) / n, 3),
        "plot_warning": warning,
    }


def mark(n):
    if n < MIN_COUNT:
        return "count_only"
    if n < MIN_CONCLUSIVE:
        return "inconclusive"
    return "ok"


def group_block(rows, label, extra=None):
    """Full table row: split across plots, recall of the three methods."""
    r = distribution(rows)
    row = {"group": label}
    row.update(r)
    row["mark"] = mark(r["n"])
    if extra:
        row.update(extra)
    if r["n"] >= MIN_COUNT:
        row["recall"] = {m: ci_recall(rows, m) for m in METHODS}
    else:
        row["recall"] = {
            m: {"recall": round(sum(1 for f in rows if f["det"][m]) / r["n"], 4),
                "hits": sum(1 for f in rows if f["det"][m]),
                "n": r["n"], "ci_low": None, "ci_high": None}
            for m in METHODS
        }
    return row


def difference_block(rows, label):
    r = distribution(rows)
    if r["n"] < MIN_COUNT:
        return None
    return {
        "group": label,
        "n": r["n"],
        "n_plots": r["n_plots"],
        "frac_dominant_plot": r["frac_dominant_plot"],
        "plot_warning": r["plot_warning"],
        "mark": mark(r["n"]),
        "SAT_minus_AMS3D": ci_difference(rows, "SAT", "AMS3D"),
        "SAT_minus_WS": ci_difference(rows, "SAT", "WS"),
        "AMS3D_minus_WS": ci_difference(rows, "AMS3D", "WS"),
    }


# ---------------------------------------------------------------------------
def main():
    rows = load()
    output = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": SOURCE,
        "distance_m": float(DIST),
        "nboot": NBOOT,
        "seed": 0,
        "resampling_unit": "plot",
        "bootstrap_note": ("resampling stratified by site when the group "
                           "joins both sites; percentiles 2.5 and 97.5"),
    }

    # -- 1. by species and site ---------------------------------------------
    table1 = []
    others_summary = {}
    for site in ["cham", "pell"]:
        site_rows = [f for f in rows if f["site"] == site]
        counts = collections.Counter(f["code"] for f in site_rows)
        frequent = sorted([c for c, n in counts.items() if n >= MIN_SEPARATED],
                         key=lambda c: -counts[c])
        rare = [c for c in counts if c not in frequent]
        for sp_code in frequent:
            sub = [f for f in site_rows if f["code"] == sp_code]
            table1.append(group_block(
                sub, "%s/%s" % (site, sp_code),
                {"site": SITE_NAME[site], "site_code": site,
                 "code": sp_code, "sci_name": sub[0]["sci_name"],
                 "common_name": sub[0]["common_name"], "leaf": sub[0]["leaf"]}))
        sub = [f for f in site_rows if f["code"] in rare]
        if sub:
            table1.append(group_block(
                sub, "%s/other" % site,
                {"site": SITE_NAME[site], "site_code": site,
                 "code": "other", "sci_name": "several", "common_name": "several",
                 "leaf": "several",
                 "species_inside": sorted(rare),
                 "n_species_inside": len(rare)}))
            others_summary[site] = {
                "n_trees": len(sub),
                "n_species": len(rare),
                "species": {c: counts[c] for c in sorted(rare,
                                                          key=lambda x: -counts[x])},
            }
        # total for the site, to have the reference
        table1.append(group_block(
            site_rows, "%s/ALL" % site,
            {"site": SITE_NAME[site], "site_code": site,
             "code": "ALL", "sci_name": "all", "common_name": "all",
             "leaf": "all"}))
    output["by_species_and_site"] = table1
    output["others_detail"] = others_summary

    # -- 2. species present in both sites -----------------------------------
    presence = collections.defaultdict(set)
    for f in rows:
        presence[f["sci_name"]].add(f["site"])
    shared = sorted([c for c, s in presence.items()
                          if len(s) == 2 and c != "undetermined"],
                         key=lambda c: -sum(1 for f in rows
                                            if f["sci_name"] == c))
    table2 = []
    for sci_of in shared:
        sub = [f for f in rows if f["sci_name"] == sci_of]
        row = group_block(sub, "both/%s" % sci_of,
                            {"sci_name": sci_of, "common_name": sub[0]["common_name"],
                             "leaf": sub[0]["leaf"], "site": "both",
                             "codes": sorted(set(f["code"] for f in sub))})
        row["site_breakdown"] = {}
        for site in ["cham", "pell"]:
            ss = [f for f in sub if f["site"] == site]
            if not ss:
                continue
            row["site_breakdown"][site] = group_block(
                ss, "%s/%s" % (site, sci_of), {"site": SITE_NAME[site]})
        table2.append(row)
    output["shared"] = table2
    output["shared_species_list"] = shared

    # -- 2b. aggregate by leaf type -----------------------------------------
    # The reading by species points to a simpler cut, so it is also measured
    # head on. The larch is kept separate because it is a conifer that drops its
    # leaves and it is the only such case.
    leaf_type_table = []
    for leaf in ["conifer", "deciduous conifer", "broadleaf"]:
        for scope in ["both", "cham", "pell"]:
            sub = [f for f in rows if f["leaf"] == leaf
                   and (scope == "both" or f["site"] == scope)]
            if not sub:
                continue
            row = group_block(sub, "%s/%s" % (scope, leaf),
                                {"leaf": leaf, "scope": scope})
            leaf_type_table.append(row)
    output["by_leaf_type"] = leaf_type_table

    # -- 3. matched differences ---------------------------------------------
    deltas = []
    for row in table1:
        sub = [f for f in rows
               if f["site"] == row["site_code"]
               and (row["code"] == "ALL"
                    or (row["code"] == "other"
                        and f["code"] in row.get("species_inside", []))
                    or f["code"] == row["code"])]
        b = difference_block(sub, row["group"])
        if b:
            b["site"] = row["site"]
            b["code"] = row["code"]
            b["sci_name"] = row["sci_name"]
            deltas.append(b)
    for sci_of in shared:
        sub = [f for f in rows if f["sci_name"] == sci_of]
        b = difference_block(sub, "both/%s" % sci_of)
        if b:
            b["site"] = "both"
            b["code"] = "shared"
            b["sci_name"] = sci_of
            deltas.append(b)
        for site in ["cham", "pell"]:
            ss = [f for f in sub if f["site"] == site]
            b = difference_block(ss, "%s/%s" % (site, sci_of))
            if b:
                b["site"] = SITE_NAME[site]
                b["code"] = "shared"
                b["sci_name"] = sci_of
                deltas.append(b)
    for row in leaf_type_table:
        sub = [f for f in rows if f["leaf"] == row["leaf"]
               and (row["scope"] == "both" or f["site"] == row["scope"])]
        b = difference_block(sub, row["group"])
        if b:
            b["site"] = row["scope"]
            b["code"] = "leaf"
            b["sci_name"] = row["leaf"]
            deltas.append(b)
    # the two sites together, everything
    b = difference_block(rows, "global/ALL")
    b["site"] = "both"
    b["code"] = "ALL"
    b["sci_name"] = "all"
    deltas.append(b)
    output["paired_differences"] = deltas

    # -- 4. checks -----------------------------------------------------------
    comp = {}
    for site, expected in [("cham", 894), ("pell", 543)]:
        sum_n = sum(row["n"] for row in table1
                   if row["site_code"] == site and row["code"] != "ALL")
        comp["sum_%s" % site] = sum_n
        comp["cuadra_%s" % site] = (sum_n == expected)
        comp["expected_%s" % site] = expected
    comp["total"] = sum(1 for _ in rows)
    comp["total_matches"] = (comp["total"] == 1437)
    # no tree left unmapped to a scientific name
    unmapped = sorted(set(f["code"] for f in rows
                          if f["code"] not in SPECIES))
    comp["unmapped_codes"] = unmapped
    comp["n_plots"] = len(set(f["plot"] for f in rows))
    output["checks"] = comp
    output["comparisons_examined"] = dict(COUNTER)
    output["comparisons_examined"]["total"] = sum(COUNTER.values())

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=1, ensure_ascii=False)

    # -- readable printing ---------------------------------------------------
    def line(row, pad=""):
        e = row["recall"]
        def cell_of(m):
            v = e[m]
            if v["ci_low"] is None:
                return "%5.3f      (n/a)   " % v["recall"]
            return "%5.3f [%.3f-%.3f]" % (v["recall"], v["ci_low"], v["ci_high"])
        print("%-34s n=%4d plt=%2d dom=%.2f sup=%.2f  SAT %s  AMS3D %s  WS %s  %s%s" % (
            pad + row["group"], row["n"], row["n_plots"],
            row["frac_dominant_plot"], row["frac_suppressed"],
            cell_of("SAT"), cell_of("AMS3D"), cell_of("WS"), row["mark"],
            "  WARNING" if row["plot_warning"] else ""))

    print("\n=== 1. recall by species and site, 3.5 m ===")
    for row in table1:
        line(row)
    print("\nothers:", json.dumps(others_summary, ensure_ascii=False))

    print("\n=== 2. species in the two sites ===")
    for row in table2:
        line(row)
        for site in ["cham", "pell"]:
            if site in row["site_breakdown"]:
                line(row["site_breakdown"][site], "   ")

    print("\n=== 2b. by leaf type ===")
    for row in leaf_type_table:
        line(row)

    print("\n=== 3. matched differences (same trees) ===")
    for b in deltas:
        def cd(k):
            v = b[k]
            return "%+.3f [%+.3f,%+.3f]%s" % (
                v["diff"], v["ci_low"], v["ci_high"],
                " ~0" if v["crosses_zero"] else "  *")
        print("%-34s n=%4d  SAT-AMS3D %s  SAT-WS %s  AMS3D-WS %s  %s" % (
            b["group"], b["n"], cd("SAT_minus_AMS3D"), cd("SAT_minus_WS"),
            cd("AMS3D_minus_WS"), b["mark"]))

    print("\n=== 4. checks ===")
    print(json.dumps(comp, ensure_ascii=False, indent=1))
    print("comparisons looked at:", output["comparisons_examined"])
    print("\nwritten:", OUTPUT)


if __name__ == "__main__":
    main()
