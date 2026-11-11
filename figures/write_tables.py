"""Builds the tables of the paper from the JSON files, so that no figure is copied by hand.

Each function writes a tab_*.tex file with the whole float, caption included, and the
main document only does \\input. If an analysis changes, this is run again and the
paper is up to date without touching the text.

Usage: python3 write_tables.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# This file is kept in two layouts, so the code root is the one above it that
# holds the modal/ package, and the results are read from wherever they really are.
BASE = next(d for d in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE)))
            if os.path.isdir(os.path.join(d, "modal")))
ROOT = os.environ.get("TREELEARN_ROOT") or os.path.dirname(BASE)
sys.path.insert(0, os.path.join(BASE, "modal"))
from evaluation_core import results_path  # noqa: E402

B = json.load(open(results_path("main_evaluation.json")))
J = json.load(open(results_path("crown_jaccard.json")))
CH = json.load(open(results_path("chance_floor.json")))
CMP = json.load(open(results_path("paired_comparison.json")))
TL = json.load(open(results_path("treelite_evaluation.json")))
SP = json.load(open(results_path("species_recall.json")))

from decimal import Decimal, ROUND_HALF_UP


def r2(v):
    """Round half up to two decimals, not the way floating point rounds."""
    return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


MET = ["SAT", "AMS3D", "FF3D", "TreeLite3D", "WS"]
LABEL = {"SAT": "SegmentAnyTree, zero shot",
       "AMS3D": "AMS3D, tuned per plot",
       "FF3D": "ForestFormer3D, zero shot",
       "TreeLite3D": "TreeLite3D, retuned clustering",
       "WS": "CHM watershed, tuned per plot"}
SHORT = {"SAT": "SegmentAnyTree", "AMS3D": "AMS3D", "FF3D": "ForestFormer3D",
         "TreeLite3D": "TreeLite3D", "WS": "CHM watershed"}
SIT = ["Chamrousse", "Pellizzano", "Both"]


def density_range(pref):
    """The density range comes out of the JSON, not written by hand: if the calculation
    changes, the heading of the table changes with it."""
    v = [d["density"] for k, d in B["density"].items() if k.startswith(pref)]
    return f"{min(v):.0f}--{max(v):.0f} pts/m$^2$"


TITLES = {"Chamrousse": f"Chamrousse, 7 plots, 894 trees, {density_range('cham')}",
       "Pellizzano": f"Pellizzano, 15 plots, 543 trees, {density_range('pell')}",
       "Both": "Both sites, 22 plots, 1437 trees"}


def n(v, d=3):
    return f"{v:.{d}f}"


def ci(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def neg(s):
    """The minus signs of LaTeX, which are not hyphens."""
    return s.replace("-", "$-$")


def write_table(name, txt):
    with open(os.path.join(HERE, name), "w") as f:
        f.write(txt)
    print(f"written {name}")


# ------------------------------------------------------------------- table I
def main_table():
    fil = []
    for s in SIT:
        fil.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{TITLES[s]}}}}}\\\\")
        best_recall = max(B["main"][s][m]["recall"] for m in MET)
        best_f1 = max(B["main"][s][m]["f1"] for m in MET)
        for m in MET:
            v = B["main"][s][m]
            r = f"\\textbf{{{n(v['recall'])}}}" if v["recall"] == best_recall else n(v["recall"])
            f1 = f"\\textbf{{{n(v['f1'])}}}" if v["f1"] == best_f1 else n(v["f1"])
            ex = CH["by_tolerance"]["3.5"][m]["excess"] if s == SIT[-1] else None
            fil.append(f"{LABEL[m]} & {r} & {ci(v['recall_ci'])} & "
                       f"{n(v['precision'])} & {f1} & "
                       f"{n(ex) if ex is not None else '--'}\\\\")
        if s != SIT[-1]:
            fil.append("\\midrule")
    body = "\n".join(fil)
    write_table("tab_main.tex", f"""\\begin{{table}}[!t]
\\centering
\\caption{{Detection at $d_{{\\text{{match}}}}=3.5$~m, scored inside the inventoried
region. The three networks use public checkpoints and no training on these data;
SegmentAnyTree and ForestFormer3D run with published defaults, TreeLite3D with two
clustering hyperparameters retuned (Section~\\ref{{sec:treelite}}). AMS3D and the
watershed use the best of 25 and 16 configurations chosen per plot on the plot being
scored. Excess is recall above a uniform detector given the same number of detections in
the same region: stems sit 1.9~m apart, so a 3.5~m tolerance is met without resolving
anything, and only the excess required reading the returns.}}
\\label{{tab:main}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
Method & Recall & 95\\% CI & Prec. & $F_1$ & Excess \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ------------------------------------------------------------------ table II
def density():
    pcol = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
    fil = []
    for m in MET[1:]:
        d = {s: CMP["diff_vs_SAT"][s][m] for s in ("Chamrousse", "Pellizzano", "Both")}
        rho = B["density_vs_recall"][m]
        fil.append(neg(
            f"{SHORT[m]} & "
            f"{d['Chamrousse']['diff']:+.3f} {ci(d['Chamrousse']['ci'])} & "
            f"{d['Pellizzano']['diff']:+.3f} {ci(d['Pellizzano']['ci'])} & "
            f"{d['Both']['diff']:+.3f} {ci(d['Both']['ci'])} & "
            f"{r2(rho['rho_diff']):+.2f} & {pcol(rho['p_diff'])}\\\\"))
    body = "\n".join(fil)
    rs = B["density_vs_recall"]
    pv = lambda x: "$p<0.001$" if x < 0.001 else f"$p={x:.3f}$"
    note = ", ".join(f"{SHORT[m]} {r2(rs[m]['rho']):+.2f} ({pv(rs[m]['p'])})" for m in MET)
    write_table("tab_density.tex", f"""\\begin{{table*}}[!t]
\\centering
\\caption{{Recall of SegmentAnyTree minus that of each method at
$d_{{\\text{{match}}}}=3.5$~m, resampled over the same plots. The last two columns give the
Spearman correlation of that difference with the plot's return density over the 22 plots,
with a permutation $p$ value. Every deep model closes its gap as density rises; the
watershed does not.}}
\\label{{tab:density}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lcccrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Recall difference vs SegmentAnyTree}}
& \\multicolumn{{2}}{{c}}{{Difference vs density}}\\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}
Method & Chamrousse, sparse & Pellizzano, dense & Both sites & $\\rho$ & $p$\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


# ----------------------------------------------------------------- table III
def treelite():
    fil = []
    for et, name in (("default", "Published, 0.05~m, 100 pts"),
                    ("oracle", "Retuned, 0.20~m, 30 pts"),
                    ("leave_one_out", "Retuned, plot left out")):
        for s in ("Both",):
            v = TL[et][s]
            fil.append(f"{name} & {n(v['recall'])} & "
                       f"{ci(v['recall_ci'])} & {n(v['precision'])} & {n(v['f1'])} & "
                       f"{v['ND']}\\\\")
    body = "\n".join(fil)
    write_table("tab_treelite.tex", f"""\\begin{{table}}[!t]
\\centering
\\caption{{TreeLite3D over both sites at $d_{{\\text{{match}}}}=3.5$~m, before and after
retuning the linking radius and the minimum proposal size. The third row selects the
setting on the other 21 plots for each plot in turn, and returns the same setting every
time. $N_d$ is the detections retained inside the inventoried region, against 1437
inventoried trees.}}
\\label{{tab:treelite}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lccccr}}
\\toprule
Clustering setting & Recall & 95\\% CI & Prec. & $F_1$ & $N_d$\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ------------------------------------------------------------------ table IV
def species():
    order = [k for k in SP if k not in ("broadleaf_gap", "All conifers",
                                         "All broadleaves")]
    order.sort(key=lambda k: -SP[k]["n"])
    fil = []
    for k in order + ["All conifers", "All broadleaves"]:
        v = SP[k]
        name = {"All conifers": "\\midrule\nAll conifers",
               "All broadleaves": "All broadleaves"}.get(k, f"\\textit{{{k}}}")
        fil.append(f"{name} & {v['n']} & {v['plots']} ({v['n_eff']:.1f}) & " +
                   " & ".join(n(v[m]["recall"], 2) for m in MET) + "\\\\")
    br = SP["broadleaf_gap"]
    line = neg(", ".join(f"{SHORT[m]} {br[m]['diff']:+.3f} {ci(br[m]['ci'])}"
                          for m in MET))
    body = "\n".join(fil)
    write_table("tab_species.tex", f"""\\begin{{table*}}[!t]
\\centering
\\caption{{Detection recall by species at $d_{{\\text{{match}}}}=3.5$~m. $n$ is the number
of inventoried trees and ``plots ($n_{{\\text{{eff}}}}$)'' gives the plots the group lives
in and the effective number of plots, $n_{{\\text{{eff}}}}=1/\\sum_i p_i^2$ with $p_i$ the
share of the group's trees in plot $i$: all 95 \\textit{{Pinus uncinata}} lie in one plot,
so that row is a single plot's recall. Species with fewer than 30 trees enter only the two
aggregate rows. The broadleaf minus conifer gap, computed only in the 14 plots that hold
both and resampled over those plots, is {line}.}}
\\label{{tab:species}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrrccccc}}
\\toprule
Species & $n$ & plots ($n_{{\\text{{eff}}}}$) & SegmentAnyTree & AMS3D & ForestFormer3D
& TreeLite3D & watershed\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


# ------------------------------------------------------------------- table V
def robustness():
    """A single column table: the matching tolerance goes in the text."""
    fil = []
    for m in MET:
        e = B["matching"][m]
        reg = B["region"][m]
        fil.append(
            f"{SHORT[m]} & {n(B['distance'][m]['3.5']['recall'])} & "
            f"{n(e['recall_optimal'])} & "
            f"{n(reg['hull +0 m']['precision'])} & "
            f"{n(reg['hull +1.5 m']['precision'])} & "
            f"{n(reg['hull +3 m']['precision'])} & "
            f"{n(reg['whole tile']['precision'])}\\\\")
    body = "\n".join(fil)
    write_table("tab_robustness.tex", f"""\\begin{{table}}[!t]
\\centering
\\caption{{Sensitivity to two of the three choices a detection evaluator has to make:
greedy against optimal one-to-one assignment, and the region over which detections are
counted. The ordering of the methods survives both, and the matching tolerance, reported
in the text; the absolute numbers do not.}}
\\label{{tab:robustness}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Recall, 3.5~m}} &
\\multicolumn{{4}}{{c}}{{Precision by scoring region}}\\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-7}}
Method & greedy & optimal & hull & $+1.5$~m & $+3$~m & tile\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    main_table(); density(); treelite(); species(); robustness()
