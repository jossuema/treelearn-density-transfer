"""Genera los cuadros del articulo desde los JSON, para que ninguna cifra se copie a mano.

Cada funcion escribe un fichero tab_*.tex con el flotante entero, pie incluido, y el
documento principal solo hace \\input. Si cambia un analisis, se vuelve a correr esto y
el articulo queda al dia sin tocar el texto.

Uso: python3 tablas.py
"""
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))

B = json.load(open(os.path.join(R, "robustez_cinco.json")))
J = json.load(open(os.path.join(R, "jaccard_cinco.json")))
CMP = json.load(open(os.path.join(R, "comparacion_modelos.json")))
TL = json.load(open(os.path.join(R, "treelite_evaluacion.json")))
ESP = json.load(open(os.path.join(R, "especie_cinco.json")))

from decimal import Decimal, ROUND_HALF_UP


def r2(v):
    """Redondeo a dos decimales al alza, no el de la coma flotante."""
    return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


MET = ["SAT", "AMS3D", "FF3D", "TreeLite3D", "WS"]
ETI = {"SAT": "SegmentAnyTree, zero shot",
       "AMS3D": "AMS3D, tuned per plot",
       "FF3D": "ForestFormer3D, zero shot",
       "TreeLite3D": "TreeLite3D, retuned clustering",
       "WS": "CHM watershed, tuned per plot"}
CORTO = {"SAT": "SegmentAnyTree", "AMS3D": "AMS3D", "FF3D": "ForestFormer3D",
         "TreeLite3D": "TreeLite3D", "WS": "CHM watershed"}
SIT = ["Chamrousse", "Pellizzano", "Ambos"]


def rango(pref):
    """El rango de densidad sale del JSON, no escrito a mano: si cambia el calculo,
    cambia tambien el encabezado del cuadro."""
    v = [d["densidad"] for k, d in B["densidad"].items() if k.startswith(pref)]
    return f"{min(v):.0f}--{max(v):.0f} pts/m$^2$"


TIT = {"Chamrousse": f"Chamrousse, 7 plots, 894 trees, {rango('cham')}",
       "Pellizzano": f"Pellizzano, 15 plots, 543 trees, {rango('pell')}",
       "Ambos": "Both sites, 22 plots, 1437 trees"}


def n(v, d=3):
    return f"{v:.{d}f}"


def ic(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def neg(s):
    """Los signos menos de LaTeX, que no son guiones."""
    return s.replace("-", "$-$")


def escribe(nombre, txt):
    with open(os.path.join(AQUI, nombre), "w") as f:
        f.write(txt)
    print(f"escrito {nombre}")


# ------------------------------------------------------------------ cuadro I
def principal():
    fil = []
    for s in SIT:
        fil.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{TIT[s]}}}}}\\\\")
        mejor_r = max(B["principal"][s][m]["recall"] for m in MET)
        mejor_f = max(B["principal"][s][m]["f1"] for m in MET)
        for m in MET:
            v = B["principal"][s][m]
            r = f"\\textbf{{{n(v['recall'])}}}" if v["recall"] == mejor_r else n(v["recall"])
            f1 = f"\\textbf{{{n(v['f1'])}}}" if v["f1"] == mejor_f else n(v["f1"])
            fil.append(f"{ETI[m]} & {r} & {ic(v['recall_ic'])} & "
                       f"{n(v['precision'])} & {f1} & {n(J[s][m]['J'])}\\\\")
        if s != SIT[-1]:
            fil.append("\\midrule")
    cuerpo = "\n".join(fil)
    escribe("tab_principal.tex", f"""\\begin{{table}}[!t]
\\centering
\\caption{{Detection at $d_{{\\text{{match}}}}=3.5$~m, scored inside the inventoried
region. The three networks use public checkpoints and no training on these data;
SegmentAnyTree and ForestFormer3D run with published defaults, TreeLite3D with two
clustering hyperparameters retuned (Section~\\ref{{sec:treelite}}). AMS3D and the
watershed use the best of 25 and 16 configurations chosen per plot on the plot being
scored. $J$ is the mean 2-D crown Jaccard over matched pairs, all crowns represented as
equivalent circles.}}
\\label{{tab:principal}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.2pt}}
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
Method & Recall & 95\\% CI & Prec. & $F_1$ & $J$ \\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ----------------------------------------------------------------- cuadro II
def densidad():
    pcol = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
    fil = []
    for m in MET[1:]:
        d = {s: CMP["dif_vs_SAT"][s][m] for s in ("Chamrousse", "Pellizzano", "Ambos")}
        rho = B["densidad_vs_acierto"][m]
        fil.append(neg(
            f"{CORTO[m]} & "
            f"{d['Chamrousse']['dif']:+.3f} {ic(d['Chamrousse']['ic'])} & "
            f"{d['Pellizzano']['dif']:+.3f} {ic(d['Pellizzano']['ic'])} & "
            f"{d['Ambos']['dif']:+.3f} {ic(d['Ambos']['ic'])} & "
            f"{r2(rho['rho_dif']):+.2f} & {pcol(rho['p_dif'])}\\\\"))
    cuerpo = "\n".join(fil)
    rs = B["densidad_vs_acierto"]
    pv = lambda x: "$p<0.001$" if x < 0.001 else f"$p={x:.3f}$"
    nota = ", ".join(f"{CORTO[m]} {r2(rs[m]['rho']):+.2f} ({pv(rs[m]['p'])})" for m in MET)
    escribe("tab_densidad.tex", f"""\\begin{{table*}}[!t]
\\centering
\\caption{{Recall of SegmentAnyTree minus that of each method at
$d_{{\\text{{match}}}}=3.5$~m, resampled over the same plots. The last two columns give the
Spearman correlation of that difference with the plot's return density over the 22 plots,
with a permutation $p$ value. Every deep model closes its gap as density rises; the
watershed does not. Correlation of a method's own plot recall with density:
{nota}.}}
\\label{{tab:densidad}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lcccrr}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Recall difference vs SegmentAnyTree}}
& \\multicolumn{{2}}{{c}}{{Difference vs density}}\\\\
\\cmidrule(lr){{2-4}}\\cmidrule(lr){{5-6}}
Method & Chamrousse, sparse & Pellizzano, dense & Both sites & $\\rho$ & $p$\\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


# ---------------------------------------------------------------- cuadro III
def treelite():
    fil = []
    for et, nom in (("por defecto", "Published, 0.05~m, 100 pts"),
                    ("oracular", "Retuned, 0.20~m, 30 pts"),
                    ("una_fuera", "Retuned, plot left out")):
        for s in ("Ambos",):
            v = TL[et][s]
            fil.append(f"{nom} & {n(v['recall'])} & "
                       f"{ic(v['recall_ic'])} & {n(v['precision'])} & {n(v['f1'])} & "
                       f"{v['ND']}\\\\")
    cuerpo = "\n".join(fil)
    escribe("tab_treelite.tex", f"""\\begin{{table}}[!t]
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
{cuerpo}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


# ----------------------------------------------------------------- cuadro IV
def especies():
    orden = [k for k in ESP if k not in ("brecha_frondosas", "Todas las coniferas",
                                         "Todas las frondosas")]
    orden.sort(key=lambda k: -ESP[k]["n"])
    fil = []
    for k in orden + ["Todas las coniferas", "Todas las frondosas"]:
        v = ESP[k]
        nom = {"Todas las coniferas": "\\midrule\nAll conifers",
               "Todas las frondosas": "All broadleaves"}.get(k, f"\\textit{{{k}}}")
        fil.append(f"{nom} & {v['n']} & {v['parcelas']} ({v['n_eff']:.1f}) & " +
                   " & ".join(n(v[m]["recall"], 2) for m in MET) + "\\\\")
    br = ESP["brecha_frondosas"]
    linea = neg(", ".join(f"{CORTO[m]} {br[m]['dif']:+.3f} {ic(br[m]['ic'])}"
                          for m in MET))
    cuerpo = "\n".join(fil)
    escribe("tab_especie.tex", f"""\\begin{{table*}}[!t]
\\centering
\\caption{{Detection recall by species at $d_{{\\text{{match}}}}=3.5$~m. $n$ is the number
of inventoried trees and ``plots ($n_{{\\text{{eff}}}}$)'' gives the plots the group lives
in and the effective number of plots, $n_{{\\text{{eff}}}}=1/\\sum_i p_i^2$ with $p_i$ the
share of the group's trees in plot $i$: all 95 \\textit{{Pinus uncinata}} lie in one plot,
so that row is a single plot's recall. Species with fewer than 30 trees enter only the two
aggregate rows. The broadleaf minus conifer gap, computed only in the 14 plots that hold
both and resampled over those plots, is {linea}.}}
\\label{{tab:especie}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrrccccc}}
\\toprule
Species & $n$ & plots ($n_{{\\text{{eff}}}}$) & SegmentAnyTree & AMS3D & ForestFormer3D
& TreeLite3D & watershed\\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


# ------------------------------------------------------------------ cuadro V
def robustez():
    """Cuadro a una columna: la tolerancia de emparejamiento va en el texto."""
    fil = []
    for m in MET:
        e = B["emparejamiento"][m]
        reg = B["region"][m]
        fil.append(
            f"{CORTO[m]} & {n(B['distancia'][m]['3.5']['recall'])} & "
            f"{n(e['recall_hungaro'])} & "
            f"{n(reg['casco +0 m']['precision'])} & "
            f"{n(reg['casco +1,5 m']['precision'])} & "
            f"{n(reg['casco +3 m']['precision'])} & "
            f"{n(reg['tile entero']['precision'])}\\\\")
    cuerpo = "\n".join(fil)
    escribe("tab_robustez.tex", f"""\\begin{{table}}[!t]
\\centering
\\caption{{Sensitivity to two of the three choices a detection evaluator has to make:
greedy against optimal one-to-one assignment, and the region over which detections are
counted. The ordering of the methods survives both, and the matching tolerance, reported
in the text; the absolute numbers do not.}}
\\label{{tab:robustez}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Recall, 3.5~m}} &
\\multicolumn{{4}}{{c}}{{Precision by scoring region}}\\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-7}}
Method & greedy & optimal & hull & $+1.5$~m & $+3$~m & tile\\\\
\\midrule
{cuerpo}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


if __name__ == "__main__":
    principal(); densidad(); treelite(); especies(); robustez()
