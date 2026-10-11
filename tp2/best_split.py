import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = pd.read_csv("results_final/best_table.csv").dropna(subset=["F1"])
cols = ["model", "setting", "F1", "AP", "precision", "recall", "FP (no-motor)", "median err (px)", "train (min)", "forward (ms)"]
md, tex = "", ""
for cond in ["Pretrained", "From scratch"]:
    t = d[d["condition"] == cond].sort_values("F1", ascending=False)[cols].round(3)
    print("\n" + cond); print(t.to_string(index=False))
    md += f"### {cond}\n\n" + t.to_markdown(index=False) + "\n\n"
    tex += f"% {cond}\n" + t.to_latex(index=False, float_format="%.3f") + "\n"
open("results_final/best_split.md", "w").write(md)
open("results_final/best_split.tex", "w").write(tex)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, cond in zip(axs, ["Pretrained", "From scratch"]):
    t = d[d["condition"] == cond].set_index("model")[["F1", "AP"]]
    t.plot.bar(ax=ax, rot=0, legend=(cond == "Pretrained"))
    ax.set_title(cond, fontsize=10); ax.set_ylim(0, 1); ax.set_xlabel(""); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("results_final/best_split_bars.png", dpi=200)
