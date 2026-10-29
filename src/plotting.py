import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.data import RESULTS_DIR

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8a8985"
MODEL_COLORS = {"XGBoost": SERIES[0], "RandomForest": SERIES[1], "LogReg": SERIES[2],
                "kNN": SERIES[3], "NaiveBayes": SERIES[4], "Dummy": NEUTRAL}
MODEL_MARKERS = {"XGBoost": "o", "RandomForest": "s", "LogReg": "^", "kNN": "D", "NaiveBayes": "v", "Dummy": "x"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq", ["#f7f9fc", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#c43c3b", "#e34948", "#f0efec", "#2a78d6", "#1c5cab"])


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": TEXT_MUTED,
        "axes.labelcolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
    })


def heatmap(ax, data, cmap, vmin, vmax, fmt="{:.2f}", xlabels=None, ylabels=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(data.shape[1]), xlabels if xlabels is not None else [], rotation=45, ha="right")
    ax.set_yticks(range(data.shape[0]), ylabels if ylabels is not None else [])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            position = (value - vmin) / (vmax - vmin)
            dark = abs(position - 0.5) > 0.3 if vmin < 0 else position > 0.6
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8,
                    color="white" if dark else TEXT)
    return im


def save(fig, name):
    fig.savefig(RESULTS_DIR / name, bbox_inches="tight")
