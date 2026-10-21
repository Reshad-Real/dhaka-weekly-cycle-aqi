"""Shared figure style.

Every figure is drawn at the width at which it is printed, the 6.72-inch text width of the
manuscript and the supplement (483.7 pt), so the sizes below are the printed sizes: 8 pt for
tick labels, legends and annotations, 8.5 pt for axis labels and 9 pt for panel titles.
"""
WIDTH = 6.7            # inches, inserted at \textwidth (6.72 in) without rescaling
DPI = 300
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e9e8e4"
SMALL = 8              # the smallest text in any figure


def apply(plt):
    plt.rcParams.update({
        "font.size": SMALL, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "axes.titlepad": 4,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "savefig.dpi": DPI,
    })
