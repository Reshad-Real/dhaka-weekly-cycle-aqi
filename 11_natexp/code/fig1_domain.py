"""Fig. 1: study domain and monitoring sites.

(a) Dhaka region: OpenStreetMap road length per 0.05-degree cell (carried over from the
    previous analysis), district boundaries (geoBoundaries ADM2), and the monitoring sites.
(b) The same region: mean TROPOMI tropospheric NO2 column on non-event days.
(c) Bangladesh: divisions (ADM1), the Dhaka-region box, and the national replication sites.

Site positions: the WAQI station listing; the US Embassy from OpenStreetMap (data/station_locations.csv).
Usage: python -I fig1_domain.py <FYDP_D root>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")
BOX = [90.0, 23.4, 90.85, 24.15]
INK, INK2 = "#0b0b0b", "#52514e"
USED = {"US Embassy", "BARC", "Darussalam", "DoE", "Gazipur", "Narayanganj", "Narsingdi"}


def main(root):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    geo = os.path.join(root, "Traffic", "gates", "data")
    a1 = gpd.read_file(os.path.join(geo, "bgd_ADM1.geojson"))
    a2 = gpd.read_file(os.path.join(geo, "bgd_ADM2.geojson"))
    a2c = a2.cx[BOX[0]:BOX[2], BOX[1]:BOX[3]]
    st = pd.read_csv(os.path.join(DATA, "station_locations.csv"))
    dose = pd.read_csv(os.path.join(root, "Traffic", "data_availability", "01_dose_field", "dose_field_0p05deg.csv"))
    no2 = pd.read_csv(os.path.join(root, "02_data", "raw", "no2_field_baseline.csv"))

    fig = plt.figure(figsize=(11.5, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.78], wspace=0.28)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    def sites(a, labels=True):
        dh = st[(st.lon >= BOX[0]) & (st.lon <= BOX[2]) & (st.lat >= BOX[1]) & (st.lat <= BOX[3])]
        for _, r in dh.iterrows():
            emb = r.station == "US Embassy"
            used = r.station in USED
            a.scatter(r.lon, r.lat, s=140 if emb else 42, marker="*" if emb else "o",
                      facecolor=("#c1121f" if emb else ("#0b0b0b" if used else "white")),
                      edgecolor="white" if used else INK, linewidth=0.8 if used else 1.2,
                      zorder=10 if emb else 8)
            if labels:
                name = {"US Embassy": "US Embassy", "DoE": "DoE (Agargaon)"}.get(r.station, r.station)
                dx = {"Darussalam": -0.015, "DoE": -0.006, "BARC": 0.012, "US Embassy": 0.018}.get(r.station, 0.012)
                dy = {"Darussalam": -0.008, "DoE": 0.036, "BARC": -0.024, "US Embassy": 0.024}.get(r.station, 0.0)
                ha = {"Darussalam": "right", "DoE": "right"}.get(r.station, "left")
                a.text(r.lon + dx, r.lat + dy, name, fontsize=6.8, ha=ha, va="center", color=INK, zorder=9,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))

    # (a) road density
    a = ax[0]
    piv = dose.pivot_table(index="lat", columns="lon", values="road_length_km")
    piv = piv.loc[BOX[1] - 0.05:BOX[3] + 0.05, BOX[0] - 0.05:BOX[2] + 0.05]
    m = a.pcolormesh(piv.columns, piv.index, piv.values, cmap="Blues", shading="nearest", vmin=0)
    cb = fig.colorbar(m, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("road length per 0.05° cell (km)", fontsize=7)
    a2c.boundary.plot(ax=a, color=INK2, lw=0.7, zorder=4)
    sites(a)
    a.set_xlim(BOX[0], BOX[2]); a.set_ylim(BOX[1], BOX[3]); a.set_aspect("equal")
    a.set_title("(a) Road network and monitoring sites", loc="left", fontsize=9)
    a.set_xlabel("Longitude (°E)"); a.set_ylabel("Latitude (°N)")

    # (b) NO2 field
    a = ax[1]
    g = no2.copy()
    g["la"] = (g.latitude / 0.05).round() * 0.05
    g["lo"] = (g.longitude / 0.05).round() * 0.05
    p2 = g.pivot_table(index="la", columns="lo", values="tropospheric_NO2_column_number_density")
    m = a.pcolormesh(p2.columns, p2.index, p2.values, cmap="Oranges", shading="nearest")
    cb = fig.colorbar(m, ax=a, fraction=0.046, pad=0.02)
    cb.set_label("mean TROPOMI NO$_2$ column (µmol m$^{-2}$)", fontsize=7)
    a2c.boundary.plot(ax=a, color=INK2, lw=0.7, zorder=4)
    sites(a, labels=False)
    a.set_xlim(BOX[0], BOX[2]); a.set_ylim(BOX[1], BOX[3]); a.set_aspect("equal")
    a.set_title("(b) NO$_2$ column, non-event days", loc="left", fontsize=9)
    a.set_xlabel("Longitude (°E)")

    # (c) Bangladesh and the national sites
    a = ax[2]
    a1.plot(ax=a, color="#f2f1ee", edgecolor=INK2, lw=0.5)
    a.add_patch(plt.Rectangle((BOX[0], BOX[1]), BOX[2] - BOX[0], BOX[3] - BOX[1], fill=False, ec="#c1121f", lw=1.2))
    nat = st[~((st.lon >= BOX[0]) & (st.lon <= BOX[2]) & (st.lat >= BOX[1]) & (st.lat <= BOX[3]))]
    a.scatter(nat.lon, nat.lat, s=26, color=INK, edgecolor="white", lw=0.6, zorder=5)
    for _, r in nat.iterrows():
        lab = r.station.replace(" Chattogram", "").replace("Agrabad CDA", "Agrabad")
        off = {"BTV": (0.08, 0.10), "Agrabad": (0.08, -0.14)}.get(lab, (0.08, 0.0))
        a.text(r.lon + off[0], r.lat + off[1], lab, fontsize=6.3, va="center", color=INK)
    a.text(BOX[2] + 0.05, BOX[3] + 0.08, "Dhaka region (a, b)", fontsize=6.3, color="#c1121f")
    a.set_xlim(88.0, 92.75); a.set_ylim(20.6, 26.7); a.set_aspect("equal")
    a.set_title("(c) Bangladesh", loc="left", fontsize=9)
    a.set_xlabel("Longitude (°E)")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="*", ls="", ms=10, mfc="#c1121f", mec="white", label="US Embassy (reference)"),
               Line2D([], [], marker="o", ls="", ms=6.5, mfc=INK, mec="white", label="DoE CAMS, used for PM$_{2.5}$"),
               Line2D([], [], marker="o", ls="", ms=6.5, mfc="white", mec=INK, label="DoE CAMS, gases only (Savar)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.42, -0.01))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.17)
    fig.savefig(os.path.join(OUT, "fig1_domain.png"), dpi=220)
    print("wrote fig1_domain.png")


if __name__ == "__main__":
    main(sys.argv[1])
