"""Fig. 1: study domain and monitoring sites.

(a) Dhaka region: OpenStreetMap road length per 0.05-degree cell (carried over from the
    previous analysis), district boundaries (geoBoundaries ADM2), and the monitoring sites.
(b) The same region: mean TROPOMI tropospheric NO2 column on non-event days.
(c) Bangladesh: divisions (ADM1), the Dhaka-region box, and the national replication sites.

Drawn at the printed width (figstyle.py). Site positions: the WAQI station listing; the US
Embassy from OpenStreetMap (data/station_locations.csv).
Usage: python -I fig1_domain.py <FYDP_D root>
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
DATA = os.path.join(HERE, "..", "data")
BOX = [90.0, 23.4, 90.85, 24.15]
INK, INK2 = fs.INK, fs.INK2
RED = "#c1121f"
USED = {"US Embassy", "BARC", "Darussalam", "DoE", "Gazipur", "Narayanganj", "Narsingdi"}
# label position (lon, lat) and alignment for each Dhaka-region site; the four central sites
# lie within 0.07 degrees of each other, so their labels sit outside the cluster on leaders
LABEL = {"US Embassy": (90.48, 23.725, "left"), "DoE": (90.31, 23.875, "right"),
         "Darussalam": (90.29, 23.79, "right"), "BARC": (90.35, 23.685, "right"),
         "Savar": (90.30, 23.954, "left"), "Gazipur": (90.44, 23.994, "left"),
         "Narsingdi": (90.717, 23.875, "center"), "Narayanganj": (90.525, 23.626, "left")}


def main(root):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fs.apply(plt)
    geo = os.path.join(root, "Traffic", "gates", "data")
    a1 = gpd.read_file(os.path.join(geo, "bgd_ADM1.geojson"))
    a2 = gpd.read_file(os.path.join(geo, "bgd_ADM2.geojson"))
    a2c = a2.cx[BOX[0]:BOX[2], BOX[1]:BOX[3]]
    st = pd.read_csv(os.path.join(DATA, "station_locations.csv"))
    dose = pd.read_csv(os.path.join(root, "Traffic", "data_availability", "01_dose_field", "dose_field_0p05deg.csv"))
    no2 = pd.read_csv(os.path.join(root, "02_data", "raw", "no2_field_baseline.csv"))

    fig = plt.figure(figsize=(fs.WIDTH, 6.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], left=0.085, right=0.975, top=0.965,
                          bottom=0.06, wspace=0.22, hspace=0.22)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
    leg = fig.add_subplot(gs[1, 1])
    leg.axis("off")

    def sites(a, labels=True):
        dh = st[(st.lon >= BOX[0]) & (st.lon <= BOX[2]) & (st.lat >= BOX[1]) & (st.lat <= BOX[3])]
        for _, r in dh.iterrows():
            emb = r.station == "US Embassy"
            used = r.station in USED
            a.scatter(r.lon, r.lat, s=70 if emb else 22, marker="*" if emb else "o",
                      facecolor=(RED if emb else (INK if used else "white")),
                      edgecolor="white" if used else INK, linewidth=0.6 if used else 0.9,
                      zorder=10 if emb else 8)
            if labels:
                name = {"DoE": "DoE (Agargaon)"}.get(r.station, r.station)
                tx, ty, ha = LABEL[r.station]
                far = abs(tx - r.lon) > 0.03 or abs(ty - r.lat) > 0.03
                a.annotate(name, xy=(r.lon, r.lat), xytext=(tx, ty), ha=ha, va="center", color=INK,
                           fontsize=fs.SMALL, zorder=9,
                           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85),
                           arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2, shrinkA=0, shrinkB=3)
                           if far else None)

    def frame(a):
        a.set_xlim(BOX[0], BOX[2]); a.set_ylim(BOX[1], BOX[3]); a.set_aspect("equal")
        a.set_xticks([90.0, 90.2, 90.4, 90.6, 90.8])
        a.set_yticks([23.4, 23.6, 23.8, 24.0])
        a.set_xlabel("Longitude (°E)")

    # (a) road density
    a = ax[0]
    piv = dose.pivot_table(index="lat", columns="lon", values="road_length_km")
    piv = piv.loc[BOX[1] - 0.05:BOX[3] + 0.05, BOX[0] - 0.05:BOX[2] + 0.05]
    m = a.pcolormesh(piv.columns, piv.index, piv.values, cmap="Blues", shading="nearest", vmin=0)
    cb = fig.colorbar(m, ax=a, orientation="horizontal", fraction=0.045, pad=0.16, aspect=32)
    cb.set_label("road length per 0.05° cell (km)")
    a2c.boundary.plot(ax=a, color=INK2, lw=0.5, zorder=4)
    sites(a)
    frame(a)
    a.set_title("(a) Road network and monitoring sites", loc="left")
    a.set_ylabel("Latitude (°N)")

    # (b) NO2 field
    a = ax[1]
    g = no2.copy()
    g["la"] = (g.latitude / 0.05).round() * 0.05
    g["lo"] = (g.longitude / 0.05).round() * 0.05
    p2 = g.pivot_table(index="la", columns="lo", values="tropospheric_NO2_column_number_density")
    m = a.pcolormesh(p2.columns, p2.index, p2.values, cmap="Oranges", shading="nearest")
    cb = fig.colorbar(m, ax=a, orientation="horizontal", fraction=0.045, pad=0.16, aspect=32)
    cb.set_label("TROPOMI NO$_2$ column (µmol m$^{-2}$)")
    a2c.boundary.plot(ax=a, color=INK2, lw=0.5, zorder=4)
    sites(a, labels=False)
    frame(a)
    a.set_title("(b) NO$_2$ column, non-event days", loc="left")

    # (c) Bangladesh and the national sites
    a = ax[2]
    a1.plot(ax=a, color="#f2f1ee", edgecolor=INK2, lw=0.4)
    a.add_patch(plt.Rectangle((BOX[0], BOX[1]), BOX[2] - BOX[0], BOX[3] - BOX[1], fill=False, ec=RED, lw=1.0))
    nat = st[~((st.lon >= BOX[0]) & (st.lon <= BOX[2]) & (st.lat >= BOX[1]) & (st.lat <= BOX[3]))]
    a.scatter(nat.lon, nat.lat, s=16, color=INK, edgecolor="white", lw=0.5, zorder=5)
    for _, r in nat.iterrows():
        lab = r.station.replace(" Chattogram", "").replace("Agrabad CDA", "Agrabad")
        dx, dy, ha = {"BTV": (0.12, 0.16, "left"), "Agrabad": (0.12, -0.2, "left"),
                      "Mymensingh": (-0.12, 0.0, "right"), "Cumilla": (0.12, -0.12, "left")}.get(lab, (0.12, 0.0, "left"))
        a.text(r.lon + dx, r.lat + dy, lab, fontsize=fs.SMALL, ha=ha, va="center", color=INK)
    a.text(BOX[0], BOX[3] + 0.16, "Dhaka region", fontsize=fs.SMALL, color=RED, ha="left")
    a.set_xlim(88.0, 92.75); a.set_ylim(20.6, 26.7); a.set_aspect("equal")
    a.set_xticks([88, 89, 90, 91, 92])
    a.set_title("(c) Bangladesh", loc="left")
    a.set_xlabel("Longitude (°E)")
    a.set_ylabel("Latitude (°N)")

    handles = [Line2D([], [], marker="*", ls="", ms=8, mfc=RED, mec="white", label="US Embassy (reference)"),
               Line2D([], [], marker="o", ls="", ms=5, mfc=INK, mec="white", label="DoE CAMS, used for PM$_{2.5}$"),
               Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=INK, label="DoE CAMS, gases only (Savar)"),
               Patch(fc="none", ec=RED, lw=1.0, label="Dhaka region, panels (a) and (b)")]
    leg.legend(handles=handles, loc="center left", frameon=False, borderaxespad=0, labelspacing=0.8)
    fig.savefig(os.path.join(OUT, "fig1_domain.png"))
    print("wrote fig1_domain.png")


if __name__ == "__main__":
    main(sys.argv[1])
