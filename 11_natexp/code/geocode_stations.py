"""Coordinates of the monitoring sites.

The DoE CAMS stations take the coordinates the World Air Quality Index project publishes for
each station (the "geo" field of the station's page on aqicn.org, read 17 October 2026),
together with the host institution it names.  An earlier version geocoded the host
institutions with OpenStreetMap Nominatim; several of those positions fell on the city or
division centroid, up to 51 km from the station, so they are no longer used.

The US Embassy monitor is not in that listing.  Its position is the embassy compound as
OpenStreetMap Nominatim places it (Madani Avenue, Baridhara), geocoded on 16 October 2026.

Writes 11_natexp/data/station_locations.csv.  No network access is needed.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "station_locations.csv")

# station as named in the analysis -> (host institution, latitude, longitude, source)
SITES = {
    "US Embassy": ("U.S. Embassy, Madani Avenue, Baridhara", 23.7967, 90.4220, "OpenStreetMap Nominatim"),
    "BARC": ("Bangladesh Agricultural Research Council, Farmgate", 23.759410, 90.388679, "WAQI"),
    "Darussalam": ("National Institute of Mass Communication, Darus Salam", 23.780775, 90.355410, "WAQI"),
    "DoE": ("Department of Environment, Paribesh Bhaban, Agargaon", 23.777394, 90.372603, "WAQI"),
    "Savar": ("Atomic Energy Research Establishment, Savar", 23.953704, 90.279794, "WAQI"),
    "Gazipur": ("Dhirasram Road, Gazipur", 23.994128, 90.422315, "WAQI"),
    "Narayanganj": ("Khanpur Hospital, Narayanganj", 23.626079, 90.507200, "WAQI"),
    "Narsingdi": ("Narsingdi Sadar Upazila Complex", 23.932456, 90.716598, "WAQI"),
    "Agrabad CDA Chattogram": ("Agrabad CDA Residential Area, Chattogram", 22.323020, 91.802336, "WAQI"),
    "BTV Chattogram": ("Bangladesh Television Chattogram Center, Khulshi", 22.360487, 91.800454, "WAQI"),
    "Barishal": ("Divisional Forest Officer Office, Barishal", 22.710287, 90.362598, "WAQI"),
    "Cumilla": ("Department of Environment, Cumilla", 23.472988, 91.180671, "WAQI"),
    "Khulna": ("Jalil Sarani, Boyra, Khulna", 22.835775, 89.529056, "WAQI"),
    "Mymensingh": ("Department of Environment, Mymensingh", 24.762458, 90.402102, "WAQI"),
    "Rajshahi": ("Divisional Forest Officer Office, Paba, Rajshahi", 24.383320, 88.608007, "WAQI"),
    "Rangpur": ("Bangladesh Television Rangpur Subcenter", 25.747371, 89.228931, "WAQI"),
    "Sylhet": ("Bangladesh Red Crescent Society, Sylhet", 24.888334, 91.867347, "WAQI"),
}


def main():
    rows = ["station,host,lat,lon,source"]
    for st, (host, lat, lon, src) in SITES.items():
        rows.append('"%s","%s",%.6f,%.6f,"%s"' % (st, host, lat, lon, src))
    open(OUT, "w", encoding="utf-8").write("\n".join(rows) + "\n")
    print("wrote %d stations -> %s" % (len(SITES), OUT))


if __name__ == "__main__":
    main()
