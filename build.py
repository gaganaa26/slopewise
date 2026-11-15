"""
SLOPEWISE build.py  -  Engine 1 outputs  ->  app/data.js + app/assets/*.png
Adds: AOI crop, OSM roads/places (Overpass, cached), optional DEM slope + downslope
flow-path exposure, optional rainfall value, transparent priority index,
SEPARATE measurement-confidence score, plain-language explanations.

Usage:
  python src/build.py --stems <engine1_stem_oldest> [<stem2> ...] --lon 79.56 --lat 30.55 \
        --site "Joshimath, Uttarakhand" --maturity VALIDATED [--dem data/aux/dem.tif] [--rain_mm 120]
"""
import argparse, json, os, re, math
import numpy as np, rasterio, geopandas as gpd, requests
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.transform import rowcol, Affine
from shapely.geometry import Point, LineString, box
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--stems", nargs="+", required=True, help="engine1 output stems (file name without suffix)")
ap.add_argument("--lon", type=float, required=True)
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--half_km", type=float, default=15)
ap.add_argument("--site", default="Study area")
ap.add_argument("--maturity", default="VALIDATED")
ap.add_argument("--dem", default=None, help="GeoTIFF DEM in EPSG:4326 (Copernicus/SRTM)")
ap.add_argument("--rain_mm", type=float, default=None, help="accumulated rainfall (mm, e.g. 30-day IMERG) you looked up")
ap.add_argument("--max_hotspots", type=int, default=40)
A = ap.parse_args()

OUT, APP = "outputs", "app"
os.makedirs(f"{APP}/assets", exist_ok=True); os.makedirs("data/aux", exist_ok=True)
dlat = A.half_km / 111.0
dlon = A.half_km / (111.0 * math.cos(math.radians(A.lat)))
W, S, E, N = A.lon - dlon, A.lat - dlat, A.lon + dlon, A.lat + dlat
WEIGHTS = dict(D=0.35, T=0.20, R=0.20, E=0.15, S=0.10)
clip = lambda x, a=0.0, b=1.0: max(a, min(b, x))


# ---------------------------------------------------------------- raster helpers
def to_ll(path):
    with rasterio.open(path) as s:
        t, w, h = calculate_default_transform(s.crs, "EPSG:4326", s.width, s.height, *s.bounds)
        out = np.full((h, w), np.nan, "float32")
        reproject(rasterio.band(s, 1), out, dst_transform=t, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    return out, t


def crop(arr, t):
    win = from_bounds(W, S, E, N, t).round_offsets().round_lengths()
    r0, c0 = max(int(win.row_off), 0), max(int(win.col_off), 0)
    r1 = min(int(win.row_off + win.height), arr.shape[0]); c1 = min(int(win.col_off + win.width), arr.shape[1])
    if r1 <= r0 or c1 <= c0:
        raise SystemExit("AOI does not overlap this granule - check --lon/--lat (lon first = east, lat = north)")
    return arr[r0:r1, c0:c1], t * Affine.translation(c0, r0)


def sample(arr, t, lon, lat, k=1):
    r, c = rowcol(t, lon, lat); r, c = int(r), int(c)
    sub = arr[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
    return float(np.nanmean(sub)) if sub.size and np.isfinite(sub).any() else None


def render(ep, name):
    d, c, sg = ep["d"], ep["c"], ep["sigma"]
    v = np.isfinite(d); a = np.abs(np.where(v, d, 0))
    disp = np.zeros(d.shape + (4,), np.uint8)
    disp[v & (a < 2 * sg)] = (74, 222, 128, 70)
    disp[v & (a >= 2 * sg) & (a < 4 * sg)] = (250, 204, 21, 190)
    disp[v & (a >= 4 * sg)] = (239, 68, 68, 225)
    grey = np.zeros_like(disp); grey[~v & np.isfinite(c)] = (148, 163, 184, 110)
    step = max(1, max(d.shape) // 1800)
    for tag, img in (("disp", disp), ("grey", grey)):
        plt.imsave(f"{APP}/assets/{name}_{tag}.png", img[::step, ::step])


# ---------------------------------------------------------------- epochs (one per Engine-1 run)
epochs = []
for stem in A.stems:
    d, t = to_ll(f"{OUT}/{stem}_disp_mm.tif"); c, _ = to_ll(f"{OUT}/{stem}_coherence.tif")
    d, tt = crop(d, t); c, _ = crop(c, t)
    v = d[np.isfinite(d)]
    sigma = max(1.4826 * float(np.median(np.abs(v - np.median(v)))), 1.0) if v.size else 1.0
    ds = sorted({m[:8] for m in re.findall(r"\d{8}T\d{6}", stem)})
    nr, nc = d.shape; west, north = tt.c, tt.f
    bounds = [[north + tt.e * nr, west], [north, west + tt.a * nc]]
    fmtd = lambda s: f"{s[:4]}-{s[4:6]}-{s[6:]}"
    label = (f"{fmtd(ds[0])} -> {fmtd(ds[-1])}" if len(ds) >= 2 else stem[:40])
    epochs.append(dict(stem=stem, d=d, c=c, tt=tt, sigma=sigma, dates=ds, bounds=bounds, label=label,
                       valid_pct=float(np.isfinite(d).mean() * 100)))
epochs.sort(key=lambda e: e["dates"][0] if e["dates"] else "")
for i, e in enumerate(epochs):
    render(e, f"ep{i}"); print(f"epoch {i}: {e['label']}  sigma={e['sigma']:.2f} mm  valid={e['valid_pct']:.0f}%")
latest = epochs[-1]

# ---------------------------------------------------------------- hotspots of the latest epoch
hs = gpd.read_file(f"{OUT}/{latest['stem']}_hotspots.geojson")
hs = hs[hs.within(box(W, S, E, N))].copy() if len(hs) else hs
if hs.empty:
    raise SystemExit("No hotspots inside the AOI. Re-run engine1 with a lower coherence threshold (e.g. 0.3) or enlarge --half_km.")
hs["rank_key"] = hs["snr"] * np.sqrt(hs["area_ha"])
hs = hs.sort_values("rank_key", ascending=False).head(A.max_hotspots).reset_index(drop=True)
crs_m = gpd.GeoSeries([Point(A.lon, A.lat)], crs=4326).estimate_utm_crs()
hs_m = hs.to_crs(crs_m)


# ---------------------------------------------------------------- OSM (roads + places)
def get_osm():
    S, W, N, E = A.lat-0.1, A.lon-0.1/math.cos(math.radians(A.lat)), A.lat+0.1, A.lon+0.1/math.cos(math.radians(A.lat))
    cache = f"data/aux/osm_{A.lat:.3f}_{A.lon:.3f}_{A.half_km:.0f}.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    q = (f'[out:json][timeout:90];(way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|track|service"]'
         f'({S},{W},{N},{E});node["place"~"city|town|village|hamlet|suburb"]({S},{W},{N},{E}););out geom tags;')
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter", "https://overpass.kumi.systems/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"):
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": "slopewise-hackathon/1.0 (student project)", "Accept": "*/*"}, timeout=120); r.raise_for_status()
            js = r.json(); json.dump(js, open(cache, "w")); return js
        except Exception as ex:
            print("Overpass failed:", url, ex)
    return None


def parse_osm(js):
    ways, places = [], []
    for el in (js or {}).get("elements", []):
        tg = el.get("tags", {})
        if el["type"] == "way" and len(el.get("geometry", [])) > 1:
            ways.append(dict(name=tg.get("name") or tg.get("ref") or "unnamed road", hw=tg.get("highway", "road"),
                             geometry=LineString([(p["lon"], p["lat"]) for p in el["geometry"]])))
        elif el["type"] == "node" and "place" in tg:
            places.append(dict(name=tg.get("name", "unnamed place"), kind=tg["place"], geometry=Point(el["lon"], el["lat"])))
    return (gpd.GeoDataFrame(ways, crs=4326) if ways else None), (gpd.GeoDataFrame(places, crs=4326) if places else None)


roads, places = parse_osm(get_osm())
roads_m = roads.to_crs(crs_m) if roads is not None else None
places_m = places.to_crs(crs_m) if places is not None else None
roads_union = None
if roads_m is not None:
    g = roads_m.geometry
    roads_union = g.union_all() if hasattr(g, "union_all") else g.unary_union
print("OSM:", "roads=%d places=%d" % (len(roads) if roads is not None else 0, len(places) if places is not None else 0))

# ---------------------------------------------------------------- DEM (optional)
DEM = SLOPE = None
if A.dem:
    with rasterio.open(A.dem) as s:
        if s.crs is None or not s.crs.is_geographic:
            print("WARNING: DEM is not in geographic CRS (EPSG:4326) - skipping terrain. Reproject it in QGIS first.")
        else:
            win = from_bounds(W - 0.01, S - 0.01, E + 0.01, N + 0.01, s.transform)
            z = s.read(1, window=win, boundless=True).astype("float32")
            z[z < -1000] = np.nan
            dt = s.window_transform(win)
            DX = abs(dt.a) * 111320 * math.cos(math.radians(A.lat)); DY = abs(dt.e) * 110540
            gy, gx = np.gradient(z)
            SLOPE = np.degrees(np.arctan(np.hypot(gx / DX, gy / DY)))
            DEM = (z, dt, DX, DY)
            print("DEM loaded:", z.shape)


def trace_downslope(lon, lat, max_m=3000):
    z, dt, DX, DY = DEM
    r, c = rowcol(dt, lon, lat); r, c = int(r), int(c)
    if not (0 <= r < z.shape[0] and 0 <= c < z.shape[1]): return None
    path, cum, dist = [(r, c)], [0.0], 0.0
    nb = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while dist < max_m:
        best, bd, bl = None, 0.0, 0.0
        for dr, dc in nb:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < z.shape[0] and 0 <= cc < z.shape[1]) or np.isnan(z[rr, cc]): continue
            L = math.hypot(dr * DY, dc * DX); drop = (z[r, c] - z[rr, cc]) / L
            if drop > bd: bd, best, bl = drop, (rr, cc), L
        if best is None: break          # reached a pit / flat area / DEM edge
        r, c = best; dist += bl; path.append(best); cum.append(dist)
    rows = np.array([p[0] for p in path]) + 0.5; cols = np.array([p[1] for p in path]) + 0.5
    xs, ys = dt * (cols, rows)
    hit = None
    if roads_union is not None and len(xs) > 1:
        pm = gpd.GeoSeries([Point(x, y) for x, y in zip(xs, ys)], crs=4326).to_crs(crs_m)
        idx = np.where(pm.distance(roads_union).values < 30)[0]
        if len(idx): hit = float(cum[int(idx[0])])
    return dict(path=[[float(y), float(x)] for x, y in zip(xs[::3], ys[::3])], length_m=float(dist), hit_m=hit)


# ---------------------------------------------------------------- priority + confidence
def priority(comps):
    av = {k: v for k, v in comps.items() if v is not None}
    ws = sum(WEIGHTS[k] for k in av)
    return (sum(WEIGHTS[k] * v for k, v in av.items()) / ws if ws else None), ws


out_h = []
for i, row in hs.iterrows():
    lon, lat, pm_ = row.geometry.x, row.geometry.y, hs_m.geometry.iloc[i]
    snr = float(row["snr"]); coh = float(row["mean_coh"]); vf = float(row["valid_frac_bbox"])
    road = place = None
    if roads_m is not None:
        dd = roads_m.distance(pm_); j = dd.idxmin()
        road = dict(name=str(roads_m.loc[j, "name"]), kind=str(roads_m.loc[j, "hw"]), dist_m=float(dd.min()))
    if places_m is not None:
        dd = places_m.distance(pm_); j = dd.idxmin()
        place = dict(name=str(places_m.loc[j, "name"]), kind=str(places_m.loc[j, "kind"]), dist_m=float(dd.min()))
    slope = sample(SLOPE, DEM[1], lon, lat, 2) if SLOPE is not None else None
    down = trace_downslope(lon, lat) if DEM is not None else None
    series = [dict(label=e["label"], mm=sample(e["d"], e["tt"], lon, lat, 1)) for e in epochs]

    Dn = clip(snr / 6.0)
    Tn = clip(slope / 45.0) if slope is not None else None
    Rn = clip(A.rain_mm / 300.0) if A.rain_mm is not None else None
    parts = []
    if road: parts.append(clip(1 - road["dist_m"] / 1000.0))
    if place: parts.append(0.8 * clip(1 - place["dist_m"] / 1500.0))
    En = max(parts) if parts else None
    if En is not None and down and down["hit_m"] is not None: En = clip(En + 0.25)
    comps = dict(D=Dn, T=Tn, R=Rn, E=En, S=None)          # S needs >=3 consistent epochs: not computed here
    P, ws = priority(comps)
    conf = 0.4 * clip((coh - 0.3) / 0.5) + 0.3 * vf + 0.3 * clip(snr / 4.0)
    cl = "High" if conf >= 0.65 else "Medium" if conf >= 0.4 else "Low"

    why = [f"LOS change of {row['mean_mm']:+.1f} mm (max |{row['max_abs_mm']:.1f}| mm) over {row['area_ha']:.1f} ha, "
           f"{snr:.1f}x the local noise floor ({latest['sigma']:.1f} mm)",
           f"Mean coherence {coh:.2f}; {vf * 100:.0f}% valid pixels in the cluster footprint -> confidence {cl}"]
    why.append(f"Terrain slope about {slope:.0f} deg" if slope is not None else "Terrain slope not evaluated (no DEM supplied)")
    why.append(f"Accumulated rainfall context: {A.rain_mm:.0f} mm (user-supplied)" if A.rain_mm is not None else "Rainfall context not evaluated (none supplied)")
    if road: why.append(f"Nearest mapped road: {road['name']} ({road['kind']}), {road['dist_m']:.0f} m")
    if place: why.append(f"Nearest settlement: {place['name']}, {place['dist_m']:.0f} m")
    if down: why.append(f"Steepest-descent path reaches a road after {down['hit_m']:.0f} m" if down["hit_m"] is not None
                        else "Steepest-descent path (3 km max) does not meet a mapped road")
    out_h.append(dict(lon=lon, lat=lat, n_px=int(row["n_px"]), area_ha=float(row["area_ha"]), mean_mm=float(row["mean_mm"]),
                      max_abs_mm=float(row["max_abs_mm"]), coh=coh, valid_frac=vf, snr=snr, slope=slope, road=road,
                      place=place, down=down, comps=comps, P=P, completeness=ws, conf=dict(score=conf, label=cl),
                      why=why, series=series))

out_h.sort(key=lambda h: -(h["P"] if h["P"] is not None else -1))
for k, h in enumerate(out_h, 1): h["id"] = k

meta = dict(site=A.site, center=[A.lat, A.lon], half_km=A.half_km, maturity=A.maturity, weights=WEIGHTS,
            rain_mm=A.rain_mm, dem=DEM is not None, osm=roads is not None, cur_sigma=latest["sigma"],
            epochs=[dict(label=e["label"], granule=e["stem"], dates=e["dates"], sigma=e["sigma"], bounds=e["bounds"],
                         disp=f"assets/ep{i}_disp.png", grey=f"assets/ep{i}_grey.png") for i, e in enumerate(epochs)])
roads_out = [dict(c=[[y, x] for x, y in g.coords], h=hw, n=nm) for g, hw, nm in
             zip(roads.geometry, roads["hw"], roads["name"])] if roads is not None else []
places_out = [dict(n=r["name"], k=r["kind"], lat=r.geometry.y, lon=r.geometry.x) for _, r in places.iterrows()] if places is not None else []
open(f"{APP}/data.js", "w", encoding="utf-8").write(
    "window.SLOPEWISE=" + json.dumps(dict(meta=meta, hotspots=out_h, roads=roads_out, places=places_out)) + ";")
print(f"\nDONE: {len(out_h)} hotspots -> app/data.js. Open app/index.html in a browser.")
for h in out_h[:5]:
    print(f"  #{h['id']}  {h['mean_mm']:+.1f} mm  conf={h['conf']['label']}  P={h['P'] if h['P'] is None else round(h['P'], 2)}")
