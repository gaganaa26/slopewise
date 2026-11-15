import sys, os, json
import numpy as np, h5py, rasterio
from rasterio.transform import Affine
from scipy import ndimage as ndi
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

LAMBDA = 0.2384   # m, approx NISAR L-band; VERIFY against file metadata

def find(f, name, must=None):
    hits = []
    def v(n, o):
        if isinstance(o, h5py.Dataset) and n.split("/")[-1] == name and (must is None or must in n):
            hits.append(n)
    f.visititems(v)
    hits.sort(key=lambda s: ("frequencyA" not in s, s))
    return hits

def pick(f, name, must=None, required=True):
    h = find(f, name, must)
    if not h:
        if required: raise KeyError(f"{name} not found (must={must})")
        return None
    print(f"  using {name}: {h[0]}  ({len(h)} candidate(s))")
    return h[0]

def pick_coord(f, name, n):
    # the file has several coordinate grids (e.g. pixelOffsets); use the one matching the phase grid
    for p in find(f, name, "frequencyA"):
        if f[p].shape[0] == n:
            print(f"  using {name}: {p}  (length {n})")
            return f[p][()]
    raise KeyError(f"no {name} with length {n}")

def get_epsg(f):
    for p in find(f, "projection"):
        d = f[p]
        e = d.attrs.get("epsg_code", None)
        if e is None: e = d[()]
        try: return int(np.asarray(e).item())
        except Exception: continue
    raise KeyError("EPSG not found; check projection dataset")

def run(path, coh_thr=0.5, k=3.0, min_px=9, min_mm=10.0, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    with h5py.File(path, "r") as f:
        up = pick(f, "unwrappedPhase")
        base = up.rsplit("/", 1)[0]
        coh = pick(f, "coherenceMagnitude", must=base.split("/unwrappedInterferogram")[0] + "/unwrappedInterferogram")
        cc  = pick(f, "connectedComponents", must="unwrappedInterferogram")
        iono = pick(f, "ionospherePhaseScreen", required=False)
        epsg = get_epsg(f)
        phase = f[up][()].astype("float32")
        xs = pick_coord(f, "xCoordinates", phase.shape[1])
        ys = pick_coord(f, "yCoordinates", phase.shape[0])
        c = f[coh][()].astype("float32")
        comp = f[cc][()]
        ion = f[iono][()].astype("float32") if iono else None

    print("shapes:", phase.shape, c.shape, comp.shape, None if ion is None else ion.shape, "EPSG", epsg)
    if ion is not None and ion.shape == phase.shape:
        ionv = np.isfinite(ion) & (ion != 0)
        phase = np.where(ionv, phase - ion, phase)
    valid = np.isfinite(phase) & (phase != 0) & np.isfinite(c) & (c >= coh_thr) & (comp > 0)

    d = -LAMBDA * phase / (4 * np.pi)
    d[~valid] = np.nan
    # remove a best-fit plane (long-wavelength orbit/atmosphere ramp) using reliable pixels
    H_, W_ = d.shape
    yy, xx = np.nonzero(valid & (c > 0.5))
    if len(yy) > 2000:
        sel = np.random.default_rng(0).choice(len(yy), 50000 if len(yy) > 50000 else len(yy), replace=False)
        yy, xx = yy[sel], xx[sel]
        A_ = np.c_[xx, yy, np.ones(len(xx))]
        coef = np.linalg.lstsq(A_, d[yy, xx], rcond=None)[0]
        d = d - (coef[0] * np.arange(W_)[None, :] + coef[1] * np.arange(H_)[:, None] + coef[2]).astype("float32")
        print("  removed planar ramp from displacement")
    ref = np.nanmedian(d[valid & (c > 0.8)]) if np.any(valid & (c > 0.8)) else np.nanmedian(d)
    d = (d - ref) * 1000.0
    dv = d[valid]
    sigma = 1.4826 * np.nanmedian(np.abs(dv - np.nanmedian(dv)))
    print(f"valid {valid.mean()*100:.1f}% | range {np.nanmin(dv):.1f}..{np.nanmax(dv):.1f} mm | robust sigma {sigma:.2f} mm")

    thr = max(k * sigma, min_mm)
    hot = valid & (np.abs(d) > thr)
    hot = ndi.binary_opening(hot, structure=np.ones((2, 2)))
    lab, n = ndi.label(hot, structure=np.ones((3, 3)))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    tf = Affine(dx, 0, xs[0] - dx / 2, 0, dy, ys[0] - dy / 2)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)

    rows = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() < min_px: continue
        dd, cc_ = d[sl][m], c[sl][m]
        rr, cl = np.nonzero(m)
        cx = xs[sl[1].start + int(cl.mean())]; cy = ys[sl[0].start + int(rr.mean())]
        lon, lat = to_ll.transform(cx, cy)
        box_valid = valid[sl].mean()
        rows.append(dict(id=len(rows)+1, n_px=int(m.sum()), area_ha=float(m.sum()*abs(dx*dy)/1e4),
                         mean_mm=float(dd.mean()), max_abs_mm=float(np.abs(dd).max()),
                         mean_coh=float(cc_.mean()), valid_frac_bbox=float(box_valid),
                         snr=float(np.abs(dd.mean())/sigma if sigma > 0 else np.nan),
                         geometry=Point(lon, lat)))
    gdf = gpd.GeoDataFrame(rows, crs=4326)
    stem = os.path.splitext(os.path.basename(path))[0]
    gdf.to_file(f"{outdir}/{stem}_hotspots.geojson", driver="GeoJSON")

    prof = dict(driver="GTiff", height=d.shape[0], width=d.shape[1], count=1, dtype="float32",
                crs=f"EPSG:{epsg}", transform=tf, nodata=np.nan, compress="deflate")
    for name, arr in [("disp_mm", d), ("coherence", c)]:
        with rasterio.open(f"{outdir}/{stem}_{name}.tif", "w", **prof) as dst:
            dst.write(arr.astype("float32"), 1)
    print(f"hotspots: {len(gdf)} -> {outdir}/{stem}_hotspots.geojson")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    v = max(thr * 2, 20)
    ax[0].imshow(d, cmap="RdBu", vmin=-v, vmax=v); ax[0].set_title("LOS displacement (mm, masked)")
    ax[1].imshow(c, cmap="gray", vmin=0, vmax=1); ax[1].set_title("coherence")
    fig.savefig(f"{outdir}/{stem}_preview.png", dpi=110)

if __name__ == "__main__":
    run(sys.argv[1], coh_thr=float(sys.argv[2]) if len(sys.argv) > 2 else 0.5)
