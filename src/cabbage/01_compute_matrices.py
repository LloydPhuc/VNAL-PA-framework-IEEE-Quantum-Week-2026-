"""
=============================================================================
 01 (CABBAGE) - COMPUTE & CACHE MATRICES  (Relevance + Redundancy) FROM RAW
=============================================================================
 Khac voi eggplant: KHONG dung file MI/redundancy precomputed cua cabbage.
 Tinh CA HAI tu raw (chi dung reflectance_260bands.npy + Cabbage_Labels.npy):
   - relevance.npy     : Rel(i) = MI(band_i; label), KSG estimator, bootstrap
                         (giong PHUONG PHAP cua eggplant GT: knn=10, MinMaxScale),
                         clip ve >= 0.
   - redundancy_lw.npy : Red(i,j) = |corr| tu Ledoit-Wolf shrinkage covariance.

 CHAY DAU TIEN. Cac script 02..06 doc 2 file cache nay.
 LUU Y: buoc nay nang nhat (100-iter KSG MI). Giam mi_iters/mi_per_class neu can.
=============================================================================
"""
import os
import gc
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.covariance import LedoitWolf


def kfind(basename):
    hits = glob.glob(f"/kaggle/input/**/{basename}", recursive=True)
    return hits[0] if hits else None


def resolve_paths():
    base = r"D:\HYPERSPECTRAL 1\INPUT\CABBAGE"     # local fallback (neu co)
    if os.path.isdir("/kaggle/input"):
        RAW = {
            "reflectance": kfind("reflectance_260bands.npy"),
            "labels":      kfind("Cabbage_Labels.npy"),
            "wavelengths": kfind("wavelengths_260.npy"),
        }
        ROOT = "/kaggle/working/CABBAGE"
    else:
        RAW = {
            "reflectance": os.path.join(base, "reflectance_260bands.npy"),
            "labels":      os.path.join(base, "Cabbage_Labels.npy"),
            "wavelengths": os.path.join(base, "wavelengths_260.npy"),
        }
        ROOT = r"D:\HYPERSPECTRAL 1\FULL_PIPELINE\CABBAGE"
    return RAW, ROOT


CFG = {
    "mi_iters":      100,      # so bootstrap (giam de chay nhanh hon)
    "mi_per_class":  5000,     # px/class moi iter (eggplant GT dung 15000; giam cho 12h)
    "mi_neighbors":  10,       # KSG k (giong eggplant)
    "red_samples":   100000,   # px de uoc luong covariance
    "seed":          42,
}


def load_clean(path_X, path_y):
    print("[Data] loading reflectance + labels ...", flush=True)
    X = np.load(path_X, mmap_mode="r")
    y = np.load(path_y)
    rows, cols = np.where(y > 0)
    Xc = X[rows, cols, :].astype(np.float32)
    yc = (y[rows, cols] - int(y[rows, cols].min())).astype(np.int16)
    del X, y
    gc.collect()
    print(f"  -> {Xc.shape[0]} px, {Xc.shape[1]} bands, {len(np.unique(yc))} classes", flush=True)
    return Xc, yc


def compute_relevance_ksg(Xc, yc, n_iters, per_class, n_neighbors, base_seed):
    classes = np.unique(yc)
    mat = np.zeros((n_iters, Xc.shape[1]), dtype=np.float64)
    for it in range(n_iters):
        seed = base_seed + it
        rng = np.random.default_rng(seed)
        idx = []
        for c in classes:
            ci = np.where(yc == c)[0]
            rep = len(ci) < per_class
            idx.extend(rng.choice(ci, size=per_class, replace=rep))
        idx = np.array(idx)
        Xs = MinMaxScaler().fit_transform(Xc[idx])
        mat[it] = mutual_info_classif(Xs, yc[idx], discrete_features=False,
                                      n_neighbors=n_neighbors, random_state=seed)
        if (it + 1) % 10 == 0 or it == 0:
            print(f"  [KSG MI] {it+1}/{n_iters}", flush=True)
    return mat.mean(axis=0)


def main():
    RAW, ROOT = resolve_paths()
    CACHE = os.path.join(ROOT, "cache")
    os.makedirs(CACHE, exist_ok=True)
    for k, v in RAW.items():
        if v is None or not os.path.exists(v):
            raise FileNotFoundError(f"Khong tim thay {k}: {v}")

    Xc, yc = load_clean(RAW["reflectance"], RAW["labels"])

    print("[Relevance] computing KSG MI from raw ...", flush=True)
    rel = compute_relevance_ksg(Xc, yc, CFG["mi_iters"], CFG["mi_per_class"],
                                CFG["mi_neighbors"], CFG["seed"])
    rel = np.maximum(rel, 0.0)
    np.save(os.path.join(CACHE, "relevance.npy"), rel)
    print(f"  -> relevance.npy  shape={rel.shape}  min={rel.min():.4f} max={rel.max():.4f}", flush=True)

    print("[Redundancy] computing Ledoit-Wolf covariance from raw ...", flush=True)
    rng = np.random.default_rng(CFG["seed"])
    sel = rng.choice(Xc.shape[0], size=min(CFG["red_samples"], Xc.shape[0]), replace=False)
    Xs = StandardScaler().fit_transform(Xc[sel].astype(np.float64))
    lw = LedoitWolf().fit(Xs)
    cov = lw.covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    R = np.abs(cov / np.outer(d, d))
    np.fill_diagonal(R, 0.0)
    R = np.clip(R, 0.0, 1.0)
    np.save(os.path.join(CACHE, "redundancy_lw.npy"), R)
    print(f"  -> redundancy_lw.npy  shape={R.shape}  shrinkage={lw.shrinkage_:.4f}  "
          f"mean|corr|={R[np.triu_indices_from(R,1)].mean():.4f}", flush=True)

    del Xc, yc
    gc.collect()
    print("\nDONE. cache ->", CACHE, flush=True)


if __name__ == "__main__":
    main()
