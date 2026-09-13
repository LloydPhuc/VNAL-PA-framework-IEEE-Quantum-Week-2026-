"""
=============================================================================
 01 - COMPUTE & CACHE MATRICES  (Relevance + Redundancy)
=============================================================================
 Tao 2 ma tran dau vao cho toan bo pipeline, luu vao FULL_PIPELINE/cache:
   - relevance.npy     : Rel(i) = MI(band_i ; label), KSG estimator, 100 bootstrap
                         (LAY tu file precomputed da tin cay: mean_relevance_bits_100iters.npy),
                         clip ve >= 0 (VNAL can Rel >= 0).
   - redundancy_lw.npy : Red(i,j) = |correlation| suy ra tu LEDOIT-WOLF shrinkage
                         covariance (DUNG theo bai bao). Tinh lai TU RAW, KHONG dung
                         file "..._bits..." cu.

 Vi sao Ledoit-Wolf: uoc luong covariance on dinh khi so chieu lon so voi mau,
 shrink ve dang chuan -> ma tran correlation tot, dung dinh nghia trong paper.

 CHAY DAU TIEN trong pipeline. Cac script 02..06 doc 2 file cache nay.
=============================================================================
"""
import os
import gc
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


def kfind(basename, hint):
    """Tren Kaggle mount path co the khac nhau -> tu do de quy /kaggle/input theo ten file."""
    if os.path.exists(hint):
        return hint
    hits = glob.glob(f"/kaggle/input/**/{basename}", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Khong tim thay '{basename}' (hint={hint}). "
                            f"Hay attach dataset chua file nay vao notebook.")


def resolve_paths():
    base = r"D:\HYPERSPECTRAL 1"
    raw = os.path.join(base, "INPUT", "RAW DATA")
    mi = os.path.join(base, "INPUT", "new MI")
    if os.path.isdir("/kaggle/input"):
        RAW = {
            "reflectance": kfind("reflectance_260bands-003.npy", ""),
            "labels":      kfind("Eggplant_Labels.npy", ""),
            "wavelengths": kfind("wavelengths_260bands.npy", ""),
            "mi_mean":     kfind("mean_relevance_bits_100iters.npy", ""),
        }
        ROOT = "/kaggle/working/FULL_PIPELINE"
    else:
        RAW = {
            "reflectance": os.path.join(raw, "reflectance_260bands-003.npy"),
            "labels":      os.path.join(raw, "Eggplant_Labels.npy"),
            "wavelengths": os.path.join(raw, "wavelengths_260bands.npy"),
            "mi_mean":     os.path.join(mi, "mean_relevance_bits_100iters.npy"),
        }
        ROOT = os.path.join(base, "FULL_PIPELINE")
    return RAW, ROOT


CFG = {
    "red_samples": 100000,   # so pixel de uoc luong covariance (du lon so voi 260 band)
    "seed": 42,
}


def main():
    RAW, ROOT = resolve_paths()
    CACHE = os.path.join(ROOT, "cache")
    os.makedirs(CACHE, exist_ok=True)

    # ---------- Relevance (KSG 100-iter, precomputed) ----------
    print("[Relevance] load precomputed KSG-100 MI ...")
    rel = np.load(RAW["mi_mean"]).astype(np.float64)
    rel = np.maximum(rel, 0.0)
    np.save(os.path.join(CACHE, "relevance.npy"), rel)
    print(f"  -> relevance.npy  shape={rel.shape}  min={rel.min():.4f} max={rel.max():.4f}")

    # ---------- Redundancy (Ledoit-Wolf shrinkage) ----------
    print("[Redundancy] computing Ledoit-Wolf covariance from raw ...")
    X = np.load(RAW["reflectance"], mmap_mode="r")
    y = np.load(RAW["labels"])
    rows, cols = np.where(y > 0)
    n_fg = len(rows)
    rng = np.random.default_rng(CFG["seed"])
    sel = rng.choice(n_fg, size=min(CFG["red_samples"], n_fg), replace=False)
    Xs = X[rows[sel], cols[sel], :].astype(np.float64)
    del X, y
    gc.collect()
    print(f"  sampled {Xs.shape[0]} px x {Xs.shape[1]} bands")

    Xs = StandardScaler().fit_transform(Xs)        # chuan hoa truoc khi shrink
    lw = LedoitWolf().fit(Xs)
    cov = lw.covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    R = np.abs(corr)
    np.fill_diagonal(R, 0.0)
    R = np.clip(R, 0.0, 1.0)
    np.save(os.path.join(CACHE, "redundancy_lw.npy"), R)
    print(f"  -> redundancy_lw.npy  shape={R.shape}  shrinkage={lw.shrinkage_:.4f}  "
          f"mean|corr|={R[np.triu_indices_from(R,1)].mean():.4f}")

    print("\nDONE. cache ->", CACHE)


if __name__ == "__main__":
    main()
