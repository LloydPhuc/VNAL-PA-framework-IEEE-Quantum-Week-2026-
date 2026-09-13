"""
=============================================================================
 BASELINE BAND SELECTION - Reproducible (Kaggle-ready, <12h)
=============================================================================
 Sinh ra DANH SACH BAND cho cac phuong phap baseline trong Bang I, cai dat
 BAM SAT BAI BAO GOC cua tung phuong phap.

 NGUYEN TAC TRANH BIAS (rat quan trong):
   - KHONG dung ma tran cua paper VNAL-PA
       (mean_relevance_bits_100iters.npy / redundancy_matrix_bits_final.npy).
   - MOI feature can thiet (relevance, redundancy, similarity, projection...)
     deu duoc TINH LAI TU RAW theo cach THONG THUONG, dung uoc luong chuan:
       * Relevance(MI)   : sklearn.mutual_info_classif (knn, n_neighbors=3 default)
       * Relevance(F)    : sklearn.f_classif (ANOVA F) - dung cho mRMR continuous
       * Redundancy/Sim  : |Pearson correlation| (np.corrcoef)
       * UVE             : PLS regression coefficient stability
       * SPA             : successive orthogonal projections tren ma tran du lieu
   => Moi baseline xai dung loai feature ma BAI BAO GOC cua no quy dinh,
      khong xai feature "uu ai" tu phuong phap de xuat.

 PHUONG PHAP & NGUON GOC:
   - Top-MI                : chon top-K theo MI (Peng 2005, dang don gian nhat)
   - mRMR (FCD/MID)        : Peng, Long, Ding 2005 - continuous => F-test relevance,
                             |corr| redundancy, tieu chi MID (Difference)
   - DPP-greedy            : Kulesza & Taskar 2012 - greedy MAP log-det
                             (fast greedy: Chen et al. 2018)
   - Cluster-Exemplar      : cluster band theo (1-|corr|), chon exemplar/cluster
   - SPA                   : Araujo et al. 2001 - Successive Projections Algorithm
   - UVE                   : Centner et al. 1996 - Uninformative Variable Elimination
   - GA                    : wrapper Genetic Algorithm (TUY CHON - rat ton thoi gian)
   - Random                : K band ngau nhien (trung binh nhieu lan)

 KAGGLE TIME (12h):
   - Nap raw 1 lan, lay subset can bang lop de tinh feature.
   - Tat ca baseline (tru GA) chay nhanh (vai phut). GA mac dinh TAT.
   - Co timer cho tung buoc; co the chinh sample size trong CFG.

 INPUT (giong het 12445.py):
   reflectance / labels / wavelengths  (path Kaggle ben duoi)
 OUTPUT:
   baseline_bands.json , baseline_bands.csv
=============================================================================
"""

import os
import gc
import json
import time
import math
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.cluster import AgglomerativeClustering
from sklearn.cross_decomposition import PLSRegression

# =============================================================================
# CONFIG
# =============================================================================

import glob as _glob
def _base(p):                       # basename an toan cho ca Windows lan POSIX
    return p.replace("\\", "/").split("/")[-1]
def _kfind(hint):
    if os.path.exists(hint):
        return hint
    hits = _glob.glob(f"/kaggle/input/**/{_base(hint)}", recursive=True)
    return hits[0] if hits else hint

LOCAL_PATHS = {
    "reflectance": r"D:\HYPERSPECTRAL 1\INPUT\RAW DATA\reflectance_260bands.npy",
    "labels":      r"D:\HYPERSPECTRAL 1\INPUT\RAW DATA\Cabbage_Labels.npy",
    "wavelengths": r"D:\HYPERSPECTRAL 1\INPUT\RAW DATA\wavelengths_260.npy",
}
PATHS = {k: _kfind(v) for k, v in LOCAL_PATHS.items()}

OUTPUT_DIR = "/kaggle/working/CABBAGE/results"
if not os.path.isdir("/kaggle/input"):
    OUTPUT_DIR = r"D:\HYPERSPECTRAL 1\FULL_PIPELINE\CABBAGE\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "N_BANDS": 260,
    "K": 64,
    "SEED": 42,

    # --- subset de TINH FEATURE (can bang lop) ---
    "feat_per_class": 5000,     # MI/F/corr  (15000 px tong) - du on dinh, van nhanh
    "mi_neighbors": 3,          # sklearn default (cach thong thuong, KHAC paper)

    # --- SPA / UVE dung subset rieng (nho hon cho nhanh) ---
    "spa_samples": 3000,
    "spa_n_starts": 1,          # 1 = seed tu band variance lon nhat (nhanh)
    "uve_samples": 5000,
    "uve_resamples": 50,
    "uve_components": 12,
    "uve_noise": 1e-10,

    # --- DPP ---
    "dpp_quality_scale": 1.0,   # q_i = exp(scale * rel_norm)

    # --- Random ---
    "random_runs": 5,

    # --- GA wrapper (TON THOI GIAN - mac dinh TAT, dung band co san) ---
    "run_GA": False,
    "ga_pop": 30,
    "ga_gen": 20,
    "ga_eval_per_class": 800,   # subset nho de cham SVM trong GA
    "ga_folds": 3,

    # --- Cac phuong phap TU TINH (nhanh, vectorized) ---
    "methods": ["Top-MI", "mRMR", "DPP-greedy", "Cluster-Exemplar", "SPA", "UVE", "Random"],

    # --- Cac phuong phap DOC BAND CO SAN (GA/SPA/UVE da chay rieng) ---
    "prebuilt": {
        "GA":  "selected_bands_GA.json",
        "SPA": "selected_bands_SPA.json",
        "UVE": "selected_bands_UVE.json",
    },
    "prebuilt_dirs": [
        r"D:\HYPERSPECTRAL 1\BAND CÁC PP KHÁC",
        "/kaggle/input/datasets/lnguynminhtn/qubo-scripts/BAND CÁC PP KHÁC",
        "/kaggle/input/qubo-scripts/BAND CÁC PP KHÁC",
    ],
}

K = CFG["K"]
N = CFG["N_BANDS"]
_T0 = None


def tlog(msg):
    global _T0
    if _T0 is None:
        _T0 = time.perf_counter()
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# DATA
# =============================================================================

def load_clean(path_X, path_y):
    tlog("Loading reflectance + labels ...")
    X = np.load(path_X, mmap_mode="r")
    y = np.load(path_y)
    rows, cols = np.where(y > 0)
    Xc = X[rows, cols, :].astype(np.float32)
    yc = (y[rows, cols] - int(y[rows, cols].min())).astype(np.int16)
    del X, y
    gc.collect()
    tlog(f"  -> {Xc.shape[0]} px, {Xc.shape[1]} bands, {len(np.unique(yc))} classes")
    return Xc, yc


def balanced_subset(Xc, yc, per_class, seed=42, scale="minmax"):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(yc):
        ci = np.where(yc == c)[0]
        rep = len(ci) < per_class
        idx.extend(rng.choice(ci, size=per_class, replace=rep))
    idx = np.array(idx)
    X = Xc[idx].astype(np.float64)
    if scale == "minmax":
        X = MinMaxScaler().fit_transform(X)
    elif scale == "std":
        X = StandardScaler().fit_transform(X)
    return X, yc[idx]


# =============================================================================
# STANDARD FEATURES (tinh thong thuong - khong dung matrix cua paper)
# =============================================================================

def compute_standard_features(Xc, yc):
    feats = {}
    X, y = balanced_subset(Xc, yc, CFG["feat_per_class"], CFG["SEED"], "minmax")

    tlog("Feature: MI relevance (sklearn, n_neighbors=%d) ..." % CFG["mi_neighbors"])
    feats["mi"] = mutual_info_classif(X, y, discrete_features=False,
                                      n_neighbors=CFG["mi_neighbors"],
                                      random_state=CFG["SEED"])

    tlog("Feature: ANOVA F relevance (sklearn.f_classif) ...")
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0)
    feats["f"] = F

    tlog("Feature: |Pearson correlation| redundancy ...")
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    feats["corr"] = C                       # giu dau (PSD) cho DPP
    feats["abs_corr"] = np.abs(C)           # redundancy cho mRMR/cluster
    return feats


def _minmax(v):
    v = np.asarray(v, dtype=np.float64)
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-12)


# =============================================================================
# BASELINE METHODS (bam sat bai bao goc)
# =============================================================================

def m_top_mi(feats, k):
    """Top-MI: chon K band co MI(band;label) cao nhat."""
    return sorted(np.argsort(feats["mi"])[::-1][:k].tolist())


def m_mrmr(feats, k):
    """mRMR FCD/MID (Peng 2005, continuous): rel=F-test (chuan hoa), red=|corr|.
       score(j) = rel[j] - mean_{i in S} |corr[j,i]|."""
    rel = _minmax(feats["f"])
    R = feats["abs_corr"]
    n = len(rel)
    selected = [int(np.argmax(rel))]
    cand = [i for i in range(n) if i != selected[0]]
    while len(selected) < k:
        red = R[np.ix_(cand, selected)].mean(axis=1)
        scores = rel[cand] - red
        b = cand[int(np.argmax(scores))]
        selected.append(b)
        cand.remove(b)
    return sorted(selected)


def m_dpp_greedy(feats, k):
    """DPP greedy MAP (Kulesza 2012; fast greedy Chen 2018).
       L = diag(q) * S * diag(q), q=quality tu relevance, S=correlation (PSD)."""
    rel = _minmax(feats["mi"])
    q = np.exp(CFG["dpp_quality_scale"] * rel)
    S = feats["corr"].copy()
    np.fill_diagonal(S, 1.0)
    L = (q[:, None] * S) * q[None, :]

    item = L.shape[0]
    cis = np.zeros((k, item))
    di2s = np.copy(np.diag(L))
    selected = [int(np.argmax(di2s))]
    while len(selected) < k:
        kk = len(selected) - 1
        j = selected[-1]
        ci_opt = cis[:kk, j]
        di_opt = math.sqrt(max(di2s[j], 1e-12))
        eis = (L[j, :] - np.dot(ci_opt, cis[:kk, :])) / di_opt
        cis[kk, :] = eis
        di2s = di2s - eis ** 2
        di2s[selected] = -np.inf
        nxt = int(np.argmax(di2s))
        if di2s[nxt] < 1e-10:
            break
        selected.append(nxt)
    return sorted(selected)


def m_cluster_exemplar(feats, k):
    """Cluster band thanh K cum theo distance (1-|corr|), exemplar = band MI cao nhat/cum."""
    D = 1.0 - feats["abs_corr"]
    np.fill_diagonal(D, 0.0)
    try:
        cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    except TypeError:  # sklearn cu dung 'affinity'
        cl = AgglomerativeClustering(n_clusters=k, affinity="precomputed", linkage="average")
    labels = cl.fit_predict(D)
    mi = feats["mi"]
    selected = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        selected.append(int(members[np.argmax(mi[members])]))
    # neu thieu (cum rong), bu bang Top-MI
    if len(selected) < k:
        for b in np.argsort(mi)[::-1]:
            if b not in selected:
                selected.append(int(b))
            if len(selected) == k:
                break
    return sorted(selected[:k])


def m_spa(Xc, yc, k):
    """SPA (Araujo 2001): successive orthogonal projections, chon band giam collinearity.
       Seed = band variance lon nhat; co the multi-start."""
    X, _ = balanced_subset(Xc, yc, CFG["spa_samples"] // len(np.unique(yc)) + 1,
                           CFG["SEED"], "std")
    n_bands = X.shape[1]
    var = np.var(X, axis=0)
    seeds = np.argsort(var)[::-1][:CFG["spa_n_starts"]]

    def run_from(seed_band):
        selected = [int(seed_band)]
        for _ in range(k - 1):
            Xs = X[:, selected]
            Q, _ = np.linalg.qr(Xs)
            proj = Q @ (Q.T @ X)
            resid = X - proj
            norms = np.linalg.norm(resid, axis=0)
            norms[selected] = -1.0
            selected.append(int(np.argmax(norms)))
        return selected

    best, best_score = None, -np.inf
    for sb in seeds:
        chain = run_from(sb)
        # tieu chi: tong norm residual cuoi cung (cang lon = it collinear)
        Xs = X[:, chain]
        Q, _ = np.linalg.qr(Xs)
        score = np.linalg.norm(X - Q @ (Q.T @ X))
        if score > best_score:
            best_score, best = score, chain
    return sorted(best)


def m_uve(Xc, yc, k):
    """UVE (Centner 1996): on dinh he so PLS qua bootstrap; chon top-K theo |mean/std|."""
    X, y = balanced_subset(Xc, yc, CFG["uve_samples"] // len(np.unique(yc)) + 1,
                           CFG["SEED"], "std")
    classes = np.unique(y)
    Yoh = np.zeros((len(y), len(classes)))
    for i, c in enumerate(classes):
        Yoh[y == c, i] = 1.0

    n_samples, n_real = X.shape
    rng = np.random.default_rng(CFG["SEED"])
    mags = []
    for r in range(CFG["uve_resamples"]):
        idx = rng.choice(n_samples, size=int(0.7 * n_samples), replace=False)
        noise = rng.normal(0, CFG["uve_noise"], size=(len(idx), n_real))
        Xa = np.hstack([X[idx], noise])
        pls = PLSRegression(n_components=CFG["uve_components"])
        pls.fit(Xa, Yoh[idx])
        coef = np.asarray(pls.coef_)
        if coef.shape[-1] == Xa.shape[1]:     # (targets, features)
            feat_coef = coef
        else:                                 # (features, targets)
            feat_coef = coef.T
        mag = np.linalg.norm(feat_coef, axis=0)   # (features_total,)
        mags.append(mag[:n_real])                 # chi giu bien that
    mags = np.array(mags)
    stability = np.abs(mags.mean(axis=0) / (mags.std(axis=0) + 1e-12))
    return sorted(np.argsort(stability)[::-1][:k].tolist())


def m_random(k, n, runs, seed):
    """Random K band, tra ve list cac lan (de tinh avg ben evaluation)."""
    out = []
    for i in range(runs):
        rng = np.random.default_rng(seed + i)
        out.append(sorted(rng.choice(n, size=k, replace=False).tolist()))
    return out


def m_ga(Xc, yc, k, feats):
    """GA wrapper (TUY CHON): fitness = SVM-RBF CV OA tren subset nho. RAT TON THOI GIAN."""
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    X, y = balanced_subset(Xc, yc, CFG["ga_eval_per_class"], CFG["SEED"], "minmax")
    rng = np.random.default_rng(CFG["SEED"])

    def fitness(bands):
        skf = StratifiedKFold(n_splits=CFG["ga_folds"], shuffle=True, random_state=CFG["SEED"])
        accs = []
        for tr, va in skf.split(X[:, bands], y):
            clf = SVC(kernel="rbf", C=10.0, gamma="scale")
            clf.fit(X[tr][:, bands], y[tr])
            accs.append(accuracy_score(y[va], clf.predict(X[va][:, bands])))
        return float(np.mean(accs))

    def rand_ind():
        return sorted(rng.choice(N, size=k, replace=False).tolist())

    pop = [rand_ind() for _ in range(CFG["ga_pop"])]
    fits = [fitness(ind) for ind in pop]
    for g in range(CFG["ga_gen"]):
        order = np.argsort(fits)[::-1]
        pop = [pop[i] for i in order]
        fits = [fits[i] for i in order]
        tlog(f"  [GA] gen {g+1}/{CFG['ga_gen']} best OA={fits[0]*100:.2f}%")
        elite = pop[:max(2, CFG["ga_pop"] // 5)]
        new = list(elite)
        while len(new) < CFG["ga_pop"]:
            p1, p2 = rng.choice(len(elite), size=2)
            union = list(set(pop[p1]) | set(pop[p2]))
            child = sorted(rng.choice(union, size=min(k, len(union)), replace=False).tolist())
            while len(child) < k:
                b = int(rng.integers(0, N))
                if b not in child:
                    child.append(b)
            # mutation
            if rng.random() < 0.3:
                child[int(rng.integers(0, k))] = int(rng.integers(0, N))
                child = sorted(set(child))
                while len(child) < k:
                    b = int(rng.integers(0, N))
                    if b not in child:
                        child.append(b)
            new.append(sorted(child))
        pop = new
        fits = [fitness(ind) for ind in pop]
    best = pop[int(np.argmax(fits))]
    return sorted(best)


def load_prebuilt_bands():
    """Doc band CO SAN cho GA/SPA/UVE (da chay rieng theo bai bao goc).
       Ho tro key: band_indices / bands / selected_bands."""
    found = {}
    for name, fname in CFG["prebuilt"].items():
        path = None
        for d in CFG["prebuilt_dirs"]:
            p = os.path.join(d, fname)
            if os.path.exists(p):
                path = p
                break
        if path is None:                       # fallback: tu do de quy /kaggle/input
            hits = _glob.glob(f"/kaggle/input/**/{fname}", recursive=True)
            if hits:
                path = hits[0]
        if path is None:
            print(f"[WARN] khong thay band co san cho {name} ({fname}) -> bo qua")
            continue
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        idx = j.get("band_indices") or j.get("bands") or j.get("selected_bands")
        found[name] = sorted(int(b) for b in idx)
        print(f"  [prebuilt] {name:5s} <- {os.path.basename(path)}  ({len(found[name])} bands)")
    return found


# =============================================================================
# RUN
# =============================================================================

def main():
    set_seeds(CFG["SEED"])
    tlog("=== BASELINE BAND SELECTION (reproducible) ===")
    wavelengths = np.load(PATHS["wavelengths"])
    Xc, yc = load_clean(PATHS["reflectance"], PATHS["labels"])

    feats = compute_standard_features(Xc, yc)

    results = {}
    methods = list(CFG["methods"])
    if CFG["run_GA"] and "GA" not in methods:
        methods.append("GA")

    for name in methods:
        t0 = time.perf_counter()
        if name == "Top-MI":
            bands = m_top_mi(feats, K)
        elif name == "mRMR":
            bands = m_mrmr(feats, K)
        elif name == "DPP-greedy":
            bands = m_dpp_greedy(feats, K)
        elif name == "Cluster-Exemplar":
            bands = m_cluster_exemplar(feats, K)
        elif name == "SPA":
            bands = m_spa(Xc, yc, K)
        elif name == "UVE":
            bands = m_uve(Xc, yc, K)
        elif name == "Random":
            runs = m_random(K, N, CFG["random_runs"], CFG["SEED"])
            results["Random_runs"] = runs
            bands = runs[0]
        elif name == "GA":
            bands = m_ga(Xc, yc, K, feats)
        else:
            continue
        results[name] = bands
        tlog(f"{name:18s} -> {len(bands)} bands  ({time.perf_counter()-t0:.1f}s)")

    # GA/SPA/UVE: dung band CO SAN (khong chay lai) tru khi da tinh o tren
    for name, bands in load_prebuilt_bands().items():
        if name not in results:
            results[name] = bands

    # luu
    out = {
        "K": K, "N": N,
        "feature_note": "MI=sklearn knn(n=3); F=ANOVA; redundancy/sim=|Pearson corr|; "
                        "KHONG dung matrix cua paper VNAL-PA.",
        "wavelengths_nm": wavelengths.tolist(),
        "bands": {m: results[m] for m in results if m != "Random_runs"},
        "wavelengths_selected_nm": {
            m: [float(wavelengths[b]) for b in results[m]]
            for m in results if m != "Random_runs"
        },
    }
    if "Random_runs" in results:
        out["random_runs"] = results["Random_runs"]
    with open(os.path.join(OUTPUT_DIR, "baseline_bands.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    rows = [{"method": m, "n_bands": len(results[m]), "bands": json.dumps(results[m])}
            for m in results if m != "Random_runs"]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "baseline_bands.csv"), index=False)

    del Xc, yc
    gc.collect()
    tlog(f"DONE. Output -> {OUTPUT_DIR}")
    for m in results:
        if m != "Random_runs":
            print(f"  {m:18s}: {results[m]}")


if __name__ == "__main__":
    main()
