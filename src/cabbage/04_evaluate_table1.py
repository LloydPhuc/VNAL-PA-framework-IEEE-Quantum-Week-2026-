"""
=============================================================================
 EVALUATE TABLE I  -  Combined SVM evaluation (Shuffle + Spatial split)
=============================================================================
 Gop band cua TAT CA phuong phap (baselines + VNAL-PA) vao 1 file danh gia
 chung de tai dung Bang I:
   - Doc band tu:  baseline_results/baseline_bands.json   (cac baseline)
                   vnal_pa_results/vnal_pa_summary.json    (VNAL-PA)
   - Them All-bands(260) va Random.
   - Danh gia SVM-RBF (C=10, gamma=scale) bang:
       * Shuffle Split : StratifiedKFold (5-fold) tren subset can bang lop.
       * Spatial Split : GroupKFold theo o khong gian (tranh ro ri khong gian).
   - Xuat: table1_results.csv  +  table1_grouped_bar.png

 LUU Y khop paper:
   - 9000 px (3000/class), K=64, 5-fold  (chinh trong CFG).
   - Spatial split: anh duoc chia thanh luoi o; cac pixel cung o khong bao gio
     vua o train vua o test -> phan anh dung "spatial generalization".
=============================================================================
"""

import os
import gc
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score

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

# Noi chua band da chon (output cua 02_vnal_pa.py va 03_baselines)
if os.path.isdir("/kaggle/input"):
    RESULTS = "/kaggle/working/CABBAGE/results"
else:
    RESULTS = r"D:\HYPERSPECTRAL 1\FULL_PIPELINE\CABBAGE\results"
BASELINE_JSON = os.path.join(RESULTS, "baseline_bands.json")
VNAL_JSON     = os.path.join(RESULTS, "vnal_pa_summary.json")
OUTPUT_DIR    = RESULTS
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "K": 64,
    "SEED": 42,
    "per_class": 3000,        # 3000*3 = 9000 px (giong paper)
    "folds": 5,
    "svm_C": 10.0,
    "svm_gamma": "scale",
    "spatial_grid": 8,        # chia anh thanh 8x8 = 64 o khong gian
    "random_runs": 5,
    # thu tu hien thi tren bang/figure
    "method_order": ["All-bands (260)", "GA", "UVE", "SPA", "Cluster-Exemplar",
                     "mRMR", "DPP-greedy", "Top-MI", "Random", "VNAL-PA"],
}
K = CFG["K"]


# =============================================================================
# DATA  (giu ca toa do de spatial split)
# =============================================================================

def load_clean_with_coords(path_X, path_y):
    print("[Data] Loading ...")
    X = np.load(path_X, mmap_mode="r")
    y = np.load(path_y)
    rows, cols = np.where(y > 0)
    Xc = X[rows, cols, :].astype(np.float32)
    yc = (y[rows, cols] - int(y[rows, cols].min())).astype(np.int16)
    coords = np.stack([rows, cols], axis=1).astype(np.int32)
    del X, y
    gc.collect()
    print(f"   -> {Xc.shape[0]} px, {Xc.shape[1]} bands, {len(np.unique(yc))} classes")
    return Xc, yc, coords


def balanced_idx(yc, per_class, seed):
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(yc):
        ci = np.where(yc == c)[0]
        rep = len(ci) < per_class
        idx.extend(rng.choice(ci, size=per_class, replace=rep))
    return np.array(idx)


def spatial_block_ids(coords, grid):
    """Gan moi pixel vao 1 o luoi grid x grid -> group id cho GroupKFold."""
    r, c = coords[:, 0], coords[:, 1]
    rb = np.floor((r - r.min()) / (r.max() - r.min() + 1e-9) * grid).astype(int)
    cb = np.floor((c - c.min()) / (c.max() - c.min() + 1e-9) * grid).astype(int)
    rb = np.clip(rb, 0, grid - 1)
    cb = np.clip(cb, 0, grid - 1)
    return rb * grid + cb


# =============================================================================
# EVALUATION
# =============================================================================

def eval_shuffle(Xc, yc, bands, seed):
    idx = balanced_idx(yc, CFG["per_class"], seed)
    X = MinMaxScaler().fit_transform(Xc[idx][:, bands].astype(np.float64))
    y = yc[idx]
    skf = StratifiedKFold(n_splits=CFG["folds"], shuffle=True, random_state=seed)
    oas, f1s = [], []
    for tr, va in skf.split(X, y):
        clf = SVC(kernel="rbf", C=CFG["svm_C"], gamma=CFG["svm_gamma"], random_state=seed)
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[va])
        oas.append(accuracy_score(y[va], yp))
        f1s.append(f1_score(y[va], yp, average="macro"))
    return float(np.mean(oas)), float(np.std(oas)), float(np.mean(f1s))


def eval_spatial(Xc, yc, coords, bands, seed):
    idx = balanced_idx(yc, CFG["per_class"], seed)
    X = MinMaxScaler().fit_transform(Xc[idx][:, bands].astype(np.float64))
    y = yc[idx]
    groups = spatial_block_ids(coords[idx], CFG["spatial_grid"])
    n_groups = len(np.unique(groups))
    folds = min(CFG["folds"], n_groups)
    gkf = GroupKFold(n_splits=folds)
    oas, f1s = [], []
    for tr, va in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < len(np.unique(y)):
            continue
        clf = SVC(kernel="rbf", C=CFG["svm_C"], gamma=CFG["svm_gamma"], random_state=seed)
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[va])
        oas.append(accuracy_score(y[va], yp))
        f1s.append(f1_score(y[va], yp, average="macro"))
    if not oas:
        return None, None, None
    return float(np.mean(oas)), float(np.std(oas)), float(np.mean(f1s))


# =============================================================================
# LOAD BANDS
# =============================================================================

def load_all_bands():
    methods = {}
    random_runs = None
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, encoding="utf-8") as f:
            bj = json.load(f)
        for m, b in bj.get("bands", {}).items():
            methods[m] = b
        random_runs = bj.get("random_runs")
    else:
        print(f"[WARN] khong thay {BASELINE_JSON}")

    if os.path.exists(VNAL_JSON):
        with open(VNAL_JSON, encoding="utf-8") as f:
            vj = json.load(f)
        methods["VNAL-PA"] = vj["selected_bands"]
    else:
        print(f"[WARN] khong thay {VNAL_JSON}")

    return methods, random_runs


# =============================================================================
# MAIN
# =============================================================================

def main():
    np.random.seed(CFG["SEED"])
    Xc, yc, coords = load_clean_with_coords(PATHS["reflectance"], PATHS["labels"])
    N = Xc.shape[1]

    methods, random_runs = load_all_bands()
    methods["All-bands (260)"] = list(range(N))

    rows = []

    def run_one(name, bands):
        t0 = time.perf_counter()
        s_oa, s_std, s_f1 = eval_shuffle(Xc, yc, bands, CFG["SEED"])
        sp_oa, sp_std, sp_f1 = eval_spatial(Xc, yc, coords, bands, CFG["SEED"])
        rows.append({
            "method": name, "n_bands": len(bands),
            "shuffle_OA": s_oa, "shuffle_OA_std": s_std, "shuffle_F1": s_f1,
            "spatial_OA": sp_oa, "spatial_OA_std": sp_std, "spatial_F1": sp_f1,
            "bands": json.dumps(sorted(bands)),
        })
        sp = f"{sp_oa*100:5.2f}" if sp_oa is not None else "  -- "
        print(f"  {name:22s} shuffle OA={s_oa*100:5.2f}%  spatial OA={sp}%  "
              f"({time.perf_counter()-t0:.1f}s)")

    # cac phuong phap don
    for name, bands in methods.items():
        if name == "Random":
            continue
        run_one(name, bands)

    # Random: trung binh nhieu lan
    if random_runs:
        accs = {"s_oa": [], "s_f1": [], "sp_oa": [], "sp_f1": []}
        for i, bands in enumerate(random_runs[:CFG["random_runs"]]):
            s_oa, _, s_f1 = eval_shuffle(Xc, yc, bands, CFG["SEED"] + i)
            sp_oa, _, sp_f1 = eval_spatial(Xc, yc, coords, bands, CFG["SEED"] + i)
            accs["s_oa"].append(s_oa); accs["s_f1"].append(s_f1)
            if sp_oa is not None:
                accs["sp_oa"].append(sp_oa); accs["sp_f1"].append(sp_f1)
        rows.append({
            "method": f"Random (avg {len(random_runs)})", "n_bands": K,
            "shuffle_OA": np.mean(accs["s_oa"]), "shuffle_OA_std": np.std(accs["s_oa"]),
            "shuffle_F1": np.mean(accs["s_f1"]),
            "spatial_OA": np.mean(accs["sp_oa"]) if accs["sp_oa"] else None,
            "spatial_OA_std": np.std(accs["sp_oa"]) if accs["sp_oa"] else None,
            "spatial_F1": np.mean(accs["sp_f1"]) if accs["sp_f1"] else None,
            "bands": json.dumps(random_runs[:CFG["random_runs"]]),  # list cac lan
        })
        print(f"  {'Random (avg)':22s} shuffle OA={np.mean(accs['s_oa'])*100:5.2f}%")

    df = pd.DataFrame(rows).sort_values("shuffle_OA", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "table1_results.csv"), index=False)
    print("\n", df.drop(columns=["bands"]).to_string(index=False))

    # ---- file GOP band cua TAT CA phuong phap (1 cho duy nhat) ----
    wl = np.load(PATHS["wavelengths"])
    all_bands = {}
    for r in rows:
        if r["method"].startswith("Random"):
            all_bands[r["method"]] = {"runs": random_runs[:CFG["random_runs"]]}
        else:
            b = json.loads(r["bands"])
            all_bands[r["method"]] = {
                "bands": b,
                "wavelengths_nm": [float(wl[i]) for i in b],
                "n_bands": len(b),
            }
    with open(os.path.join(OUTPUT_DIR, "all_methods_bands.json"), "w", encoding="utf-8") as f:
        json.dump(all_bands, f, indent=2, ensure_ascii=False)
    print(f"  [bands] da gop tat ca phuong phap -> all_methods_bands.json")

    plot_grouped_bar(df)
    del Xc, yc, coords
    gc.collect()
    print("\nDONE ->", OUTPUT_DIR)


def plot_grouped_bar(df):
    """Grouped bar: OA shuffle vs spatial moi phuong phap (figure-designer: experimental-results)."""
    d = df.dropna(subset=["shuffle_OA"]).copy()
    d = d.sort_values("shuffle_OA")
    x = np.arange(len(d))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.barh(x - w/2, d["shuffle_OA"] * 100, height=w, color="#4C72B0", label="Shuffle split")
    sp = d["spatial_OA"].fillna(0) * 100
    b2 = ax.barh(x + w/2, sp, height=w, color="#DD8452", label="Spatial split")
    ax.set_yticks(x); ax.set_yticklabels(d["method"])
    ax.set_xlabel("Overall Accuracy (%)")
    ax.set_title("SVM-RBF classification accuracy by band-selection method (K=64)")
    ax.legend(loc="lower right")
    # highlight VNAL-PA
    for i, m in enumerate(d["method"]):
        if "VNAL" in m:
            ax.get_yticklabels()[i].set_color("crimson")
            ax.get_yticklabels()[i].set_fontweight("bold")
    lo = max(0, d[["shuffle_OA", "spatial_OA"]].min().min() * 100 - 3)
    ax.set_xlim(lo, max(d["shuffle_OA"].max(), d["spatial_OA"].max()) * 100 + 3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "table1_grouped_bar.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "table1_grouped_bar.pdf"), bbox_inches="tight")  # vector
    plt.close()


if __name__ == "__main__":
    main()
