"""
=============================================================================
 02 - VNAL-PA  (Variance-Normalized Annealing Landscape Penalty)
=============================================================================
 BAN GOC THEO PAPER (da chot):
   - w* chon theo NANG LUONG: w* = argmin E_norm  (Eq.6, unsupervised) -> KHONG
     dung accuracy.
   - KHONG co coverage repulsion (bo hoan toan).
   - Redundancy = Ledoit-Wolf (doc tu cache/redundancy_lw.npy, do 01 tao ra).
   - Relevance  = KSG-100 MI       (doc tu cache/relevance.npy).

 CAC BUOC:
   (A) Monte-Carlo uoc luong sigma_Rel, sigma_Red tren subset K-band   [Eq.4]
   (B) Quet w: E_norm = -(1-w) Rel_hat.x + w x'Red_hat x               [Eq.5]
   (C) w* = argmin_w E_norm (trong vung feasible VR>=0.95)             [Eq.6]
   (D) Binary search lambda* = min{lambda : VR>=0.95}                  [Eq.8]
   (E) Giai QUBO cuoi E_VNAL-PA -> chon K band                        [Eq.9]

 OUTPUT: results/vnal_pa_summary.json , results/selected_bands.png/.pdf
 (Danh gia SVM nam o 04_evaluate_table1.py)
=============================================================================
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import neal
    NEAL_OK = True
except ImportError:
    NEAL_OK = False


def resolve():
    import glob
    base = r"D:\HYPERSPECTRAL 1"
    if os.path.isdir("/kaggle/input"):
        ROOT = "/kaggle/working/FULL_PIPELINE"
        hits = glob.glob("/kaggle/input/**/wavelengths_260bands.npy", recursive=True)
        WL = hits[0] if hits else "/kaggle/input/wavelengths_260bands.npy"
    else:
        ROOT = os.path.join(base, "FULL_PIPELINE")
        WL = os.path.join(base, "INPUT", "RAW DATA", "wavelengths_260bands.npy")
    return ROOT, WL


CFG = {
    "K": 64, "SEED": 42,
    "mc_samples": 5000, "mc_seed": 42,
    "w_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "vr_thresh": 0.95,
    "sa_seeds": [42, 123, 456],
    "sa_reads_grid": 200, "sa_sweeps_grid": 500,
    "sa_reads_final": 200, "sa_sweeps_final": 2500,
    "lambda_bs_steps": 8,
}
K = CFG["K"]


def empirical_energy_stats(rel, R, k, n_samples, seed):
    rng = np.random.default_rng(seed)
    B = len(rel)
    rels = np.empty(n_samples); reds = np.empty(n_samples)
    for s in range(n_samples):
        sel = rng.choice(B, size=k, replace=False)
        rels[s] = -np.sum(rel[sel])
        sub = R[np.ix_(sel, sel)]
        reds[s] = (np.sum(sub) - np.trace(sub)) / 2.0
    return max(float(np.std(rels)), 1e-10), max(float(np.std(reds)), 1e-10)


def build_qubo(rel, R, k, w, sig_rel, sig_red, lam):
    beta_n = (1.0 - w) / sig_rel
    alpha_n = w / sig_red
    B = len(rel)
    Q = {}
    for i in range(B):
        Q[(i, i)] = -beta_n * rel[i] + lam * (1 - 2 * k)
        row = R[i]
        for j in range(i + 1, B):
            Q[(i, j)] = alpha_n * row[j] + 2.0 * lam
    return Q, alpha_n, beta_n


def solve_qubo_sa(Q, k, num_reads, num_sweeps, seed):
    ss = neal.SimulatedAnnealingSampler().sample_qubo(
        Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    best, best_e = None, float("inf")
    valid, total = 0, 0
    for d in ss.data(["sample", "energy", "num_occurrences"]):
        occ = int(d.num_occurrences); total += occ
        sel = [b for b, v in d.sample.items() if v == 1]
        if len(sel) == k:
            valid += occ
            if d.energy < best_e:
                best_e = float(d.energy); best = d.sample
    return {
        "bands": sorted([b for b, v in best.items() if v == 1]) if best else [],
        "energy": best_e if best else None,
        "vr": valid / max(1, total),
    }


def objective_energy(rel, R, bands, alpha_n, beta_n):
    bands = list(bands)
    sub = R[np.ix_(bands, bands)]
    return float(-beta_n * np.sum(rel[bands]) + alpha_n * (np.sum(sub) - np.trace(sub)) / 2.0)


def lambda_universal(rel, R, sig_rel, sig_red):
    beta_max = 1.0 / sig_rel
    alpha_max = 1.0 / sig_red
    gain_add = beta_max * np.max(rel)
    gain_rem = beta_max * np.max(rel) + alpha_max * np.max(np.sum(R, axis=1))
    return float(max(gain_add, gain_rem) * 1.1)


def binary_search_lambda(rel, R, k, w, sig_rel, sig_red, lam_hi):
    lo, hi, opt = 0.0, lam_hi, lam_hi
    for step in range(CFG["lambda_bs_steps"]):
        mid = (lo + hi) / 2.0
        Q, _, _ = build_qubo(rel, R, k, w, sig_rel, sig_red, mid)
        vr = float(np.mean([solve_qubo_sa(Q, k, CFG["sa_reads_grid"], CFG["sa_sweeps_grid"], s)["vr"]
                            for s in CFG["sa_seeds"]]))
        print(f"    [BS] step {step+1}: lam={mid:.4f}  VR={vr*100:.1f}%")
        if vr >= CFG["vr_thresh"]:
            hi, opt = mid, mid
        else:
            lo = mid
    return opt


def main():
    if not NEAL_OK:
        print("[FATAL] can dwave-neal: pip install dwave-neal")
        return
    ROOT, WL = resolve()
    CACHE = os.path.join(ROOT, "cache")
    RESULTS = os.path.join(ROOT, "results")
    os.makedirs(RESULTS, exist_ok=True)

    rel = np.load(os.path.join(CACHE, "relevance.npy"))
    R = np.load(os.path.join(CACHE, "redundancy_lw.npy"))
    np.fill_diagonal(R, 0.0)
    wavelengths = np.load(WL)

    print("=" * 64)
    print(f"  VNAL-PA  N={len(rel)}  K={K}  (w*=energy, no-coverage, LW-redundancy)")
    print("=" * 64)

    sig_rel, sig_red = empirical_energy_stats(rel, R, K, CFG["mc_samples"], CFG["mc_seed"])
    lam_univ = lambda_universal(rel, R, sig_rel, sig_red)
    print(f"[stats] sigma_rel={sig_rel:.4f} sigma_red={sig_red:.4f} | lambda_univ={lam_univ:.4f}")

    # ---- Phase 1: grid w (chon theo E_norm) ----
    print("\n[Phase 1] grid w (criterion = min E_norm) ...")
    rows = []
    for w in CFG["w_grid"]:
        Q, alpha_n, beta_n = build_qubo(rel, R, K, w, sig_rel, sig_red, lam_univ)
        for s in CFG["sa_seeds"]:
            res = solve_qubo_sa(Q, K, CFG["sa_reads_grid"], CFG["sa_sweeps_grid"], s)
            e = objective_energy(rel, R, res["bands"], alpha_n, beta_n) if res["bands"] else None
            rows.append({"w": w, "seed": s, "energy": e, "vr": res["vr"]})
        vr_m = np.mean([r["vr"] for r in rows if r["w"] == w])
        print(f"   w={w:.1f}  VR={vr_m*100:5.1f}%")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "vnal_phase1_w_grid.csv"), index=False)

    agg = df.groupby("w").agg(vr=("vr", "mean"), energy=("energy", "mean")).reset_index()
    feasible = agg[agg["vr"] >= CFG["vr_thresh"]]
    pool = feasible if not feasible.empty else agg
    w_star = float(pool.loc[pool["energy"].idxmin(), "w"])
    print(f"\n[w*] = {w_star}  (argmin E_norm, Eq.6)")

    # ---- Phase 2: binary search lambda* ----
    print(f"\n[Phase 2] binary search lambda* @ w*={w_star} ...")
    lam_star = binary_search_lambda(rel, R, K, w_star, sig_rel, sig_red, lam_univ)
    print(f"[lambda*] = {lam_star:.4f}")

    # ---- Phase 3: final solve ----
    print("\n[Phase 3] final solve ...")
    Q, alpha_n, beta_n = build_qubo(rel, R, K, w_star, sig_rel, sig_red, lam_star)
    best = {"bands": [], "energy": float("inf")}
    vrs = []
    for s in CFG["sa_seeds"]:
        res = solve_qubo_sa(Q, K, CFG["sa_reads_final"], CFG["sa_sweeps_final"], s)
        vrs.append(res["vr"])
        if res["bands"] and res["energy"] < best["energy"]:
            best = res
    bands = best["bands"]
    print(f"[VNAL-PA] {len(bands)} bands, VR={np.mean(vrs)*100:.1f}%")

    summary = {
        "K": K, "N": len(rel),
        "w_star": w_star, "lambda_star": lam_star, "lambda_universal": lam_univ,
        "sigma_rel": sig_rel, "sigma_red": sig_red,
        "redundancy": "Ledoit-Wolf shrinkage |corr|", "coverage_repulsion": False,
        "w_star_criterion": "min E_norm (Eq.6, unsupervised)",
        "selected_bands": bands,
        "selected_wavelengths_nm": [float(wavelengths[b]) for b in bands],
    }
    with open(os.path.join(RESULTS, "vnal_pa_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- plot ----
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(wavelengths, rel, color="navy", lw=1)
    ax[0].scatter(wavelengths[bands], rel[bands], color="crimson", zorder=5, label="VNAL-PA")
    ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Relevance MI (bits)")
    ax[0].set_title("Relevance spectrum + selected bands"); ax[0].legend()
    ax[1].vlines(wavelengths[bands], 0, 1, color="crimson")
    ax[1].set_xlim(wavelengths.min() - 10, wavelengths.max() + 10)
    ax[1].set_yticks([]); ax[1].set_xlabel("Wavelength (nm)")
    ax[1].set_title(f"Distribution of {len(bands)} selected bands")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "selected_bands.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(RESULTS, "selected_bands.pdf"), bbox_inches="tight")
    plt.close()
    print("\nDONE ->", RESULTS)


if __name__ == "__main__":
    main()
