"""
=============================================================================
 EXPERIMENT B : UNIVERSAL LAMBDA co thuc su cho phep "doi band" de dang?
=============================================================================
 GIA THUYET (van de vat ly cua penalty fixed-K):
   QUBO:  E = E_norm(x,w) + lambda*(sum(x) - K)^2
   - Tai trang thai hop le sum(x)=K  ->  penalty = 0.
   - Mot lan SWAP (1 band vao, 1 band ra) bang single-bit-flip BUOC phai di qua
     trang thai trung gian sum(x)=K±1, noi penalty = lambda*1^2 = lambda.
   => RAO NANG LUONG cua moi swap ~ lambda.
   - Xac suat Boltzmann vuot rao o nhiet do T:  P_swap(T) ~ exp(-lambda / T).
   - Khi T ha thap, neu lambda LON (vd lambda_universal) thi exp(-lambda/T) -> 0
     => SWAP bi DONG BANG => KET CUC BO.

 lambda_universal (cong thuc trong paper) duoc thiet ke de DAM BAO feasibility
 cho MOI w (lay max-case) nen no CO TINH BAO THU/LON -> chinh no gay kho doi band.
 => Day la LY DO can binary-search lambda* (nho nhat van giu VR>=0.95).

 THI NGHIEM:
   B1 (giai tich) : rao = lambda ; ve P_swap(T)=exp(-lambda/T) cho lambda* vs
                    lambda_universal doc theo lich nhiet do annealing.
   B2 (thuc nghiem): SA single-flip CO DO DAC, quet lambda. Do:
                    - VR (ty le sweep cuoi con feasible)
                    - chat luong nghiem E_norm (thap = tot)
                    - so SWAP thuc hien o pha nhiet do thap (mixing)
                    - ty le chap nhan buoc len doc (uphill) o pha lanh
   KY VONG: lambda qua nho -> mat feasibility; lambda vua (~lambda*) -> feasible
            VA nhieu swap -> nghiem tot; lambda qua lon (~universal) -> feasible
            nhung dong bang -> nghiem te hon. => lambda* la diem vang.

 OUTPUT: exp_lambda_barrier.png/.pdf , exp_lambda_sweep.png/.pdf ,
         exp_lambda_analysis.csv/.json
=============================================================================
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths: doc tu cache + summary cua VNAL ----
ROOT = "/kaggle/working/CABBAGE" if os.path.isdir("/kaggle/input") else r"D:\HYPERSPECTRAL 1\FULL_PIPELINE\CABBAGE"
CACHE = os.path.join(ROOT, "cache")
P_REL = os.path.join(CACHE, "relevance.npy")
P_R   = os.path.join(CACHE, "redundancy_lw.npy")
VNAL_JSON = os.path.join(ROOT, "results", "vnal_pa_summary.json")
OUTPUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {
    "K": 64,
    "SEED": 42,
    "mc_samples": 5000,
    "w": None,                 # None -> lay tu vnal_summary, else 0.5
    "n_sweeps": 1200,
    "seeds": [42, 123, 456],
    "track_frac": 0.7,         # pha "lanh" = 30% sweep cuoi
    "T0_factor": 1.0,          # T0 = T0_factor * max|deltaE| ban dau
    "T1_factor": 1e-3,         # T1 = T1_factor * T0
    # he so lambda* de quet (neu khong co lambda* se dung lambda_universal lam moc)
    "lambda_multipliers_of_star": [0.25, 0.5, 1.0, 2.0],
}
K = CFG["K"]


# ---------- core (tu VNAL) ----------
def empirical_stats(rel, R, k, M, seed):
    rng = np.random.default_rng(seed)
    B = len(rel)
    rels = np.empty(M); reds = np.empty(M)
    for s in range(M):
        sel = rng.choice(B, size=k, replace=False)
        rels[s] = -np.sum(rel[sel])
        sub = R[np.ix_(sel, sel)]
        reds[s] = (np.sum(sub) - np.trace(sub)) / 2.0
    return max(rels.std(), 1e-10), max(reds.std(), 1e-10)


def lambda_universal(rel, R, sig_rel, sig_red):
    beta_max = 1.0 / sig_rel
    alpha_max = 1.0 / sig_red
    gain_add = beta_max * np.max(rel)
    gain_rem = beta_max * np.max(rel) + alpha_max * np.max(np.sum(R, axis=1))
    return float(max(gain_add, gain_rem) * 1.1)


def build_hW(rel, R, k, w, sig_rel, sig_red, lam):
    """h (linear) va W (symmetric, diag 0) sao cho
       E(x) = h.x + 0.5 x'Wx = E_norm + lambda(sum-K)^2 - lambda*K^2."""
    beta_n = (1.0 - w) / sig_rel
    alpha_n = w / sig_red
    h = -beta_n * rel + lam * (1 - 2 * k)
    W = alpha_n * R + 2.0 * lam
    np.fill_diagonal(W, 0.0)
    return h, W, alpha_n, beta_n


# ---------- instrumented single-flip SA ----------
def run_sa(h, W, k, n_sweeps, seed, track_frac, T0, T1):
    """T0/T1 truyen tu NGOAI va co dinh theo thang do DATA-term (doc lap lambda)
       -> day moi la cach bộc lộ dung bay: khi lambda >> data, KHONG ton tai cua so
       nhiet do nao vua toi uu data vua cho phep swap."""
    rng = np.random.default_rng(seed)
    N = len(h)
    x = np.zeros(N)
    x[rng.choice(N, size=k, replace=False)] = 1.0
    fields = h + W @ x                       # field_i = h_i + sum_j W_ij x_j
    E = float(h @ x + 0.5 * x @ (W @ x))     # = E_norm + penalty - lam*K^2
    Ts = T0 * (T1 / T0) ** np.linspace(0, 1, n_sweeps)

    swaps_total = 0                          # dem swap TOAN BO anneal (khong chi pha lanh)
    swaps_late = 0
    uphill_att = 0
    uphill_acc = 0
    feas_sweeps_late = 0
    late_sweeps = 0
    visited = set()
    last_feasible = tuple(sorted(np.where(x == 1)[0]))
    track_start = int(track_frac * n_sweeps)
    T_last_swap = float(Ts[0])               # freeze-out: nhiet do cua swap cuoi cung
    bestE = E if x.sum() == k else float("inf")
    best_x = x.copy()

    for sidx, T in enumerate(Ts):
        late = sidx >= track_start
        if late:
            late_sweeps += 1
        order = rng.permutation(N)
        for kk in order:
            dE = (1.0 - 2.0 * x[kk]) * fields[kk]
            if dE <= 0:
                acc = True
            else:
                if late:
                    uphill_att += 1
                acc = rng.random() < math.exp(-dE / max(T, 1e-12))
                if acc and late:
                    uphill_acc += 1
            if acc:
                s = 1.0 - 2.0 * x[kk]         # +1 neu 0->1, -1 neu 1->0
                x[kk] = 1.0 - x[kk]
                fields += s * W[:, kk]
                E += dE
                if x.sum() == k:
                    if E < bestE:            # luu nghiem feasible tot nhat tung gap
                        bestE = E
                        best_x = x.copy()
                    fs = tuple(sorted(np.where(x == 1)[0]))
                    if fs != last_feasible:
                        last_feasible = fs
                        swaps_total += 1
                        T_last_swap = float(T)
                        if late:
                            swaps_late += 1
                            visited.add(fs)
        if late and x.sum() == k:
            feas_sweeps_late += 1

    return {
        "final_feasible": int(x.sum() == k),
        "swaps_total": swaps_total,
        "swaps_late": swaps_late,
        "T_last_swap": T_last_swap,
        "uphill_accept_rate_late": (uphill_acc / uphill_att) if uphill_att else 0.0,
        "feasible_ratio_late": (feas_sweeps_late / late_sweeps) if late_sweeps else 0.0,
        "n_distinct_feasible_late": len(visited),
        "T0": T0, "T1": T1,
        "best_x": best_x.copy(),             # dung de tinh E_norm tot nhat
        "x": x.copy(),
    }


def objective_Enorm(rel, R, bands, alpha_n, beta_n):
    bands = list(bands)
    sub = R[np.ix_(bands, bands)]
    return float(-beta_n * np.sum(rel[bands]) + alpha_n * (np.sum(sub) - np.trace(sub)) / 2.0)


def main():
    rel = np.load(P_REL).astype(np.float64)        # KSG-100 (da clip >=0 o buoc 01)
    R = np.load(P_R).astype(np.float64)            # Ledoit-Wolf |corr| (da [0,1])
    np.fill_diagonal(R, 0.0)

    sig_rel, sig_red = empirical_stats(rel, R, K, CFG["mc_samples"], CFG["SEED"])
    lam_univ = lambda_universal(rel, R, sig_rel, sig_red)

    # w* va lambda* (uu tien tu vnal_summary)
    w = CFG["w"]
    lam_star = None
    if os.path.exists(VNAL_JSON):
        with open(VNAL_JSON, encoding="utf-8") as f:
            vj = json.load(f)
        w = vj.get("w_star", w)
        lam_star = vj.get("lambda_star", None)
    if w is None:
        w = 0.5
    if lam_star is None:
        lam_star = 0.15 * lam_univ           # uoc luong neu chua co (chi de quet)
    print(f"[setup] w={w}  sigma_rel={sig_rel:.3f}  sigma_red={sig_red:.3f}")
    print(f"[setup] lambda* = {lam_star:.3f}   lambda_universal = {lam_univ:.3f}   "
          f"(ti le {lam_univ/lam_star:.1f}x)")

    # ===== lich nhiet do CO DINH theo thang do DATA-term (KHONG chua lambda) =====
    # day la mau chot: T0/T1 phai theo data scale de bay lamb-trapping lo ra.
    beta_n = (1.0 - w) / sig_rel
    alpha_n = w / sig_red
    h_data = -beta_n * rel
    W_data = alpha_n * R.copy()
    np.fill_diagonal(W_data, 0.0)
    rng = np.random.default_rng(CFG["SEED"])
    x0 = np.zeros(len(rel)); x0[rng.choice(len(rel), K, replace=False)] = 1
    f_data = h_data + W_data @ x0
    T0 = CFG["T0_factor"] * float(np.max(np.abs((1 - 2 * x0) * f_data)))
    T1 = CFG["T1_factor"] * T0
    print(f"[schedule] T0(data)={T0:.4f}  T1={T1:.6f}  "
          f"(lambda*/T0={lam_star/T0:.2f}, lambda_univ/T0={lam_univ/T0:.2f})")

    # ============== B1: barrier / Boltzmann ==============
    Tgrid = T0 * (T1 / T0) ** np.linspace(0, 1, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lam, name, col in [(lam_star, r"$\lambda^*$ (binary search)", "#4C72B0"),
                           (lam_univ, r"$\lambda_{universal}$ (paper bound)", "#C44E52")]:
        P = np.exp(-lam / np.maximum(Tgrid, 1e-12))
        ax.plot(Tgrid, P, color=col, lw=2, label=f"{name}, barrier={lam:.1f}")
    ax.axhline(1e-3, ls=":", color="gray")
    ax.text(Tgrid.min(), 1.3e-3, "swap practically frozen (P<1e-3)", fontsize=8, color="gray")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Annealing temperature T (high -> low)")
    ax.set_ylabel(r"Swap acceptance $P_{swap}=e^{-\lambda/T}$")
    ax.set_title("B1: Larger $\\lambda$ freezes band-swaps earlier in the anneal")
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_lambda_barrier.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_lambda_barrier.pdf"), bbox_inches="tight")
    plt.close()

    # ============== B2: lambda sweep (instrumented SA) ==============
    lam_grid = sorted(set(
        [m * lam_star for m in CFG["lambda_multipliers_of_star"]] + [lam_univ, 2 * lam_univ]
    ))
    records = []
    for lam in lam_grid:
        h, W, alpha_n, beta_n = build_hW(rel, R, K, w, sig_rel, sig_red, lam)
        lamK2 = lam * K * K
        per = []
        for sd in CFG["seeds"]:
            r = run_sa(h, W, K, CFG["n_sweeps"], sd, CFG["track_frac"], T0, T1)
            bands = np.where(r["best_x"] == 1)[0]   # nghiem feasible TOT NHAT
            obj = objective_Enorm(rel, R, bands, alpha_n, beta_n) if len(bands) == K else None
            per.append({**r, "obj": obj})
        objs = [p["obj"] for p in per if p["obj"] is not None]
        rec = {
            "lambda": lam,
            "lambda_over_star": lam / lam_star,
            "is_universal": abs(lam - lam_univ) < 1e-9,
            "feasible_ratio_late": float(np.mean([p["feasible_ratio_late"] for p in per])),
            "swaps_total": float(np.mean([p["swaps_total"] for p in per])),
            "swaps_late": float(np.mean([p["swaps_late"] for p in per])),
            "T_last_swap": float(np.mean([p["T_last_swap"] for p in per])),
            "n_distinct_feasible_late": float(np.mean([p["n_distinct_feasible_late"] for p in per])),
            "best_Enorm": float(np.mean(objs)) if objs else None,
        }
        records.append(rec)
        print(f"  lambda={lam:8.2f} ({rec['lambda_over_star']:4.1f}x*)  "
              f"VR_late={rec['feasible_ratio_late']*100:5.1f}%  "
              f"swaps_tot={rec['swaps_total']:7.1f}  "
              f"T_freeze={rec['T_last_swap']:.4f}  "
              f"E_norm={rec['best_Enorm']}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "exp_lambda_analysis.csv"), index=False)
    save = {"w": w, "sigma_rel": sig_rel, "sigma_red": sig_red,
            "lambda_star": lam_star, "lambda_universal": lam_univ,
            "ratio_universal_over_star": lam_univ / lam_star,
            "records": records}
    with open(os.path.join(OUTPUT_DIR, "exp_lambda_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(save, f, indent=2, ensure_ascii=False)

    # figure B2
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = df["lambda"]
    ax1.plot(x, df["swaps_total"], "o-", color="#4C72B0", label="Total band-swaps (mixing)")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"Penalty $\lambda$ (log)")
    ax1.set_ylabel("# band-swaps over anneal", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(x, df["best_Enorm"], "s--", color="#C44E52", label="Solution energy $E_{norm}$ (lower=better)")
    ax2.set_ylabel(r"$E_{norm}$ of solution", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax1.axvline(lam_star, ls="--", color="green", alpha=0.7)
    ax1.text(lam_star, ax1.get_ylim()[1] * 0.9, r" $\lambda^*$", color="green")
    ax1.axvline(lam_univ, ls="--", color="black", alpha=0.6)
    ax1.text(lam_univ, ax1.get_ylim()[1] * 0.75, r" $\lambda_{univ}$", color="black")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center left")
    ax1.set_title("B2: Too-large $\\lambda$ keeps feasibility but freezes swaps -> worse solution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_lambda_sweep.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_lambda_sweep.pdf"), bbox_inches="tight")
    plt.close()
    print("\nDONE ->", OUTPUT_DIR)


if __name__ == "__main__":
    main()
