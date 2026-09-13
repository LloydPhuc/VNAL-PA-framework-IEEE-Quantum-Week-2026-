"""
=============================================================================
 EXPERIMENT A : Hieu qua cua VARIANCE-NORMALIZATION bang MONTE CARLO
=============================================================================
 MUC TIEU: chung minh tai sao phai chuan hoa phuong sai (Eq.4 trong paper).

 Y TUONG:
   - Relevance-energy  E_rel(S) = -sum_{i in S} Rel(i)          (tong K so hang)
   - Redundancy-energy E_red(S) =  sum_{i<j in S} Red(i,j)      (tong K(K-1)/2 cap)
   Hai dai luong nay co THANG DO/PHUONG SAI rat khac nhau (E_red la tong cua
   ~2000 cap nen bien thien lon hon nhieu). Neu gop bang trong so w ma KHONG
   chuan hoa, mot thanh phan se ap dao -> w gan nhu vo nghia.

 BANG CHUNG (3 panel):
   (a) Phan bo E_rel vs E_red THO (raw) qua M mau Monte Carlo  -> lech thang do.
   (b) Sau khi chia cho sigma -> hai phan bo ve cung thang do (~unit variance).
   (c) "Balance curve": ty le dong gop cua redundancy theo w,
        raw vs normalized. Cho thay w=0.5 chi can bang dung KHI da chuan hoa.

 OUTPUT: exp_variance_norm.png/.pdf  +  exp_variance_norm.json
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- paths: doc tu cache do 01_compute_matrices.py tao ra ----
ROOT = "/kaggle/working/FULL_PIPELINE" if os.path.isdir("/kaggle/input") else r"D:\HYPERSPECTRAL 1\FULL_PIPELINE"
CACHE = os.path.join(ROOT, "cache")
P_REL = os.path.join(CACHE, "relevance.npy")
P_R   = os.path.join(CACHE, "redundancy_lw.npy")
OUTPUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = {"K": 64, "M": 20000, "SEED": 42}


def mc_energies(rel, R, k, M, seed):
    rng = np.random.default_rng(seed)
    B = len(rel)
    E_rel = np.empty(M)
    E_red = np.empty(M)
    for s in range(M):
        sel = rng.choice(B, size=k, replace=False)
        E_rel[s] = -np.sum(rel[sel])
        sub = R[np.ix_(sel, sel)]
        E_red[s] = (np.sum(sub) - np.trace(sub)) / 2.0
    return E_rel, E_red


def main():
    rel = np.load(P_REL).astype(np.float64)        # KSG-100 (da clip >=0 o buoc 01)
    R = np.load(P_R).astype(np.float64)            # Ledoit-Wolf |corr| (da [0,1])
    np.fill_diagonal(R, 0.0)
    k = CFG["K"]

    E_rel, E_red = mc_energies(rel, R, k, CFG["M"], CFG["SEED"])
    sig_rel, sig_red = E_rel.std(), E_red.std()
    mu_rel, mu_red = E_rel.mean(), E_red.mean()
    ratio = sig_red / sig_rel

    stats = {
        "K": k, "M": CFG["M"],
        "sigma_rel": float(sig_rel), "sigma_red": float(sig_red),
        "mu_rel": float(mu_rel), "mu_red": float(mu_red),
        "sigma_ratio_red_over_rel": float(ratio),
        # w can de can bang dong gop NEU KHONG chuan hoa:
        "w_balance_raw": float(sig_rel / (sig_rel + sig_red)),
        "w_balance_normalized": 0.5,
        "interpretation": (
            "Khong chuan hoa: 1 thanh phan co sigma lon gap %.1f lan -> can w=%.3f "
            "moi can bang (lech xa 0.5). Sau chuan hoa: w=0.5 can bang dung -> w la "
            "nut dieu khien trade-off co y nghia." % (ratio, sig_rel / (sig_rel + sig_red))
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "exp_variance_norm.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # ---------------- FIGURE ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) raw - hai truc khac nhau
    ax = axes[0]
    ax.hist(E_rel, bins=60, color="#4C72B0", alpha=0.85, label="Relevance energy")
    ax.set_xlabel("Relevance energy  $E_{rel}$ (raw)", color="#4C72B0")
    ax.tick_params(axis="x", labelcolor="#4C72B0")
    ax.set_ylabel("Count")
    ax2 = ax.twiny()
    ax2.hist(E_red, bins=60, color="#C44E52", alpha=0.55, label="Redundancy energy")
    ax2.set_xlabel("Redundancy energy  $E_{red}$ (raw)", color="#C44E52")
    ax2.tick_params(axis="x", labelcolor="#C44E52")
    ax.set_title(f"(a) RAW distributions differ in scale\n"
                 r"$\sigma_{red}/\sigma_{rel}$ = " + f"{ratio:.1f}x")

    # (b) normalized - cung thang do
    ax = axes[1]
    ax.hist((E_rel - mu_rel) / sig_rel, bins=60, color="#4C72B0", alpha=0.75,
            label=r"$\widehat{E}_{rel}$", density=True)
    ax.hist((E_red - mu_red) / sig_red, bins=60, color="#C44E52", alpha=0.55,
            label=r"$\widehat{E}_{red}$", density=True)
    ax.set_xlabel("Variance-normalized energy (z-score)")
    ax.set_ylabel("Density")
    ax.set_title("(b) After /$\\sigma$: matched scale\n(both ~unit variance)")
    ax.legend()

    # (c) balance curve
    ax = axes[2]
    w = np.linspace(0.001, 0.999, 400)
    share_raw = (w * sig_red) / (w * sig_red + (1 - w) * sig_rel)
    share_norm = (w * 1.0) / (w * 1.0 + (1 - w) * 1.0)  # = w
    ax.plot(w, share_raw * 100, color="#C44E52", lw=2, label="Raw (no norm)")
    ax.plot(w, share_norm * 100, color="#4C72B0", lw=2, label="Variance-normalized")
    ax.axhline(50, ls=":", color="gray")
    wb = sig_rel / (sig_rel + sig_red)
    ax.axvline(wb, ls="--", color="#C44E52", alpha=0.7)
    ax.annotate(f"raw balance\n@ w={wb:.3f}", xy=(wb, 50), xytext=(wb + 0.05, 25),
                color="#C44E52", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#C44E52"))
    ax.axvline(0.5, ls="--", color="#4C72B0", alpha=0.7)
    ax.set_xlabel("Trade-off weight $w$")
    ax.set_ylabel("Redundancy contribution share (%)")
    ax.set_title("(c) Only after normalization\ndoes $w=0.5$ balance the objectives")
    ax.legend(loc="lower right")

    plt.suptitle("Variance-normalization via Monte Carlo equalizes objective scales",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_variance_norm.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_variance_norm.pdf"), bbox_inches="tight")
    plt.close()
    print("\nDONE ->", OUTPUT_DIR)


if __name__ == "__main__":
    main()
