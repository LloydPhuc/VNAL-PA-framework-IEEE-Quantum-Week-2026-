"""
=============================================================================
 07 - Pareto frontier via SIMULATED ANNEALING (faithful to the VNAL-PA solver),
      comparing three standardization regimes for the SAME cardinality-
      constrained QUBO:

   (R) Raw         : no standardization (sigma = 1).
   (L) Lee-global  : standardize each objective by its std over {0,1}^n
                     (Bernoulli(0.5) bits) -- Lee et al., "Standardization of
                     Multi-Objective QUBOs". This implicitly assumes ~n/2
                     selected bits, NOT the operating cardinality K.
   (V) VNAL-PA     : standardize by std over the K-cardinality subspace
                     (exactly K bands) -- our proposal.

 SOLVER = pure simulated annealing (neal), exactly as in 02_vnal_pa.py:
   lambda_universal is ONLY an upper bound for feasibility; the model is solved
   with the binary-searched MINIMAL feasible penalty lambda* (small), at which SA
   moves freely and returns exactly-K-band solutions -- no greedy, no warm start.

 Objectives (both minimized), reported in RAW units:
   f(x) = relevance energy  = -sum_{i in S} Rel(i)
   g(x) = redundancy energy =  sum_{i<j in S} Red(i,j)

 Panel (a): SA Pareto frontier (sweep w under VNAL norm, lambda* per w) with the
            equal-weight (w=0.5) SA solution of each regime + the ideal vector.
            Raw collapses to the low-redundancy extreme; Lee-global over-shoots to
            the relevance extreme (variance estimated at the wrong cardinality);
            VNAL-PA (K-subspace) lands on the balanced knee.
 Panel (b): variance ratio sigma_red/sigma_rel, global vs K-subspace.

 OUTPUT: results/exp_pareto_frontier.png / .pdf
=============================================================================
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neal

K = 64
MC_SAMPLES = 20000             # match exp05 (paper's 571x scale figure)
MC_SEED = 42
N_W = 30                       # ~30 frontier weights
VR_THRESH = 0.95
BS_STEPS = 7                   # binary-search steps for lambda*
SA_SEEDS = (42, 123)
SA_READS = 150
SA_SWEEPS = 2000
BS_READS = 60                  # lighter SA inside the lambda* search
BS_SWEEPS = 800


def resolve():
    base = r"D:\HYPERSPECTRAL 1"
    return "/kaggle/working/FULL_PIPELINE" if os.path.isdir("/kaggle/input") \
        else os.path.join(base, "FULL_PIPELINE")


# ----------------------------------------------------------------- sigma (MC)
def sigma_Ksubset(rel, R, k, n, seed):
    rng = np.random.default_rng(seed)
    B = len(rel)
    rels = np.empty(n); reds = np.empty(n)
    for s in range(n):
        sel = rng.choice(B, size=k, replace=False)
        rels[s] = -np.sum(rel[sel])
        sub = R[np.ix_(sel, sel)]
        reds[s] = (np.sum(sub) - np.trace(sub)) / 2.0
    return max(float(np.std(rels)), 1e-12), max(float(np.std(reds)), 1e-12)


def sigma_bernoulli(rel, R, n, seed):
    """Lee-global: std over X ~ Uniform({0,1}^B), each bit Bernoulli(0.5)."""
    rng = np.random.default_rng(seed)
    B = len(rel)
    rels = np.empty(n); reds = np.empty(n)
    for s in range(n):
        sel = np.where(rng.random(B) < 0.5)[0]
        rels[s] = -np.sum(rel[sel]) if sel.size else 0.0
        if sel.size > 1:
            sub = R[np.ix_(sel, sel)]
            reds[s] = (np.sum(sub) - np.trace(sub)) / 2.0
        else:
            reds[s] = 0.0
    return max(float(np.std(rels)), 1e-12), max(float(np.std(reds)), 1e-12)


# ----------------------------------------------------------------- QUBO + SA
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
    return Q


def lambda_universal(rel, R, sig_rel, sig_red):
    gain = (1.0 / sig_rel) * np.max(rel) + (1.0 / sig_red) * np.max(np.sum(R, axis=1))
    return float(gain * 1.1)


def sample(rel, R, k, w, sr, sd, lam, reads, sweeps, seed):
    Q = build_qubo(rel, R, k, w, sr, sd, lam)
    return neal.SimulatedAnnealingSampler().sample_qubo(
        Q, num_reads=reads, num_sweeps=sweeps, seed=seed)


def valid_ratio(ss, k):
    tot = val = 0
    for d in ss.data(["sample"]):
        tot += 1
        if sum(d.sample.values()) == k:
            val += 1
    return val / max(1, tot)


def best_valid(rel, R, k, w, sr, sd, lam, reads, sweeps):
    """Min-energy exactly-K sample over SA_SEEDS (fallback: closest to K)."""
    best, best_e = None, float("inf")
    closest, closest_d = None, 10 ** 9
    for seed in SA_SEEDS:
        ss = sample(rel, R, k, w, sr, sd, lam, reads, sweeps, seed)
        for d in ss.data(["sample", "energy"]):
            sel = [b for b, v in d.sample.items() if v == 1]
            if len(sel) == k:
                if d.energy < best_e:
                    best_e, best = float(d.energy), sel
            elif abs(len(sel) - k) < closest_d:
                closest_d, closest = abs(len(sel) - k), sel
    return tuple(sorted(best if best is not None else closest))


def binary_search_lambda(rel, R, k, w, sr, sd, lam_hi):
    """Smallest lambda with valid ratio >= VR_THRESH (cf. 02_vnal_pa.py)."""
    lo, hi, opt = 0.0, lam_hi, lam_hi
    for _ in range(BS_STEPS):
        mid = (lo + hi) / 2.0
        vr = np.mean([valid_ratio(sample(rel, R, k, w, sr, sd, mid,
                                         BS_READS, BS_SWEEPS, s), k)
                      for s in SA_SEEDS])
        if vr >= VR_THRESH:
            hi = opt = mid
        else:
            lo = mid
    return opt


def fg(rel, R, S):
    S = list(S)
    sub = R[np.ix_(S, S)]
    return -float(np.sum(rel[S])), float((np.sum(sub) - np.trace(sub)) / 2.0)


def pareto_filter(pts):
    pts = sorted(set(pts))
    front, best_g = [], np.inf
    for f, g in pts:
        if g < best_g - 1e-9:
            front.append((f, g)); best_g = g
    return front


def main():
    ROOT = resolve()
    CACHE = os.path.join(ROOT, "cache")
    RESULTS = os.path.join(ROOT, "results")
    os.makedirs(RESULTS, exist_ok=True)

    rel = np.load(os.path.join(CACHE, "relevance.npy")).astype(float)
    R = np.load(os.path.join(CACHE, "redundancy_lw.npy")).astype(float)
    np.fill_diagonal(R, 0.0)

    srV, sdV = sigma_Ksubset(rel, R, K, MC_SAMPLES, MC_SEED)
    srL, sdL = sigma_bernoulli(rel, R, MC_SAMPLES, MC_SEED)
    ratioV, ratioL = sdV / srV, sdL / srL
    print(f"[K-subspace] sig_rel={srV:.4f} sig_red={sdV:.3f}  ratio={ratioV:.1f}x")
    print(f"[Lee-global] sig_rel={srL:.4f} sig_red={sdL:.3f}  ratio={ratioL:.1f}x")

    # ---- frontier: sweep w (VNAL norm); lambda* binary-searched per regime,
    #      scaled across w by the per-w binding scale (one search, cheap sweep) ----
    def bind(w, sr, sd):
        return (1 - w) / sr * np.max(rel) + w / sd * np.max(np.sum(R, axis=1))

    lamV = binary_search_lambda(rel, R, K, 0.5, srV, sdV,
                                lambda_universal(rel, R, srV, sdV))
    print(f"[lambda*] VNAL @w=.5 = {lamV:.4f}")
    ref = bind(0.5, srV, sdV)

    sols = set()
    for w in np.linspace(0.05, 0.95, N_W):
        lam_w = lamV * bind(w, srV, sdV) / ref
        sols.add(best_valid(rel, R, K, w, srV, sdV, lam_w, SA_READS, SA_SWEEPS))
    front = np.array(pareto_filter([fg(rel, R, S) for S in sols]))
    print(f"[front] {len(front)} non-dominated SA subsets (from {N_W} weights)")

    # ---- equal-weight (w=0.5) SA solution under each regime ----
    regimes = {
        "raw": (1.0, 1.0, "#7f7f7f", "Raw  $w{=}0.5$"),
        "lee": (srL, sdL, "#c0392b", "Lee-global  $w{=}0.5$"),
        "vnl": (srV, sdV, "#1f77b4", "VNAL-PA ($K$)  $w{=}0.5$"),
    }
    P = {}
    for name, (sr, sd, _, _) in regimes.items():
        lam = binary_search_lambda(rel, R, K, 0.5, sr, sd,
                                   lambda_universal(rel, R, sr, sd))
        S = best_valid(rel, R, K, 0.5, sr, sd, lam, SA_READS, SA_SWEEPS)
        P[name] = fg(rel, R, S)
        print(f"[w=.5 {name:3s}] lambda*={lam:7.3f}  f={P[name][0]:.3f} g={P[name][1]:.1f}")

    f_ideal, g_ideal = float(front[:, 0].min()), float(front[:, 1].min())

    # =====================================================================
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.2),
                                   gridspec_kw={"width_ratios": [2.2, 1.0]})

    axA.plot(front[:, 1], front[:, 0], "-", color="0.6", lw=1.0, zorder=1)
    axA.scatter(front[:, 1], front[:, 0], s=24, color="black",
                label="Pareto front (SA)", zorder=2)
    axA.scatter([g_ideal], [f_ideal], marker="*", s=340, color="#2e8b57",
                edgecolor="k", linewidth=0.6, label="Ideal vector", zorder=5)
    for name, (_, _, col, lab) in regimes.items():
        axA.scatter([P[name][1]], [P[name][0]], marker="v", s=190, color=col,
                    edgecolor="k", linewidth=0.6, label=lab, zorder=6)
    axA.set_xlabel("redundancy energy $g(x)$ (raw)")
    axA.set_ylabel("relevance energy $f(x)$ (raw)")
    axA.set_title("(a) Equal-weight solution on the $K{=}64$ SA Pareto frontier")
    axA.grid(alpha=0.25)
    axA.legend(loc="upper right", fontsize=8.5)

    bars = axB.bar(["Lee-global\n$\\{0,1\\}^n$", "VNAL-PA\n$K$-subspace"],
                   [ratioL, ratioV], color=["#c0392b", "#1f77b4"],
                   edgecolor="k", linewidth=0.6, width=0.6)
    for b, v in zip(bars, [ratioL, ratioV]):
        axB.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}$\\times$",
                 ha="center", va="bottom", fontsize=10)
    axB.set_ylabel(r"$\sigma_{\mathrm{red}}/\sigma_{\mathrm{rel}}$")
    axB.set_title("(b) Scale ratio\n(global vs $K$)")
    axB.set_ylim(0, max(ratioL, ratioV) * 1.18)
    axB.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    out = os.path.join(RESULTS, "exp_pareto_frontier")
    plt.savefig(out + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(out + ".pdf", bbox_inches="tight")
    plt.close()
    print("DONE ->", out + ".png")


if __name__ == "__main__":
    main()
