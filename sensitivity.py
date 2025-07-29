# ---------------------------------------------------------------
# Monte-Carlo robustness check: Fuzzy AHP vs. FuzzyRough AHP
  % \item Case-2 criterion weights ($w_k$)
  % \item Case-2 alternative--within--criterion weights ($H_{ik}$)
  % \item $\pm$10\% uniform noise added to each $w_k$, re-normalised
  % \item 10{,}000 repetitions
# ---------------------------------------------------------------
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---------- baseline data --------------------------------------
w_fuzzy = np.array([0.2359, 0.2878, 0.2647, 0.2116])
w_fr    = np.array([0.2106, 0.3044, 0.2689, 0.2161])

H_fuzzy = np.array([
    [0.3893, 0.4669, 0.3372, 0.4675],  # A1
    [0.5507, 0.4799, 0.3875, 0.4573],  # A2
    [0.0600, 0.0531, 0.2754, 0.0753]   # A3
])
H_fr = np.array([
    [0.404, 0.457, 0.361, 0.476],      # A1
    [0.533, 0.486, 0.412, 0.444],      # A2
    [0.063, 0.057, 0.228, 0.080]       # A3
])

ALTS = ["A1", "A2", "A3"]

# ---------- helper functions ------------------------------------
def rank_order(scores):
    """Return tuple of indices sorted descending (0=A1,1=A2,2=A3)."""
    return tuple(np.argsort(scores)[::-1])

def simulation(w_base, H, n_iter=10_000, amp=0.10, seed=42):
    rng = np.random.default_rng(seed)
    base_rank = rank_order(H @ w_base)
    gaps = []
    switch_total = 0
    switch_top   = 0

    for _ in range(n_iter):
        noise = rng.uniform(-amp, amp, size=w_base.size)
        w = w_base * (1 + noise)
        w = w / w.sum()                  # re-normalise
        scores = H @ w
        rank   = rank_order(scores)

        gaps.append(np.sort(scores)[-1] - np.sort(scores)[-2])
        if rank != base_rank:
            switch_total += 1
        if rank[0] != base_rank[0]:
            switch_top += 1

    gaps = np.asarray(gaps)
    return {
        "rank_base"     : base_rank,
        "switch_prob"   : switch_total / n_iter,
        "switch_top"    : switch_top   / n_iter,
        "gap_mean"      : gaps.mean(),
        "gap_std"       : gaps.std(),
    }

# ---------- run Monte-Carlo -------------------------------------
res_fuzzy = simulation(w_fuzzy, H_fuzzy)
res_fr    = simulation(w_fr,    H_fr)

# ---------- pretty print ----------------------------------------
print("\nMonte-Carlo results (10 000 trials, +/-10%% noise)\n")


def explain(res, label):
    idx = res["rank_base"]
    ranks = [ALTS[i] for i in idx]
    print(f"{label}:")
    print(f"  baseline rank order   : {ranks}")
    print(f"  switch prob (any rank): {res['switch_prob']:.4f}")
    print(f"  switch prob (winner)  : {res['switch_top']:.4f}")
    print(f"  mean winner gap       : {res['gap_mean']:.4f}")
    print(f"  gap std. deviation    : {res['gap_std']:.4f}\n")

explain(res_fuzzy, "Fuzzy AHP")
explain(res_fr,    "Fuzzy-Rough AHP")