# ================================================================
# End‑to‑end Monte‑Carlo: Fuzzy AHP  vs.  Fuzzy‑Rough AHP
#   • Perturb *every* TFN by ±10 %  (component‑wise multiplicative noise)
#   • Recompute:   DM aggregation  →  FRN layer (for FR‑AHP only)
#                  → criterion weights  w_k
#                  → alternative weights H_ik
#                  → overall scores S_i
#   • Repeat N times, record rank switches and winner–runner gap
# ================================================================
import numpy as np

# ----------------------------------------------------------------
# 0.  DATA  (Case‑2 matrices we used earlier)
# ----------------------------------------------------------------
def parse_tfn(txt):
    """'(a,b,c)' -> (a,b,c) floats"""
    a, b, c = map(float, txt.strip().lstrip("(").rstrip(")").split(","))
    return (a, b, c)

# --- DM‑level pairwise comparison matrices (4×4, TFNs) ----------
DM1 = [[parse_tfn(x) for x in row] for row in [
    "(1,1,1) (2,3,4) (4,5,6) (4,5,6)".split(),
    "(0.25,0.33,0.5) (1,1,1) (1,1,1) (2,3,4)".split(),
    "(0.1667,0.2,0.25) (1,1,1) (1,1,1) (4,5,6)".split(),
    "(0.1667,0.2,0.25) (0.25,0.33,0.5) (0.1667,0.2,0.25) (1,1,1)".split()
]]

DM2 = [[parse_tfn(x) for x in row] for row in [
    "(1,1,1) (0.1,0.111,0.125) (2,3,4) (0.25,0.33,0.5)".split(),
    "(8,9,10) (1,1,1) (6,7,8) (4,5,6)".split(),
    "(0.25,0.33,0.5) (0.125,0.1429,0.1667) (1,1,1) (0.1667,0.2,0.25)".split(),
    "(2,3,4) (0.1667,0.2,0.25) (4,5,6) (1,1,1)".split()
]]

DM3 = [[parse_tfn(x) for x in row] for row in [
    "(1,1,1) (2,3,4) (0.1667,0.2,0.25) (0.25,0.33,0.5)".split(),
    "(0.25,0.33,0.5) (1,1,1) (0.125,0.1429,0.1667) (0.1667,0.2,0.25)".split(),
    "(4,5,6) (6,7,8) (1,1,1) (4,5,6)".split(),
    "(2,3,4) (4,5,6) (0.1667,0.2,0.25) (1,1,1)".split()
]]

DM_MATRICES = [DM1, DM2, DM3]

# --- Aggregated 3×3 TFN matrices for alternatives under each criterion
ALT_MATS = {
    "C1": [
        [(1,1,1), (0.8056,1.1778,1.5833), (5.3333,6.3333,7.3333)],
        [(2.0833,2.7778,3.5), (1,1,1), (6.6667,7.6667,8.6667)],
        [(0.1583,0.1958,0.2639), (0.1167,0.1323,0.1528), (1,1,1)]
    ],
    "C2": [
        [(1,1,1), (1.0833,1.4444,1.8333), (6.6667,7.6667,8.6667)],
        [(1.0833,1.4444,1.8333), (1,1,1), (7.3333,8.3333,9.3333)],
        [(0.1167,0.1323,0.1528), (0.1083,0.1217,0.1389), (1,1,1)]
    ],
    "C3": [
        [(1,1,1), (0.8611,1.2778,1.8333), (4.3889,5.0667,5.75)],
        [(1.0667,1.7778,2.5), (1,1,1), (5.0476,5.7222,6.4)],
        [(1.412,1.756,2.1032), (1.173,2.0787,2.4226), (1,1,1)]
    ],
    "C4": [
        [(1,1,1), (1.1944,1.6111,2.1667), (4.6667,5.6667,6.6667)],
        [(1.0667,1.75,2.4444), (1,1,1), (4,5,6)],
        [(0.1528,0.181,0.2222), (0.1698,0.2056,0.2611), (1,1,1)]
    ]
}

CRITERIA = ["C1", "C2", "C3", "C4"]
ALTS     = ["A1", "A2", "A3"]

# ----------------------------------------------------------------
# 1.  Core helpers
# ----------------------------------------------------------------
def limits(val, comp):
    lower = [v for v in comp if v <= val]
    upper = [v for v in comp if v >= val]
    return (sum(lower) / len(lower), sum(upper) / len(upper))

def apply_noise_tfn(tfn, amp, rng):
    return tuple(max(1e-9, v * (1 + rng.uniform(-amp, amp))) for v in tfn)

def centroid(triplet):
    return sum(triplet) / 3.0

# ----- fuzzy AHP -----
def fuzzy_aggregate_pcm(dm_list):
    n = len(dm_list[0])
    out = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = tuple(np.mean([dm[i][j][k] for dm in dm_list])
                              for k in range(3))
    return out

def geom_row_mean_fuzzy(pcm):
    n = len(pcm)
    gm = []
    for i in range(n):
        prod = np.array([1.0, 1.0, 1.0])
        for j in range(n):
            prod *= np.array(pcm[i][j])
        gm.append(tuple(prod**(1/n)))
    return gm

def normalise_fuzzy_weights(gm):
    max_u = max(u for _, _, u in gm)
    norm  = [tuple(x / max_u for x in t) for t in gm]
    G     = [centroid(t) for t in norm]
    total = sum(G)
    return np.array([g/total for g in G])

def compute_fuzzy_criterion_weights(dm_list):
    agg = fuzzy_aggregate_pcm(dm_list)
    gm  = geom_row_mean_fuzzy(agg)
    return normalise_fuzzy_weights(gm)

# ----- fuzzy‑rough AHP -----
def frn_pcm(dm_list):
    n = len(dm_list[0])
    FRN = {}
    for i in range(n):
        lowers = [dm[i][j][0] for dm in dm_list for j in range(n)]
        mids   = [dm[i][j][1] for dm in dm_list for j in range(n)]
        uppers = [dm[i][j][2] for dm in dm_list for j in range(n)]
        for j in range(n):
            cluster = [dm[i][j] for dm in dm_list]
            agg_l = np.mean([t[0] for t in cluster])
            agg_m = np.mean([t[1] for t in cluster])
            agg_u = np.mean([t[2] for t in cluster])
            FRN[(i,j)] = (limits(agg_l, lowers),
                          limits(agg_m, mids),
                          limits(agg_u, uppers))
    return FRN

def weights_from_FRN(FRN, n):
    gm = []
    for i in range(n):
        LL=LU=ML=MU=UL=UU=1.0
        for j in range(n):
            (Ll,Ul),(Ml,Mu),(Lu,Uu) = FRN[(i,j)]
            LL*=Ll; LU*=Ul
            ML*=Ml; MU*=Mu
            UL*=Lu; UU*=Uu
        gm.append(((LL**(1/n), LU**(1/n)),
                   (ML**(1/n), MU**(1/n)),
                   (UL**(1/n), UU**(1/n))))
    max_UU = max(g[2][1] for g in gm)
    norm   = [[(a/max_UU, b/max_UU) for (a,b) in gi] for gi in gm]
    G      = [np.mean([(l+u)/2 for (l,u) in gi]) for gi in norm]
    total  = sum(G)
    return np.array([g/total for g in G])

def compute_fr_criterion_weights(dm_list):
    FRN = frn_pcm(dm_list)
    return weights_from_FRN(FRN, 4)

# ----- alt‑level weights (reuse fuzzy / fuzzy‑rough logic) -------
def alt_weights_fuzzy(pcm):
    gm  = geom_row_mean_fuzzy(pcm)
    return normalise_fuzzy_weights(gm)

def alt_weights_fr(pcm):
    FRN = frn_pcm([pcm])        # single aggregated matrix
    return weights_from_FRN(FRN, 3)

# ----- Monte‑Carlo driver ---------------------------------------
def end_to_end_monte(iters=1000, amp=0.10, seed=0):
    rng   = np.random.default_rng(seed)
    gapF  = []; gapFR = []
    switchF = switchFR = 0
    base_rank = (1,0,2)        # A2 > A1 > A3

    for _ in range(iters):
        # ---- Step 1: noisy DM matrices ----
        noisy_dms = [
            [[apply_noise_tfn(cell, amp, rng) for cell in row] for row in dm]
            for dm in DM_MATRICES
        ]

        # ---- Step 2: criterion weights ----
        wF  = compute_fuzzy_criterion_weights(noisy_dms)
        wFR = compute_fr_criterion_weights(noisy_dms)

        # ---- Step 3: alternative weights under each criterion ----
        Hf_list  = []
        Hfr_list = []
        for crit in CRITERIA:
            noisy_alt = [
                [apply_noise_tfn(cell, amp, rng) for cell in row]
                for row in ALT_MATS[crit]
            ]
            Hf_list.append(alt_weights_fuzzy(noisy_alt))
            Hfr_list.append(alt_weights_fr(noisy_alt))
        Hf  = np.column_stack(Hf_list)   # rows alt, cols criteria
        Hfr = np.column_stack(Hfr_list)

        # ---- Step 4: overall scores ----
        scoreF  = Hf  @ wF
        scoreFR = Hfr @ wFR

        rankF  = tuple(np.argsort(scoreF)[::-1])
        rankFR = tuple(np.argsort(scoreFR)[::-1])

        if rankF != base_rank:
            switchF += 1
        if rankFR != base_rank:
            switchFR += 1

        gapF.append(np.sort(scoreF)[-1]  - np.sort(scoreF)[-2])
        gapFR.append(np.sort(scoreFR)[-1] - np.sort(scoreFR)[-2])

    return {
        "Fuzzy AHP"      : (switchF / iters,  np.mean(gapF),  np.std(gapF)),
        "Fuzzy‑Rough AHP": (switchFR / iters, np.mean(gapFR), np.std(gapFR))
    }

# ----------------------------------------------------------------
# 2.  Run experiment and print summary
# ----------------------------------------------------------------
results = end_to_end_monte(iters=3000, amp=0.10, seed=2025)

print("Monte‑Carlo robustness check (3000 trials, ±10 % TFN noise)\n")
print("{:<17}  {:>8}  {:>10}  {:>9}".format(
      "Method", "switch p", "mean gap", "gap std"))
for name, (p, gmean, gstd) in results.items():
    print("{:<17}  {:8.4f}  {:10.4f}  {:9.4f}".format(name, p, gmean, gstd))
