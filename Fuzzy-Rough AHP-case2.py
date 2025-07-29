# ---------------------------------------------------------------
# Fuzzy-Rough AHP  (Case 2 )
# ---------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------- 1. Input: aggregated TFN matrices -------------------
# Four criteria (C1-C4), three alternatives (A1-A3)

CRITERIA = ["C1", "C2", "C3", "C4"]
ALTS     = ["A1", "A2", "A3"]

TFN_DATA = {
    "C1": [  # Service
        [(1,1,1), (0.8056,1.1778,1.5833), (5.3333,6.3333,7.3333)],
        [(2.0833,2.7778,3.5), (1,1,1),   (6.6667,7.6667,8.6667)],
        [(0.1583,0.1958,0.2639),(0.1167,0.1323,0.1528),(1,1,1)]
    ],
    "C2": [  # Cost
        [(1,1,1), (1.0833,1.4444,1.8333), (6.6667,7.6667,8.6667)],
        [(1.0833,1.4444,1.8333), (1,1,1), (7.3333,8.3333,9.3333)],
        [(0.1167,0.1323,0.1528),(0.1083,0.1217,0.1389),(1,1,1)]
    ],
    "C3": [  # Time
        [(1,1,1),(0.8611,1.2778,1.8333),(4.3889,5.0667,5.75)],
        [(1.0667,1.7778,2.5),(1,1,1),(5.0476,5.7222,6.4)],
        [(1.412,1.756,2.1032),(1.173,2.0787,2.4226),(1,1,1)]
    ],
    "C4": [  # Quality
        [(1,1,1),(1.1944,1.6111,2.1667),(4.6667,5.6667,6.6667)],
        [(1.0667,1.75,2.4444),(1,1,1),(4,5,6)],
        [(0.1528,0.181,0.2222),(0.1698,0.2056,0.2611),(1,1,1)]
    ]
}

# ---------- 2. Helper: fuzzy-rough number for one row -----------
def limits(value, comp_list):
    lo = [v for v in comp_list if v <= value]
    up = [v for v in comp_list if v >= value]
    return np.mean(lo), np.mean(up)

def frn_matrix(matrix):
    """Return dict[(i,j)] = ((Ll,Ul),(Lm,Um),(Lu,Uu))"""
    n = len(matrix)
    FRN = {}
    for i in range(n):
        lowers = [matrix[i][j][0] for j in range(n)]
        mids   = [matrix[i][j][1] for j in range(n)]
        uppers = [matrix[i][j][2] for j in range(n)]
        for j in range(n):
            l, m, u = matrix[i][j]
            FRN[(i,j)] = (limits(l, lowers),
                          limits(m, mids),
                          limits(u, uppers))
    return FRN

# ---------- 3. Row geometric mean, global normalisation, H ------
def weights_from_FRN(FRN, n=3):
    """Return dicts: FR_weights, norm_intervals, G_i, H_i"""
    FRW = {}
    for i in range(n):
        LL=LU=ML=MU=UL=UU=1
        for j in range(n):
            (Ll,Ul),(Ml,Mu),(Lu,Uu) = FRN[(i,j)]
            LL*=Ll; LU*=Ul
            ML*=Ml; MU*=Mu
            UL*=Lu; UU*=Uu
        FRW[i] = ( (LL**(1/n), LU**(1/n)),
                   (ML**(1/n), MU**(1/n)),
                   (UL**(1/n), UU**(1/n)) )
    # global anchor
    max_UU = max(FRW[i][2][1] for i in range(n))
    NORM = {i: tuple((a/max_UU, b/max_UU) for (a,b) in FRW[i]) for i in range(n)}
    # defuzzify
    mid = lambda x: (x[0]+x[1])/2
    G = {i: np.mean([mid(iv) for iv in NORM[i]]) for i in range(n)}
    total = sum(G.values())
    H = {i: G[i]/total for i in range(n)}
    return FRW, NORM, G, H

# ---------- 4. Run over all criteria ----------------------------
criterion_weights = {}
alt_H = {alt: [] for alt in ALTS}      # store H_ik per criterion
for k, crit in enumerate(CRITERIA):
    FRN = frn_matrix(TFN_DATA[crit])
    FRW, NORM, G, H = weights_from_FRN(FRN)
    # save alternative weights under this criterion
    for i, alt in enumerate(ALTS):
        alt_H[alt].append(H[i])
    # row geometric mean of FRNs -> criterion weight w_k
    #   (use upper-upper values as recommended)
    w_k = max(FRW[i][2][1] for i in range(3))
    criterion_weights[crit] = w_k

# normalise criterion weights so sum=1
total_w = sum(criterion_weights.values())
criterion_weights = {c: w/total_w for c, w in criterion_weights.items()}

# ---------- 5. Build result tables ------------------------------
df_H = pd.DataFrame(alt_H, index=CRITERIA).T
df_crit = pd.Series(criterion_weights, name="w_k")

# overall scores
overall = df_H.mul(df_crit, axis=1).sum(axis=1)

print("\n=== Criterion weights (w_k) ===")
print(df_crit)
print("\n=== Normalised crisp weights H_ik (rows = alternatives) ===")
print(df_H)
print("\n=== Overall scores S_i ===")
print(overall.sort_values(ascending=False))