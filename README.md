# Submitted to Journal: _**Fuzzy Optimization and Decision Making**_ 
# Springer
**“Fuzzy‑Rough AHP vs. Classical Fuzzy AHP: A Robustness Study”**  


---

## 1  Overview
This repository contains all Python code needed to reproduce the results, tables, and Monte‑Carlo sensitivity analyses reported in our manuscript.  
Every script runs standalone from the command line; no proprietary software or hidden preprocessing steps are required.

## 2  Repository structure

| File | Purpose |
|------|---------|
| `Consistency Index.py` | Computes the Saaty Consistency Index (CI) and Consistency Ratio (CR) for any pair‑wise comparison matrix. |
| `Defuzzification.py` | Converts triangular fuzzy numbers (TFNs) to crisp values via centroid defuzzification and normalises the resulting weight vector. |
| `fuzzy geometric mean.py` | Calculates the fuzzy geometric mean of expert judgments (component‑wise) for the AHP aggregation step. |
| `Fuzzy‑Rough AHP‑case2.py` | End‑to‑end pipeline that builds the **Fuzzy‑Rough AHP** hierarchy for Case 2, producing criterion and alternative weights plus overall scores. |
| `FullSensitivity‑case2.py` | **End‑to‑end Monte‑Carlo** experiment: perturbs every TFN by ±10 %, recomputes the full FR‑AHP pipeline, and records rank switches and winner–runner gaps. |
| `sensitivity.py` | Monte‑Carlo **robustness check** that adds ±10 % noise only to the aggregated criterion weights and compares Fuzzy AHP with Fuzzy‑Rough AHP. |
| `Defuzzification and Normalize Fuzzy Weights.py` <br>*(kept for backward compatibility)* | Older helper used in early drafts; superseded by `Defuzzification.py` but left here for completeness. |

> **Note:** Scripts end with printed summaries; redirect output to a file if you prefer permanent logs.

## 3  Requirements
* Python 3.8 +  
* `numpy >= 1.20`  
* `pandas >= 1.3` *(required only for `FullSensitivity‑case2.py`)*  

Install via:

```bash
pip install -r requirements.txt
