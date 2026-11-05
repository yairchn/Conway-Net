"""Summary of diagnostics/percolation_epochs.json: per run and arm, over the plateau (epochs >= E/4):
mean and s.d. of k_c, mean epoch-to-epoch |change| of k_c, mean Jaccard overlap with the previous epoch's edge
set, giant-component fraction at the largest <k>, mean staircase amplitude; and the measurement floor at the last epoch (k_c s.d. and edge overlap across
disjoint image sets). Run: python paper_code/percolation_epochs_summary.py"""
from conwaynet import REPO_ROOT
import json

import numpy as np

d = json.load(open(f"{REPO_ROOT}/diagnostics/percolation_epochs.json"))
for run, r in d.items():
    for kind in ("abs", "pos"):
        print(f"\n{run} [{kind}]   plateau = epochs >= {r['ca']['epochs'][-1] // 4}")
        for mech in ("ca", "none"):
            rows = r[mech][kind]; E = r[mech]["epochs"][-1]; P = [x for x in rows if x["epoch"] >= E // 4]
            kc = np.array([np.nan if x["k_c"] is None else x["k_c"] for x in P]); ov = np.array([x["overlap_prev"] for x in P])
            fl = r[mech]["floor"][kind]; fkc = np.array([np.nan if x["k_c"] is None else x["k_c"] for x in fl + [rows[-1]]], dtype=float)
            print(f"  {mech:4s} k_c {np.nanmean(kc):.2f} ± {np.nanstd(kc):.2f}  |dk_c| {np.nanmean(np.abs(np.diff(kc))):.3f}  "
                  f"overlap {ov.mean():.3f}  giant@k=8 {np.mean([x['lcc_grid'][-1] for x in P]):.2f}  jump {np.mean([x['max_jump'] for x in P]):.3f} | "
                  f"floor: k_c s.d. {np.nanstd(fkc):.3f}  overlap {np.mean([x['overlap_with_main'] for x in fl]):.3f}  "
                  f"| ep0 k_c {rows[0]['k_c']}, jump {rows[0]['max_jump']:.3f}")
