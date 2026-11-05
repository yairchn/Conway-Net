"""Matched-accuracy targets for the 2-layer FCN set (configs/fcn2_matched): per dataset, the lowest best phase-1
val accuracy of the arms with lateral inhibition (one per seed), minus 0.005 (0.5 points); checks that every plain arm's best
reaches target + offset. Writes paper_runs/fcn2_matched_targets.json and prints one phase-2 command line per seed.
Run: python scripts/fcn2_matched_targets.py [prefix] [n_seeds]   (default: fcn2 5; e.g. fcn5 2 for the 5-layer set)
"""
import json
import sys

PRE = sys.argv[1] if len(sys.argv) > 1 else "fcn2"; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 5

T = {}
for ds in ("mnist", "fashionmnist", "kmnist", "emnist"):
    ca = [max(json.load(open(f"paper_runs/{PRE}_matched_{ds}_s{k}/val_history_ca.json"))) for k in range(NS)]
    no = [max(json.load(open(f"paper_runs/{PRE}_matched_{ds}_s{k}/val_history_none.json"))) for k in range(NS)]
    t = round(min(ca) - 0.005, 4)
    assert min(no) >= t + 0.0025, f"{ds}: a plain arm peaks at {min(no):.4f}, below target {t} + offset"
    T[ds] = dict(target=t, ca_best=ca, plain_best=no)
json.dump(T, open(f"paper_runs/{PRE}_matched_targets.json", "w"), indent=2)
for ds, v in T.items():
    for k in range(NS):
        print(f"configs/{PRE}_matched/{ds}_s{k}_ca.json configs/{PRE}_matched/{ds}_s{k}_none.json {v['target']}")
