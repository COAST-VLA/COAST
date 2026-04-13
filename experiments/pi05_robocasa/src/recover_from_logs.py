"""Recover clobbered-by-race-condition results from .err logs into summary.json."""
import json
import pathlib
import re

ROOT = pathlib.Path("/vast/projects/ungar/stellar/miaom/openpi-new/experiments/pi05_robocasa/steering_results")
LOG_DIR = ROOT / "logs"
TASKS = ["CloseFridge", "CoffeeSetupMug", "OpenDrawer", "OpenStandMixerHead",
         "PickPlaceCounterToCabinet", "PickPlaceCounterToStove", "TurnOnElectricKettle"]
PAT = re.compile(r"([a-zA-Z][a-zA-Z_0-9.]+):\s*SR=([0-9.]+)")
PREFIXES = ("global_", "per_step_", "random_", "pos_only_", "baseline")

for i, task in enumerate(TASKS):
    sp = ROOT / task / "summary.json"
    d = json.load(open(sp))
    by_cond = {c["condition"]: c for c in d["conditions"] if c}
    added = 0
    for pattern in [f"ps0rand-{i}_*.err", f"posonly-{i}_*.err",
                    "ppctc-global_*.err", "toek-ps9_*.err"]:
        for lf in LOG_DIR.glob(pattern):
            # only merge PPCtC/TOEK logs into matching task
            base = lf.name
            if base.startswith("ppctc-global") and task != "PickPlaceCounterToCabinet":
                continue
            if base.startswith("toek-ps9") and task != "TurnOnElectricKettle":
                continue
            text = open(lf).read()
            for m in PAT.finditer(text):
                cond, sr = m.group(1), float(m.group(2))
                if not cond.startswith(PREFIXES) and cond != "baseline":
                    continue
                if cond not in by_cond:
                    by_cond[cond] = {"condition": cond, "success_rate": sr}
                    added += 1
    merged = sorted(by_cond.values(), key=lambda x: x.get("success_rate", 0), reverse=True)
    with open(sp, "w") as f:
        json.dump({"task": task, "conditions": merged}, f, indent=2)
    print(f"{task}: +{added} recovered, total {len(merged)}")
