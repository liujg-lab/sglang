"""Compare the fresh-fit Draft install log with eligible cells in the saved fit."""
import math
import re

from supervisor import META


def check(fit, log):
    from sglang.srt.speculative.clear_tree_remote.clear_tree_ledger import (
        _new_layer_model,
        attach_layer_slope_priors,
    )

    cap = META["prefit_layer_weight"]
    models = {}
    for row in fit["layers"]:
        key = (row["w"], row["path"], row["g"])
        model = _new_layer_model()
        model.from_dict(row)
        models[key] = model
    slopes = attach_layer_slope_priors(models)
    eligible = [{"cell": key, "stored_weight": float(model.xtx[0][0]), "n": model.n}
                for key, model in sorted(models.items()) if model.slope_prior is not None
                and math.isfinite(float(model.xtx[0][0])) and float(model.xtx[0][0]) > cap]
    writes = [i for i, line in enumerate(log.splitlines()) if "[CLEAR] wrote fit file" in line]
    pattern = r"\[CLEAR\] prefit layer weight capped cap=(\d+)(?: \(off\))? cells=(\d+)\s*$"
    lines = log.splitlines()
    events = [(i, int(m[1]), int(m[2])) for i, line in enumerate(lines) if (m := re.search(pattern, line))]
    installed = events[0] if events else None
    slope_lines = [i for i, line in enumerate(lines) if line.endswith(f"[CLEAR] layer slope priors installed cells={slopes}")]
    passed = bool(len(writes) == 1 and installed and installed[1:] == (cap, len(eligible))
                  and any(writes[0] < i < installed[0] for i in slope_lines))
    return dict(passed=passed, expected_cap=cap, expected_cells=len(eligible), expected_slope_priors=slopes,
                eligible_cells=eligible, observed=[{"line": i + 1, "cap": value, "cells": cells} for i, value, cells in events])
