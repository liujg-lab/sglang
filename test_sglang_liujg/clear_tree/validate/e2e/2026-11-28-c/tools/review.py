#!/usr/bin/env python3
"""Review the four formal trials and independently re-check the cap log."""
import sys

from fit_cap import check
from supervisor import META, RAW, REFERENCE, SYNC, base, load

review = load("ct_reference_review", REFERENCE / "tools/review.py")
review.PREVIOUS = REFERENCE  # Use the immediately preceding data baseline.
original_review_mode = review.review_mode
original_check_trial = review.check_trial


def review_mode(mode, trials, rids, notes):
    result, gates = original_review_mode(mode, trials, rids, notes)
    root = RAW / mode
    fit = review.read_json(root / base.NAMES[mode])
    proof = None
    if fit is not None:
        try:
            proof = check(fit, review.read(root / "draft.log"))
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(f"{mode}: cap acceptance error: {exc}")
    live = review.read_json(root / "fit-cap.json")
    result["fit_cap"] = proof
    gates["fit_cap"] = bool(proof and live and proof["passed"] and live.get("passed")
                           and proof["expected_cap"] == live.get("expected_cap")
                           and proof["expected_cells"] == live.get("expected_cells"))
    gates["fit_fingerprint"] = (result.get("fit") or {}).get("fingerprint") == META["fit_fingerprint"]
    gates["sync_completed"] = (SYNC / "sync-complete.txt").is_file()
    return result, gates


def check_trial(trial):
    result, rids, gates = original_check_trial(trial)
    baseline = review.read_json(REFERENCE / "summary.json")
    item_hashes = {row["ordered_items_sha256"] for row in baseline["trials"]}
    gates["reference_order"] = item_hashes == {result.get("ordered_items_sha256")}
    return result, rids, gates


review.review_mode, review.check_trial = review_mode, check_trial

if __name__ == "__main__":
    sys.exit(review.main())
