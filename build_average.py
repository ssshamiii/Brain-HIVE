import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_SUBJ_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?:^|/)(?:subj|heldout)-(\d+)(?:/|$)"),
    re.compile(r"(?:^|/)(?:eval|test)-(\d+)(?:/|$)"),
)
_REP_PATTERN = re.compile(r"(?:^|/)rep-(\d+)(?:/|$)", re.IGNORECASE)


def _extract_subject_id(path: str) -> Optional[int]:
    for pat in _SUBJ_PATTERNS:
        m = pat.search(path)
        if m:
            return int(m.group(1))
    return None


def _extract_repeat_id(path: str) -> Optional[int]:
    m = _REP_PATTERN.search(path)
    return int(m.group(1)) if m else None


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _default_metric_keys(d: Dict) -> List[str]:
    """
    Auto-pick metric keys from a single eval_results.json dict.
    We keep numeric keys that start with 'eval_' and exclude runtime/speed/steps.
    """
    exclude = {
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    }
    keys = []
    for k, v in d.items():
        if k in exclude:
            continue
        if k.startswith("eval_") and _is_number(v):
            keys.append(k)
    return sorted(keys)


@dataclass(frozen=True)
class Record:
    subject_id: int
    repeat_id: int
    run_dir: str
    metrics: Dict[str, float]


def load_records(exp_root: Path, filename: str = "eval_results.json") -> List[Record]:
    """
    Walk exp_root recursively and load all *filename* files.
    Subject id is inferred from the path (subj-<id> or heldout-<id>).
    Repeat id is inferred from the path (rep-<k>).
    """
    records: List[Record] = []
    for p in exp_root.rglob(filename):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        rel = str(p.parent.as_posix())

        subj = _extract_subject_id(rel)
        if subj is None:
            # If a run doesn't encode subject in the path, skip (explicit is better than wrong).
            continue

        rep = _extract_repeat_id(rel) or 1  # if no rep tag, treat as single-run repeat 1

        metrics = {k: float(v) for k, v in data.items() if _is_number(v)}
        records.append(Record(subject_id=subj, repeat_id=rep, run_dir=str(p.parent), metrics=metrics))
    return records


def aggregate_by_subject(
    records: Iterable[Record],
    metrics: Optional[List[str]] = None,
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float], List[str]]:
    """
    Returns:
      per_subject_mean: {subject_id: {metric: mean_over_repeats}}
      overall_mean: {metric: mean_over_subjects_of_subject_means}
      metric_keys: the metrics actually used
    """
    records = list(records)
    if not records:
        return {}, {}, metrics or []

    if metrics is None:
        metrics = _default_metric_keys(records[0].metrics)

    by_subj: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k in metrics:
            if k in r.metrics:
                by_subj[r.subject_id][k].append(float(r.metrics[k]))

    per_subject_mean: Dict[int, Dict[str, float]] = {}
    for subj, mm in by_subj.items():
        per_subject_mean[subj] = {
            k: (sum(vals) / len(vals)) for k, vals in mm.items() if len(vals) > 0
        }

    overall_mean: Dict[str, float] = {}
    for k in metrics:
        subj_vals = [per_subject_mean[s][k] for s in sorted(per_subject_mean) if k in per_subject_mean[s]]
        if subj_vals:
            overall_mean[k] = sum(subj_vals) / len(subj_vals)

    return per_subject_mean, overall_mean, metrics


def print_summary(per_subject: Dict[int, Dict[str, float]], overall: Dict[str, float], metrics: List[str]) -> None:
    if not per_subject:
        print("No valid eval_results.json found (or subject id not inferrable from paths).")
        return

    subjects = sorted(per_subject.keys())
    colw = max(7, *(len(m) for m in metrics)) if metrics else 7

    header = ["subject"] + metrics
    print(" | ".join(h.ljust(colw) for h in header))
    print("-" * (len(header) * (colw + 3) - 3))

    for s in subjects:
        row = [str(s)]
        for m in metrics:
            v = per_subject[s].get(m, float("nan"))
            row.append(f"{v:.6f}" if v == v else "nan")
        print(" | ".join(c.ljust(colw) for c in row))

    print("-" * (len(header) * (colw + 3) - 3))
    row = ["MEAN"]
    for m in metrics:
        v = overall.get(m, float("nan"))
        row.append(f"{v:.6f}" if v == v else "nan")
    print(" | ".join(c.ljust(colw) for c in row))


def write_json(path: Path, per_subject: Dict[int, Dict[str, float]], overall: Dict[str, float], metrics: List[str]) -> None:
    out = {
        "metrics": metrics,
        "per_subject_mean": {str(k): v for k, v in sorted(per_subject.items())},
        "overall_mean": overall,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, per_subject: Dict[int, Dict[str, float]], overall: Dict[str, float], metrics: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + metrics)
        for s in sorted(per_subject):
            w.writerow([s] + [per_subject[s].get(m, "") for m in metrics])
        w.writerow(["MEAN"] + [overall.get(m, "") for m in metrics])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", type=str, required=True, help="Root directory that contains run output dirs.")
    ap.add_argument("--filename", type=str, default="eval_results.json", help="Eval result filename to search for.")
    ap.add_argument("--metrics", nargs="*", default=None, help="Metric keys to aggregate. Default: auto-detect.")
    ap.add_argument("--out_json", type=str, default=None, help="Optional path to write summary json.")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to write summary csv.")
    args = ap.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    records = load_records(exp_root, filename=args.filename)

    per_subject, overall, metrics = aggregate_by_subject(records, metrics=args.metrics)
    print_summary(per_subject, overall, metrics)

    if args.out_json:
        write_json(Path(args.out_json), per_subject, overall, metrics)
    if args.out_csv:
        write_csv(Path(args.out_csv), per_subject, overall, metrics)


if __name__ == "__main__":
    main()
