"""
Offline Phase D Statistical Evaluation Suite for CrossVQA (ALT_THREE_SPLITS).

Consumes per-sample `preds_{eval_split}_{docvqa,textvqa}.jsonl` files and computes:
  - DocVQA ANLS & TextVQA VQA Accuracy + Harmonic Mean
  - 95% Bootstrap Confidence Intervals (B = 1,000 resamples by default)
  - Paired Permutation Significance Tests (two-sided, p < 0.05) across shared sample_ids
  - Transfer Gain (TG vs. B1) and Forgetting Rate (FR vs. B2) with 95% Bootstrap CIs
  - Markdown & LaTeX thesis tables + optional few-shot learning curve figure

Usage:
    python EVALUATION/ALT_THREE_SPLITS/evaluate_statistical_suite.py \
        --model smolvlm_500m --namespace ebs64 --seed 42 --eval_split test
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TRAIN.src.metrics import compute_anls, compute_forgetting_rate, compute_transfer_gain, compute_vqa_accuracy

EXP_FOLDER_MAP = {
    "Z0": "Z0",
    "B1": "B1_target_only",
    "B2": "B2_source_only",
    "B3": "B3_joint",
    "B4": "B4_seq_doc2text",
    "B5": "B5_seq_text2doc",
    "M1": "M1_dual_lora",
    "M2": "M2_dual_lora_align",
    "M3": "M3_curriculum",
}


def resolve_run_dir(
    runs_root: Path,
    model: str,
    namespace: str,
    exp: str,
    seed: int,
    few_shot_pct: int = 100,
) -> Path:
    """Resolves run artifact directory matching TRAIN/ALT_THREE_SPLITS/scripts/_common.py."""
    if exp == "Z0":
        return runs_root / model / "Z0"
    exp_dir = EXP_FOLDER_MAP[exp]
    ns = namespace.strip().strip("/")
    base = runs_root / model / ns if ns else runs_root / model
    if int(few_shot_pct) < 100:
        return base / "few_shot" / f"{exp}_pct_{int(few_shot_pct):02d}" / f"seed_{int(seed)}"
    return base / exp_dir / f"seed_{int(seed)}"


def load_jsonl_scores(jsonl_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Loads sample_ids, ANLS scores, and VQA Accuracy scores from a predictions JSONL file."""
    sample_ids: List[str] = []
    anls_vals: List[float] = []
    vqa_vals: List[float] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = str(rec.get("sample_id") or idx)
            if "anls" in rec and "vqa_accuracy" in rec:
                anls = float(rec["anls"])
                vqa = float(rec["vqa_accuracy"])
            else:
                pred = rec.get("prediction", "")
                answers = rec.get("answers", [])
                anls = compute_anls(pred, answers)
                vqa = compute_vqa_accuracy(pred, answers)
            sample_ids.append(sid)
            anls_vals.append(anls)
            vqa_vals.append(vqa)

    return sample_ids, np.asarray(anls_vals, dtype=np.float64), np.asarray(vqa_vals, dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes percentile 95% bootstrap CI over 1D sample scores."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "half_width": 0.0, "n": 0}
    point = float(arr.mean())
    if n == 1 or n_bootstrap <= 0:
        return {"mean": point, "ci_low": point, "ci_high": point, "half_width": 0.0, "n": n}

    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(n_bootstrap), dtype=np.float64)
    chunk = 250
    for start in range(0, int(n_bootstrap), chunk):
        cur = min(chunk, int(n_bootstrap) - start)
        idx = rng.integers(0, n, size=(cur, n))
        boot_means[start : start + cur] = arr[idx].mean(axis=1)

    low = float(np.quantile(boot_means, alpha / 2.0))
    high = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return {
        "mean": round(point, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "half_width": round((high - low) / 2.0, 6),
        "n": int(n),
    }


def _align_paired_scores(
    ids_a: Sequence[str],
    vals_a: np.ndarray,
    ids_b: Sequence[str],
    vals_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(ids_a) == len(ids_b) and list(ids_a) == list(ids_b):
        return np.asarray(vals_a, dtype=np.float64), np.asarray(vals_b, dtype=np.float64)
    map_b = {sid: float(v) for sid, v in zip(ids_b, vals_b)}
    aligned_a, aligned_b = [], []
    for sid, va in zip(ids_a, vals_a):
        if sid in map_b:
            aligned_a.append(float(va))
            aligned_b.append(map_b[sid])
    return np.asarray(aligned_a, dtype=np.float64), np.asarray(aligned_b, dtype=np.float64)


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired sign-flip permutation test on aligned sample-level scores (A - B)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError(f"Aligned score arrays must be non-empty and equal length, got {len(a)} and {len(b)}.")

    diffs = a - b
    obs = float(diffs.mean())
    if np.allclose(diffs, 0.0):
        return {"delta": 0.0, "p_value": 1.0, "n": int(len(diffs))}

    rng = np.random.default_rng(int(seed))
    abs_obs = abs(obs) - 1e-12
    exceed = 0
    chunk = 500
    for start in range(0, int(n_permutations), chunk):
        cur = min(chunk, int(n_permutations) - start)
        signs = rng.choice((-1.0, 1.0), size=(cur, len(diffs)))
        perm_means = (signs * diffs[None, :]).mean(axis=1)
        exceed += int(np.sum(np.abs(perm_means) >= abs_obs))

    p_val = float((exceed + 1) / (int(n_permutations) + 1))
    return {"delta": round(obs, 6), "p_value": round(p_val, 6), "n": int(len(diffs))}


def _fmt_pct_ci(stat: Dict[str, float]) -> str:
    return f"{100.0 * stat['mean']:.2f} [{100.0 * stat['ci_low']:.2f}, {100.0 * stat['ci_high']:.2f}]"


def _fmt_latex_ci(stat: Dict[str, float]) -> str:
    return f"{100.0 * stat['mean']:.2f} $\\pm$ {100.0 * stat['half_width']:.2f}"


def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| Run | DocVQA ANLS (95% CI) | TextVQA VQA-Acc (95% CI) | Harmonic Mean | "
        "TG vs B1 ($p$) | FR vs B2 ($p$) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for r in rows:
        tg = f"{100.0 * r['tg_vs_b1']['delta']:+.2f} (p={r['tg_vs_b1']['p_value']:.4f})" if "tg_vs_b1" in r else "—"
        fr = f"{100.0 * r['fr_vs_b2']['delta']:+.2f} (p={r['fr_vs_b2']['p_value']:.4f})" if "fr_vs_b2" in r else "—"
        lines.append(
            f"| {r['label']} | {_fmt_pct_ci(r['docvqa_anls'])} | {_fmt_pct_ci(r['textvqa_vqa_accuracy'])} | "
            f"{100.0 * r['harmonic_mean']:.2f} | {tg} | {fr} |"
        )
    return "\n".join(lines) + "\n"


def render_latex_table(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Experiment & DocVQA ANLS (\\%) & TextVQA Acc (\\%) & Harmonic Mean & TG vs. B1 & FR vs. B2 \\\\",
        "\\midrule",
    ]
    for r in rows:
        tg = f"{100.0 * r['tg_vs_b1']['delta']:+.2f}" if "tg_vs_b1" in r else "--"
        fr = f"{100.0 * r['fr_vs_b2']['delta']:+.2f}" if "fr_vs_b2" in r else "--"
        lines.append(
            f"{r['label']} & {_fmt_latex_ci(r['docvqa_anls'])} & {_fmt_latex_ci(r['textvqa_vqa_accuracy'])} & "
            f"{100.0 * r['harmonic_mean']:.2f} & {tg} & {fr} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def run_suite(
    runs_root: Path,
    model: str,
    namespace: str,
    seed: int,
    eval_split: str,
    exps: List[str],
    few_shot_pcts: List[int],
    few_shot_exps: List[str],
    n_bootstrap: int,
    n_permutations: int,
    boot_seed: int,
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def _try_load(exp: str, pct: int) -> Optional[Dict[str, Any]]:
        run_dir = resolve_run_dir(runs_root, model, namespace, exp, seed, few_shot_pct=pct)
        doc_p = run_dir / f"preds_{eval_split}_docvqa.jsonl"
        tv_p = run_dir / f"preds_{eval_split}_textvqa.jsonl"
        if not (doc_p.exists() and tv_p.exists()):
            return None
        doc_ids, doc_anls, doc_vqa = load_jsonl_scores(doc_p)
        tv_ids, tv_anls, tv_vqa = load_jsonl_scores(tv_p)
        doc_stat = bootstrap_ci(doc_anls, n_bootstrap=n_bootstrap, seed=boot_seed)
        tv_stat = bootstrap_ci(tv_vqa, n_bootstrap=n_bootstrap, seed=boot_seed + 1)
        hm = (2.0 * doc_stat["mean"] * tv_stat["mean"]) / max(doc_stat["mean"] + tv_stat["mean"], 1e-8)
        return {
            "label": f"{exp} ({pct}%)" if pct < 100 else exp,
            "exp": exp,
            "few_shot_pct": pct,
            "run_dir": str(run_dir),
            "docvqa_anls": doc_stat,
            "docvqa_vqa_accuracy": bootstrap_ci(doc_vqa, n_bootstrap=n_bootstrap, seed=boot_seed + 2),
            "textvqa_anls": bootstrap_ci(tv_anls, n_bootstrap=n_bootstrap, seed=boot_seed + 3),
            "textvqa_vqa_accuracy": tv_stat,
            "harmonic_mean": round(float(hm), 6),
            "_raw": {"doc_ids": doc_ids, "doc_anls": doc_anls, "tv_ids": tv_ids, "tv_vqa": tv_vqa},
        }

    for exp in exps:
        item = _try_load(exp, 100)
        if item is not None:
            loaded[(exp, 100)] = item

    for exp in few_shot_exps:
        for pct in few_shot_pcts:
            if (exp, int(pct)) not in loaded:
                item = _try_load(exp, int(pct))
                if item is not None:
                    loaded[(exp, int(pct))] = item

    b1_full = loaded.get(("B1", 100))
    b2_full = loaded.get(("B2", 100))

    main_rows: List[Dict[str, Any]] = []
    for exp in exps:
        rec = loaded.get((exp, 100))
        if rec is None:
            continue
        if b1_full is not None and exp != "B1":
            a_tv, b_tv = _align_paired_scores(
                rec["_raw"]["tv_ids"], rec["_raw"]["tv_vqa"],
                b1_full["_raw"]["tv_ids"], b1_full["_raw"]["tv_vqa"],
            )
            if len(a_tv) > 0:
                perm = paired_permutation_test(a_tv, b_tv, n_permutations=n_permutations, seed=boot_seed)
                perm["tg_point"] = round(compute_transfer_gain(rec["textvqa_vqa_accuracy"]["mean"], b1_full["textvqa_vqa_accuracy"]["mean"]), 6)
                rec["tg_vs_b1"] = perm
        if b2_full is not None and exp != "B2":
            a_doc, b_doc = _align_paired_scores(
                b2_full["_raw"]["doc_ids"], b2_full["_raw"]["doc_anls"],
                rec["_raw"]["doc_ids"], rec["_raw"]["doc_anls"],
            )
            if len(a_doc) > 0:
                perm = paired_permutation_test(a_doc, b_doc, n_permutations=n_permutations, seed=boot_seed + 10)
                perm["fr_point"] = round(compute_forgetting_rate(b2_full["docvqa_anls"]["mean"], rec["docvqa_anls"]["mean"]), 6)
                rec["fr_vs_b2"] = perm
        main_rows.append(rec)

    fs_rows: List[Dict[str, Any]] = []
    for pct in sorted(set(int(p) for p in few_shot_pcts)):
        b1_pct = loaded.get(("B1", pct))
        for exp in few_shot_exps:
            src_rec = loaded.get((exp, pct))
            if src_rec is None:
                continue
            rec = {k: v for k, v in src_rec.items() if k != "_raw"}
            if b1_pct is not None and exp != "B1":
                a_tv, b_tv = _align_paired_scores(
                    src_rec["_raw"]["tv_ids"], src_rec["_raw"]["tv_vqa"],
                    b1_pct["_raw"]["tv_ids"], b1_pct["_raw"]["tv_vqa"],
                )
                if len(a_tv) > 0:
                    perm = paired_permutation_test(a_tv, b_tv, n_permutations=n_permutations, seed=boot_seed + pct)
                    perm["tg_point"] = round(
                        compute_transfer_gain(src_rec["textvqa_vqa_accuracy"]["mean"], b1_pct["textvqa_vqa_accuracy"]["mean"]), 6
                    )
                    rec["tg_vs_b1"] = perm
            if b2_full is not None and exp != "B2" and "fr_vs_b2" not in rec:
                a_doc, b_doc = _align_paired_scores(
                    b2_full["_raw"]["doc_ids"], b2_full["_raw"]["doc_anls"],
                    src_rec["_raw"]["doc_ids"], src_rec["_raw"]["doc_anls"],
                )
                if len(a_doc) > 0:
                    perm = paired_permutation_test(a_doc, b_doc, n_permutations=n_permutations, seed=boot_seed + 100 + pct)
                    perm["fr_point"] = round(
                        compute_forgetting_rate(b2_full["docvqa_anls"]["mean"], src_rec["docvqa_anls"]["mean"]), 6
                    )
                    rec["fr_vs_b2"] = perm
            fs_rows.append(rec)

    def _strip_raw(row_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "_raw"} for r in row_list]

    summary = {
        "model": model,
        "namespace": namespace,
        "seed": seed,
        "eval_split": eval_split,
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "main_results": _strip_raw(main_rows),
        "few_shot_results": _strip_raw(fs_rows),
    }
    (out_dir / "statistical_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if main_rows:
        (out_dir / "main_results_table.md").write_text(render_markdown_table(main_rows), encoding="utf-8")
        (out_dir / "main_results_table.tex").write_text(render_latex_table(main_rows), encoding="utf-8")
    if fs_rows:
        (out_dir / "few_shot_table.md").write_text(render_markdown_table(fs_rows), encoding="utf-8")
        (out_dir / "few_shot_table.tex").write_text(render_latex_table(fs_rows), encoding="utf-8")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.5, 4.0))
            for exp in few_shot_exps:
                pts = sorted(
                    [(r["few_shot_pct"], r["textvqa_vqa_accuracy"]) for r in fs_rows if r["exp"] == exp],
                    key=lambda x: x[0],
                )
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [100.0 * p[1]["mean"] for p in pts]
                    yerr = [100.0 * p[1]["half_width"] for p in pts]
                    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=exp)
            ax.set_xscale("log")
            ax.set_xticks(sorted(set(int(p) for p in few_shot_pcts)))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_xlabel("TextVQA Target Training Subset (%)")
            ax.set_ylabel("TextVQA VQA-Accuracy (%)")
            ax.set_title(f"Few-Shot Target Adaptation ({model}, {eval_split})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / "few_shot_learning_curve.png", dpi=160)
            plt.close(fig)
        except Exception:
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Phase D statistical evaluation suite.")
    parser.add_argument("--runs_root", default="TRAIN/ALT_THREE_SPLITS/runs")
    parser.add_argument("--model", default="smolvlm_500m")
    parser.add_argument("--namespace", default="ebs64_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_split", default="test", choices=["validation", "test"])
    parser.add_argument("--exps", nargs="+", default=["Z0", "B1", "B2", "B3", "B4"])
    parser.add_argument("--few_shot_pcts", nargs="+", type=int, default=[1, 5, 10, 25, 100])
    parser.add_argument("--few_shot_exps", nargs="+", default=["B1", "B4"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--boot_seed", type=int, default=42)
    parser.add_argument("--out_dir", default="EVALUATION/ALT_THREE_SPLITS/results")
    args = parser.parse_args()

    summary = run_suite(
        runs_root=Path(args.runs_root),
        model=args.model,
        namespace=args.namespace,
        seed=args.seed,
        eval_split=args.eval_split,
        exps=args.exps,
        few_shot_pcts=args.few_shot_pcts,
        few_shot_exps=args.few_shot_exps,
        n_bootstrap=args.n_bootstrap,
        n_permutations=args.n_permutations,
        boot_seed=args.boot_seed,
        out_dir=Path(args.out_dir),
    )
    print(f"Evaluated {len(summary['main_results'])} main runs and {len(summary['few_shot_results'])} few-shot runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
