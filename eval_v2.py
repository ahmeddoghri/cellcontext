"""Comparison: index-proximity context lookup vs. data-driven context lookup,
on data where context array index is decoupled from the latent value that
drives the true interaction effect (shuffle_identity=True)."""
import json

from adversarial import HOLDOUT_SEEDS, TUNING_SEEDS
from cellcontext_v2 import run, run_index_baseline


def summarize(seeds):
    index_reductions = [run_index_baseline(seed, shuffle_identity=True)["mae_reduction_pct"] for seed in seeds]
    data_reductions = [run(seed, shuffle_identity=True)["mae_reduction_pct"] for seed in seeds]
    return {
        "n": len(seeds),
        "index_mean_reduction_pct": round(sum(index_reductions) / len(seeds), 1),
        "data_mean_reduction_pct": round(sum(data_reductions) / len(seeds), 1),
        "index_worst": round(min(index_reductions), 1),
        "data_worst": round(min(data_reductions), 1),
    }


def main():
    print("cellcontext eval_v2: index-proximity vs. data-driven context similarity")
    print("(evaluated on shuffle_identity=True data -- index order no longer tracks true context similarity)")
    for label, seeds in (("tuning", TUNING_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        print(f"\n{label} ({len(seeds)} seeds):")
        print(json.dumps(summarize(seeds), indent=2))


if __name__ == "__main__":
    main()
