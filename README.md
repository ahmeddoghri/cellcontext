# cellcontext

**Context-aware single-cell perturbation prediction, with a baseline honest enough to beat.**

Tell the same joke to eight different people and you get eight different reactions — a couple of laughs, a wince, one person who reports you to HR. A cell reacts to a perturbation the same way. Knock out the same gene at the same dose in two different cellular contexts and you get two different transcriptional responses, not one "true" effect with noise sprinkled on top. Most perturbation-response benchmarks average across the room and call the average ground truth. cellcontext keeps track of who's actually in the room.

It's a compact, inspectable implementation inspired by [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets](https://www.nature.com/articles/s41592-025-02980-0), rebuilt small enough to read in one sitting and run without a GPU, a checkpoint, or an API key.

## The result

```bash
python cellcontext.py
```
```json
{
  "global_mae": 0.2699,
  "context_mae": 0.1813,
  "mae_reduction_pct": 32.8
}
```

Average every training response per perturbation into one number and you get `global_mae`. Look up the nearest *observed* context and transfer its residual response instead and you get `context_mae` — a 32.8% drop in error on held-out context/perturbation pairs the model never trained on. Same data, same perturbations, the only difference is refusing to pretend every cell is the same cell.

## How it works

Training data is synthetic expression profiles built from a shared per-perturbation effect plus a genuine context-specific interaction term, so there's a real, known reason context should matter — this isn't noise dressed up as signal. The baseline averages every training response per perturbation into one global effect. The context-aware method indexes responses by context and, at test time, borrows the residual from the nearest context that's actually been observed for that perturbation. It's nearest-neighbor transfer, not a foundation model, and that's on purpose: the mechanism stays visible instead of hiding behind 200M parameters.

## Run it

```bash
python cellcontext.py
python -m unittest discover -s tests -v
```

## What is tested

The test compares the context-aware method against the deliberately legible global-average baseline and requires `mae_reduction_pct >= 30`. The data generator is seeded, so the number in this README, in CI, and in the portfolio case study are the same number, not three different ones that happen to rhyme.

## Scope

This is an educational research reproduction on controlled synthetic data. It is not a clinical, diagnostic, production genomics, or safety-critical system, and it makes no claim about real single-cell datasets. The point is to make one mechanism — don't average away the context — measurable without hiding it behind a checkpoint.

## Research basis

- [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets](https://www.nature.com/articles/s41592-025-02980-0)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT
