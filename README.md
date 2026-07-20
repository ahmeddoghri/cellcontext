# cellcontext

**Context-aware single-cell perturbation prediction, with a baseline honest enough to beat.**

cellcontext is a compact, inspectable implementation inspired by [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets.](https://www.nature.com/articles/s41592-025-02980-0).
It turns the paper's core idea into a deterministic benchmark that runs on a laptop with Python's standard library.

## Run it

```bash
python cellcontext.py
python -m unittest discover -s tests -v
```

The benchmark writes its result to stdout. Audio projects also write playable WAV files to `demo/`.

## What is tested

The test compares the research-inspired method with a deliberately legible baseline and requires
`mae_reduction_pct >= 30`. The data generator is seeded, so the number in this README,
CI, and the portfolio case study can be reproduced.

## Scope

This is an educational research reproduction on controlled synthetic data. It is not a clinical,
diagnostic, production genomics, copyright-authentication, or safety-critical system. The point is
to make one mechanism measurable without hiding it behind a checkpoint or API.

## Research basis

- [Nature Methods' 2025 benchmark of 27 perturbation-response methods across 29 datasets.](https://www.nature.com/articles/s41592-025-02980-0)
- Original implementation and benchmark in this repository are MIT licensed.

## License

MIT

## Reproduced result

| Metric | Value |
|---|---:|
| `global_mae` | **0.2699** |
| `context_mae` | **0.1813** |
| `mae_reduction_pct` | **32.8** |
