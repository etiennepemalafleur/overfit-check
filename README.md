# Backtest Overfitting Checker

A static, in-browser tool that audits a set of strategy return series (e.g. a
parameter sweep) for **backtest overfitting**, using the methods of Bailey &
López de Prado:

- **Deflated Sharpe Ratio (DSR)** — the probability the best strategy's true Sharpe
  exceeds a benchmark that is *inflated for the number of trials you tried*.
- **Probability of Backtest Overfitting (PBO)** via **CSCV** (Combinatorially
  Symmetric Cross-Validation) — how often the in-sample-best strategy lands below
  the out-of-sample median, across all C(S, S/2) train/test splits.
- **Minimum Backtest Length (MinBTL)** — the years of data you'd need before a given
  in-sample Sharpe, selected from N trials, is trustworthy.

Live: https://galacticchill.github.io/overfit-check/

Paste a return matrix (rows = periods, columns = strategies), upload a CSV, or load a
demo. Everything runs client-side — no upload, no key.

## Validation
`validate.py` is a reference implementation (identical seeded PRNG + algorithms). The
browser's numbers are checked against it exactly — PRNG output, per-strategy Sharpes,
DSR, PBO, and MinBTL all match to displayed precision on the bundled demos.

```bash
python3 validate.py   # requires only the stdlib
```

Educational — not investment advice.
