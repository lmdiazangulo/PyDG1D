# Illustrative 2D Examples

Standalone scripts for paper supplementary figures. These demonstrate that
the codebase runs 2D Maxwell solvers; they do **not** extend the 1D local
KYP stability analysis.

| Script | Description |
|--------|-------------|
| [`dg2d_pec_cavity.py`](dg2d_pec_cavity.py) | DGTD resonant cavity on Gambit mesh (PEC) |
| [`fd2d_te_pulse.py`](fd2d_te_pulse.py) | FDTD TE-mode Gaussian pulse on a square domain |

Run from the repository root:

```bash
python examples/dg2d_pec_cavity.py --output examples/output
python examples/fd2d_te_pulse.py --output examples/output
```

Use `--mesh Maxwell2D_K8.neu` for a faster (coarser) DG run. Default settings use
`Maxwell2D_K146.neu`, centered flux, and CFL=1.0 (validated in `test/test_dgtd_2d.py`).

See also [`docs/PAPER_2D_SUPPLEMENT.md`](../docs/PAPER_2D_SUPPLEMENT.md) for manuscript and response-letter text.
