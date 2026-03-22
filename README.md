# Quantum Robust Training

Install the packages listed in `requirements.txt`, then run:

```bash
python3 generate_nine_panel.py
```

Outputs:

- `outputs/qrt_only_nine_panel_bs5_steps120000.png`
- `outputs/qrt_only_nine_panel_bs5_steps120000.pdf`

The package also already includes those two pre-rendered output files.

The strict HHL entrypoint is:

```bash
python3 run_full04_hhl.py --dry-run
python3 run_full04_hhl.py --steps 10
```

This run is configured for the historical `0-4 / 12x12 / 10->4->5 / bs=5 / N=2` chain. The lifted solve dimension is `609180`, and the internal power-of-two padded dimension is `1048576`.

The package forces the MindQuantum HHL branch. If MindQuantum is unavailable, the solve dimension exceeds the configured dense limit, or the quantum branch raises, the run stops with an error instead of silently switching to a classical solver.
