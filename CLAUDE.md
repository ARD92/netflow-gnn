# netflow-prototype: rules for Claude

## Never train models on a laptop

- Do not run model training on a developer laptop (macOS). This includes
  `python -m netflow_prototype train`, training experiments, seed sweeps, benchmarks,
  and any script that calls `netflow_prototype.train.train()`.
- Training runs only on the GPU server, for example:
  `python -m netflow_prototype train ... --device cuda`.
- To validate a change, use unit tests and code reasoning. When a real training run
  is needed, give the user the exact server command and ask for the log or metrics
  (`peak_gpu=`, `val_nll`, `metrics.json`).
- Exception: the pytest suite (`python -m pytest`) is allowed. Its end-to-end test
  trains a tiny model (300 flows, 3 epochs) in seconds.
