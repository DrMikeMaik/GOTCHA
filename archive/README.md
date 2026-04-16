# Archive

Experiments that were tried during the project but did not make the final cut.
They are kept here for reference, not because they are useful.

## Contents

- **diagnostic_attacks.py** — Four attack algorithms (`mean`, `stddev`,
  `delta_energy`, `pca1`) that were part of the original benchmarking suite.
  None of them recovered readable text. The only attack that worked was
  `block_flow_angle`, which remains in `attack_bench.py`.
