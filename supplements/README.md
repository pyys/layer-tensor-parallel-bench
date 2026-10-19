# Supplementary runs

This folder holds **measurements outside the frozen methodology** — runs that change one
or more of the fixed items in METHOD section 1. They are not placed in the same tables as
the main matrix and are not used as evidence for the main findings
([METHOD section 7](../METHOD.md)). FINDINGS cites them only as **supplementary findings
(S1, S2, …)**, numbered apart from the main findings.

They are published anyway because **some questions can only be answered by changing a
fixed item.** "Do the conclusions hold for a different model?", for example, cannot be
answered by a matrix that fixes the model.

## Rules

1. The report states up front **which fixed items of METHOD it changed**
2. It includes **a table of METHOD's fixed items** — for each, the original value, the
   value in this run and the reason for the change
3. Everything not changed (cell definitions, request parameters, checklist, derived-metric
   formulas) follows METHOD as is
4. Raw data goes in `../results/supplements/`, and every line carries
   `"method": "<version>-supplement"` and a field naming what changed (e.g. `"model"`)
5. If the harness differs from the original, it goes in `../harness/supplements/` with the
   differences stated in its header
6. **A comparison is drawn as a conclusion only against a main run on the same system.**
   If the system differs too, the factors are mixed and no conclusion can be drawn

## Index

| Report | Fixed items changed | Paired main run | Measured |
|---|---|---|---|
| [Gemma 4 31B, V100 16GB x4 NVLink](gemma4-31b-v100-sxm2-16gb-x4-nvlink.md) | 1-1 model (+ MTP method), 1-4 `-c` | [V100 16GB x4 NVLink](../docs/v100-sxm2-16gb-x4-nvlink.md) (same system, same GPUs) | 2026-10-19 |

**There are also supplementary measurements outside this folder,** placed inside main
reports before this folder existed — the V100 clock sweep
([V100 metrics section 5](../docs/v100-sxm2-16gb-x4-metrics.md)) and the P104
production-settings re-check ([FINDINGS Notes and Caveats 3)](../FINDINGS.md#notes-and-caveats)).
They stay where they are.
