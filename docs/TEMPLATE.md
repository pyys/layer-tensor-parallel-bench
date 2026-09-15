# <platform> — layer split vs tensor parallelism

Measured YYYY-MM-DD. Methodology **METHOD v1.2** ([`../METHOD.md`](../METHOD.md)).
Raw data [`../results/<platform>.jsonl`](../results/).
Derived metrics live in [`<platform>-metrics.md`](<platform>-metrics.md).

> Measurement conditions are in METHOD. This document holds only **this platform's
> environment, results and interpretation.**
> Conclusions that span platforms are in [`../FINDINGS.md`](../FINDINGS.md).

A sentence or two on why this platform is worth measuring — what is unusual about it,
and **which conclusions from already-measured platforms it confirms or overturns.**

---

## 1. Environment

### Hardware

| Item | Value |
|---|---|
| GPU | |
| VRAM | per card / total / free |
| Memory bandwidth | spec figure |
| Power limit | Card specification. **If adjusted during measurement, say so and give the value** |
| CPU / RAM | |

**PCIe link** — paste the raw output. **Query `gen.max` and `width.max` as well and
state whether the reported generation and width are a ceiling or an idle
downshift**, and confirm under load.

```
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```

**Topology** — raw `nvidia-smi topo -m` output. Keep only the legend entries that
actually appear.

```

```

**P2P** — `nvidia-smi topo -p2p r`.

### Software

| Item | Value |
|---|---|
| OS | |
| Driver / CUDA | |
| llama.cpp | commit hash + build number. **If it is the same build as another platform, say so** |
| `CMAKE_CUDA_ARCHITECTURES` | |
| **NCCL** | version. **Without it this measurement is void** |
| Model | name and quantization (below) |

### Model

Source repository, source file, size, the exact `llama-quantize` command, the
resulting size and sha256, and why that quantization was chosen.

⚠️ Do not use markdown footnotes here — GitHub does not render footnotes containing
code blocks properly. Keep it as an ordinary subsection.

> ⚠️ **Without NCCL installed this measurement is void.** Symptoms, before-and-after
> figures and how to check are in [METHOD 1-2](../METHOD.md). **Keep this box.** Only
> if this platform behaves differently from what METHOD describes, note the difference
> here.

---

## 2. Results

- **Prefill (pp)** = how fast the prompt is read. `prompt_per_second`
- **Generation (tg)** = final output token rate. `predicted_per_second`,
  on confirmed output rather than drafts — `predicted_n` is 400 on every turn
- Cold is one run, warm is the mean of four turns
- **Percentages** are relative to **the slowest cell for that metric**, which is 100.
  Name the three baselines here

| Metric | Baseline cell (100%) | Value |
|---|---|---|
| Cold pp | | |
| Warm pp | | |
| tg | | |

### 2-1. Generation speed ranking (all cells)

**Put the ranking first.** What most readers came for is "which one is fastest".

| Rank | Cell | tg | % |
|---|---|---|---|
| 1 | | | |

Follow with roughly three takeaways. Cover at least:

- Whether the two split modes separate cleanly or interleave
- **Whether the cold pp ranking disagrees with the tg ranking**
- **Where the ranking differs from already-measured platforms**

### 2-2. Layer split, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | | |
| 3 | | | | | | | |
| 2 | | | | | | | |
| 1 | | | | | | | |

**N=1 must be measured whenever the model fits on one card** (METHOD 1-6). That makes
`tg_1` in the derived metrics a measurement rather than a derivation. If it does not
fit, leave the row empty and **state explicitly that a derived value is used.**

### 2-3. Layer split, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | | | | | | | | |
| 3 | | | | | | | | |
| 2 | | | | | | | | |

### 2-4. Tensor parallelism, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | | |
| 3 | | | | | | | |
| 2 | | | | | | | |
| 1 | | | | | | | |

### 2-5. Tensor parallelism, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | | | | | | | | |
| 3 | | | | | | | | |
| 2 | | | | | | | | |

⚠️ Tensor parallelism changes the generated content as N changes (METHOD 6-2), so
**do not read an acceptance difference as an effect of GPU count.**

### 2-6. Temperature (and throttling where available)

Aggregated from the telemetry in METHOD 1-8.

| Cell | Peak | Mean | therm % | pcap % |
|---|---|---|---|---|
| | | | | |

⚠️ **`0x1` in `clocks_throttle_reasons.active` is GpuIdle, not throttling.** The
aggregation criteria are in METHOD 1-8. **If measured with a pre-v1.2 harness, leave
this table empty and say so.**

### Validation checklist (METHOD section 4)

| Item | Result |
|---|---|
| No `NCCL not compiled in` | |
| No `internal AllReduce init failed` | |
| `prompt_n` matches the targets | |
| Cold `cache_n` = 0 | |
| Warm `cache_n` >= 22,000 | |
| `predicted_n` = 400 | |
| Warm `tg` spread within 5% (**MTP off cells only**) | |
| Thermal and power throttle aggregated | |
| No kernel Xid during the run | |

Record failed cells with the reason.

---

## 3. Interpretation

Figures and the calculations behind them belong in
[`<platform>-metrics.md`](<platform>-metrics.md). Conclusions that span platforms are
in [`../FINDINGS.md`](../FINDINGS.md).

**Put in the section headings which conclusions from other platforms this one confirms
and which it overturns.** For example: `3-2. Cold and warm reverse the ranking — same
as the other platform`.

Cover at least:

- Does layer-split decode get faster as N grows?
- Does the cold/warm prefill ranking reverse?
- How much of a warm turn is generation?
- Is decode bandwidth bound? (roofline, plus a Q6/Q3 comparison or a clock sweep where
  possible)
- Does speculative decoding pay? (batch cost multiple)

Separate what holds only here from what transfers elsewhere.

---

## 4. MTP variance

Warm 4-turn `tg` spread. Put MTP off and MTP on cells side by side. Per METHOD
section 4 the 5% rule does not apply to MTP on cells — **report the spread as a
result, not a failure.**

| Cell | Warm 4 turns | Spread |
|---|---|---|
| | | |

---

## 5. Failures and anomalies (only if applicable)

If there were crashes, voided cells or re-measured cells, put them here.

- **What failed, under what conditions.** How did the other cells at the same card
  count behave?
- **Causes ruled out**, as a table, with the figure behind each verdict
- **If you stopped digging, why.** Mark remaining hypotheses as undetermined
- **What it means in practice** — is it avoidable, and is there an alternative?

Keep the full account in [`../FINDINGS.md`](../FINDINGS.md) and link to it here.
**Do not write at length about a combination that was never going to be the best
choice anyway.**

---

## 6. Appendix — additional fixture (optional)

Only if a second fixture was run. State clearly whether a third party can reproduce
it, and **make sure no claim in the body depends on data that cannot be reproduced.**

⚠️ When comparing fixtures, remember that two fixtures usually differ along several
dimensions at once. **Do not attribute a difference to one factor unless it was
isolated.**

---

## 7. Limits specific to this platform

Do not repeat limits already stated in METHOD or FINDINGS. Cover at least:

- Is the single-card baseline measured or derived?
- Which cells are missing, and why? **Was any cell voided by a crash?**
- Which METHOD version was used? Are any recorded fields missing?
- Determinism caveats (for example, tensor parallelism changing the output as N
  changes)
- Any way the hardware departs from stock specification (modified cards, non-standard
  cooling, and so on)

---

## 8. Raw data and cell validity

| | Content |
|---|---|
| [`../results/<platform>.jsonl`](../results/) | One JSON line per cell |

### Cell validity

| Class | Cells | Basis |
|---|---|---|
| **Valid** | | |
| **Contaminated** | | |
| **Failed** | | |

**Contaminated and failed measurements are published, not discarded.** The reader has
to be able to check the basis for the judgement. Where a cell was re-measured, **show
both values side by side.**
