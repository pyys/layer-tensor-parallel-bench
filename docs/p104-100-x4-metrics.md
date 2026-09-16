# P104-100 8GB x4 — derived metrics

Companion to [`p104-100-x4.md`](p104-100-x4.md). The formulas are in
[METHOD section 5](../METHOD.md).

Source figures are in [`../results/p104-100-x4.jsonl`](../results/p104-100-x4.jsonl).
Every value below is the mean of four warm turns; cold is a single run.

**A V100 comparison is attached to each section.** Conclusions that span platforms are
in [`../FINDINGS.md`](../FINDINGS.md).

---

## 1. Session time and crossover

A session is one cold turn plus N warm turns.

| Configuration | Time |
|---|---|
| layer / off / 4 cards | 119.8 + **59.7N** s |
| layer / off / 2 cards | 157.7 + **57.9N** s |
| tensor / off / 4 cards | 156.6 + **28.1N** s |
| tensor / off / 2 cards | 174.2 + **36.9N** s |

```
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | Turns |
|---|---|
| tensor 4 > layer 4 | **1.2** |
| **tensor 2 > layer 4** | **2.4** |

Tensor parallelism has a slower cold prefill, so it loses the first turn. But **it
takes the lead by the second or third message**, and the gap widens every turn after.

**Two cards with tensor parallelism catch four cards with layer split in three turns** —
that is the point of this table.

### V100 comparison — the crossover comes sooner

| Crossover | P104 | **V100** |
|---|---|---|
| tensor 4 > layer 4 | 1.2 turns | **0.87** |
| tensor 2 > layer 4 | 2.4 turns | **0.12** |

**On V100 tensor wins from the very first warm turn**, because the cold gap shrank
from 36.8s to 5.5s. Details in
[V100 metrics section 1](v100-sxm2-16gb-x4-metrics.md).

---

## 2. Tensor scaling efficiency

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

⚠️ **`tg_1` is not measured.** The model (13,115 MiB) does not fit on one 8GB card.
**The layer-split `tg` stands in for it** — layer split computes on one card at a time
(section 3-1 of the report), so its figure is the single-card speed. **This is a
derived value** ([Notes and Caveats 2)](../FINDINGS.md#notes-and-caveats)).

| N | tensor tg | layer tg (=1 card) | Speedup | **Efficiency** |
|---|---|---|---|---|
| 2 | 13.06 | 8.08 | 1.62x | **81%** |
| 3 | 14.14 | 7.97 | 1.77x | **59%** |
| 4 | 17.89 | 7.82 | 2.29x | **57%** |

**Two cards hold up well; beyond that it degrades.**

### 2-1. The derivation was validated on V100

On V100 16GB the model fits on one card, so **the single-card figure could be
measured.**

| | Value |
|---|---|
| **1 card, measured** | **27.71** |
| layer, 2 cards | 27.97 (+0.9%) |
| layer, 3 cards | 27.78 (+0.2%) |
| layer, 4 cards | 27.29 (−1.5%) |

**All within 1.5%.** That supports the derivation used above. The `tg_1` for **this**
platform is still a derived value.

### 2-2. The cause of the decay — our earlier guess was wrong

**This document originally attributed the decay as follows.**

> Memory bandwidth is not the bottleneck (section 4), so the decay is presumed to come
> from the **PCIe Gen1 x4 interconnect**.

**That was wrong.** V100 runs the same model and the same binary with an interconnect
**16x wider, Gen1 x4 to Gen3 x16.** The decay should have eased. **It got worse across
the whole range.**

| N | P104 speedup | P104 efficiency | **V100 speedup** | **V100 efficiency** |
|---|---|---|---|---|
| 2 | 1.62x | 81% | 1.50x | **75%** |
| 3 | 1.77x | 59% | 1.54x | **51%** |
| 4 | 2.29x | 57% | **1.68x** | **42%** |

Bandwidth improved **2.8x more than compute did** (5.7x) and it still did not help.

**The cause is undetermined.** Per-step synchronization latency or a fixed cost in the
all-reduce itself are candidates, but this benchmark did not decompose the factors.
The discussion is in [FINDINGS 5](../FINDINGS.md).

---

## 3. MTP batch economics

Whether speculative decoding pays is decided by **the cost multiple of the
verification batch.**

```
steps          = n_predict - draft_n_accepted
cost per step  = predicted_ms(MTP on) / steps
cost per token = predicted_ms(MTP off) / n_predict
cost multiple  = cost per step / cost per token
tokens/step    = n_predict / steps
```

**If tokens per step is below the cost multiple, MTP is a loss.**

| | layer 4 cards | tensor 4 cards |
|---|---|---|
| MTP off, per token | 127.9 ms | 55.9 ms |
| MTP on, **per step** | 288.3 ms | 135.5 ms |
| **Cost multiple** | **2.25** | **2.42** |
| Draft length | 2.97 | 2.98 |
| Tokens per step | 1.89 | 1.99 |
| **Predicted loss** | **−16.2%** | **−18.1%** |
| **Measured loss** | **−16.3%** | **−17.2%** |

**Prediction and measurement agree within 0.1~0.9 percentage points.** The model
works.

### 3-1. Break-even acceptance

```
draft length d = draft_n / steps
1 + d x acc = cost multiple   ->   acc = (cost multiple - 1) / d
```

| | layer 4 cards | tensor 4 cards |
|---|---|---|
| **Break-even acceptance** | **42.3%** | **47.8%** |
| Measured here (Mujeong fixture) | 29.9% | 33.1% |
| Observed in real use | 47.9% | 47.9% |

This measurement falls well short of break-even, hence the 16~18% loss. Substituting
the real-use acceptance rate of 47.9% gives **a break-even at best.**

⚠️ Acceptance reads low here because `ignore_eos` forces 400 tokens and pushes
generation past where the model would stop (METHOD 1-5). The claim "break-even in real
use" is therefore **an extrapolation from the break-even calculation**, not a direct
measurement ([Notes and Caveats 7)](../FINDINGS.md#notes-and-caveats)).

### 3-2. Why this number matters

The speculative decoding literature assumes **batch 1 is bandwidth bound, so
verification is nearly free.** A cost multiple of 2.25~2.42 means **the premise fails
on this card even at batch 1**, which is direct evidence of **a compute bound.**

The same calculation applies to any llama.cpp run. All it needs is `draft_n`,
`draft_n_accepted` and `predicted_ms`.

### 3-3. V100 comparison — acceptance barely moves, the sign flips

| | **P104 layer 4** | **V100 layer 4** |
|---|---|---|
| MTP off, per token | 127.9 ms | **36.56 ms** |
| **Cost multiple** | **2.25** | **1.891** |
| Break-even acceptance | **42.3%** | **29.9%** |
| Measured acceptance | 29.9% | 32.0% |
| **Verdict** | **−16.3%** | **+3.5%** |

**Acceptance goes 29.9% → 32.0%, essentially unchanged. What moved is the break-even
line.**

In tensor mode on V100 it is still a loss (−12.1% at three cards, −5.9% at two) —
**on the same card the sign depends on the split mode.** Full tables in
[V100 metrics section 3](v100-sxm2-16gb-x4-metrics.md); the mechanism is in
[FINDINGS 4](../FINDINGS.md).

---

## 4. Roofline

```
effective bandwidth = bytes the card reads / time that card spends
```

The model is 12.81 GiB and is read once per token. Spec bandwidth is about 320 GB/s.

| Configuration | Calculation | Effective bandwidth | Against spec |
|---|---|---|---|
| layer 4 cards | 3.20 GiB per card / 32.0 ms (active for 1/4 of the 128 ms) | 108 GB/s | **34%** |
| tensor 4 cards | 3.20 GiB per card / 55.9 ms (four at once) | 62 GB/s | **19%** |

Both sit below a third of spec. **Decode is not bound by memory bandwidth.**

The interesting part is that **layer split has the higher per-card utilization** (34%
against 19%), because tensor parallelism inserts communication waits. Total throughput
is still 2.3x higher under tensor parallelism, **because four cards work at once.**

### Against Q6

| | Model size | tensor 4 tg | Effective bandwidth |
|---|---|---|---|
| Q6_K | 20.88 GiB | 18.39 | 96 GB/s (30%) |
| Q3_K_M | 12.81 GiB | 17.89 | 62 GB/s (19%) |

**39% fewer bytes read, same speed, and only the utilization drops.** That could not
happen if bandwidth were the bottleneck.

⚠️ The Q6 figure comes from llama-bench at low context and the Q3 figure from the
server at 23k context, so the harnesses differ. The comparison is weak on its own; the
batch economics in section 3 is the independent argument
([Notes and Caveats 4)](../FINDINGS.md#notes-and-caveats)).

### 4-1. V100 comparison

| Configuration | **P104** (spec 320 GB/s) | **V100** (spec 900 GB/s) |
|---|---|---|
| 1 card | — (not measurable) | 382 GB/s (**42%**) |
| layer 4 cards | 108 GB/s (**34%**) | 376 GB/s (**42%**) |
| tensor 4 cards | 62 GB/s (19%) | 161 GB/s (**18%**) |

**V100 uses more of its bandwidth** (34% → 42%). That shares a root with the cost
multiple dropping from 2.25 to 1.89 in section 3-3 — **being less compute bound makes
verifying a batch of four cheaper.** **It does not mean it became bandwidth bound.**
It is still below half.

**On V100 this verdict was confirmed directly by a sweep that lowered only the core
clock.** This platform has no clock or throttle telemetry (v1.1), so the same
measurement was not run.
→ [V100 metrics section 5](v100-sxm2-16gb-x4-metrics.md)
