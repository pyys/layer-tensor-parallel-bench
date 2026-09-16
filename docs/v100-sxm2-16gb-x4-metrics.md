# V100-SXM2 16GB x4 — derived metrics

Companion to [`v100-sxm2-16gb-x4.md`](v100-sxm2-16gb-x4.md). The formulas are in
[METHOD section 5](../METHOD.md).

Source figures are in
[`../results/v100-sxm2-16gb-x4.jsonl`](../results/v100-sxm2-16gb-x4.jsonl).
Every value below is the mean of four warm turns; cold is a single run.

**On this platform `tg_1` is a measured value.** The 13,115 MiB model fits on one 16GB
card, so `layer-nomtp-1` was measured directly. Unlike the P104 report, the scaling
efficiency below rests on no derivation.

---

## 1. Session time and crossover

A session is one cold turn plus N warm turns, by wall clock.

| Configuration | Time |
|---|---|
| layer / off / 4 cards | 29.19 + **16.2N** s |
| layer / off / 2 cards | 35.42 + **15.9N** s |
| layer / off / 1 card | 45.19 + **16.0N** s |
| tensor / off / 4 cards | 34.68 + **9.9N** s |
| tensor / off / 2 cards | 29.85 + **10.8N** s |

```
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | Turns |
|---|---|
| tensor 4 > layer 4 | **0.87** |
| tensor 2 > layer 4 | **0.12** |
| tensor 4 > tensor 2 | **5.4** |

**Tensor wins from the very first warm turn.** P104 needed 1.2 and 2.4 turns because
its cold gap was large. Here the cold difference is 5.5s, so the crossover is almost
immediate.

**The third row matters in practice.** Card count changes the answer even within
tensor parallelism — **for any session shorter than about five turns, two cards beat
four.** Four cards start 4.8s behind on cold (34.68 against 29.85) and only recover
0.9s per warm turn.

⚠️ These figures assume **a four-turn mean warm time.** Longer sessions grow the KV
cache and the warm turn with it — linear extrapolation only holds for short sessions.

---

## 2. Tensor scaling efficiency

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

**`tg_1` = 27.71 t/s, measured (`layer-nomtp-1`).**

| N | tensor tg | Speedup | **Efficiency** |
|---|---|---|---|
| 1 | 27.45 | 0.99x | — |
| 2 | 41.64 | 1.50x | **75%** |
| 3 | 42.79 | 1.54x | **51%** |
| 4 | 46.62 | 1.68x | **42%** |

### 2-1. Against P104 — 16x the bandwidth, worse efficiency

| N | P104 speedup | P104 efficiency | **V100 speedup** | **V100 efficiency** |
|---|---|---|---|---|
| 2 | 1.62x | 81% | 1.50x | **75%** |
| 3 | 1.77x | 59% | 1.54x | **51%** |
| 4 | 2.29x | 57% | **1.68x** | **42%** |

**Interconnect bandwidth went up 16x from Gen1 x4 to Gen3 x16 and efficiency fell
across the whole range.** Bandwidth improved 2.8x more than compute did (5.7x) and it
still did not help.

With only P104 measured, this repository attributed the decay to **PCIe Gen1 x4**.
**That guess is refuted.** The cause is undetermined; per-step synchronization latency
or a fixed cost in the all-reduce itself are candidates. The discussion is in
[FINDINGS 5](../FINDINGS.md).

### 2-2. The P104 derivation was correct

On P104, `tg_1` was **substituted with the layer-split tg.** V100 allows that premise
to be checked against a measurement.

| | Value |
|---|---|
| **1 card, measured** | **27.71** |
| layer, 2 cards | 27.97 (+0.9%) |
| layer, 3 cards | 27.78 (+0.2%) |
| layer, 4 cards | 27.29 (−1.5%) |

**All within 1.5%.** Layer split computes on one card at a time, so its figure is the
single-card speed — the premise holds. **That raises confidence in the P104 efficiency
numbers.**

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

### 3-1. All five cells

Summed over four warm turns. `tensor-mtp-4` was never measured because of Xid 79.

| | layer 4 | layer 3 | layer 2 | **tensor 3** | **tensor 2** |
|---|---|---|---|---|---|
| Draft / accepted (4-turn sum) | 2440 / 780 | 2440 / 780 | 2440 / 780 | 2547 / 744 | 2599 / 723 |
| Steps per turn | 205 | 205 | 205 | 214 | 219.25 |
| Draft length | 2.976 | 2.976 | 2.976 | 2.975 | 2.964 |
| MTP off, per token | 36.56 ms | 35.91 ms | 35.67 ms | **23.31 ms** | **23.96 ms** |
| MTP on, **per step** | 69.14 ms | 67.87 ms | 66.32 ms | 49.70 ms | 46.59 ms |
| **Cost multiple** | **1.891** | **1.890** | **1.859** | **2.132** | **1.944** |
| Tokens per step | 1.951 | 1.951 | 1.951 | 1.869 | 1.824 |
| **Predicted** | **+3.2%** | **+3.2%** | **+5.0%** | **−12.3%** | **−6.2%** |
| **Measured** | **+3.5%** | **+3.4%** | **+5.2%** | **−12.1%** | **−5.9%** |

**Prediction and measurement agree within 0.3 percentage points in all five.** The
model works.

### 3-2. Break-even acceptance

```
draft length d = draft_n / steps
1 + d x acc = cost multiple   ->   acc = (cost multiple - 1) / d
```

| | layer 4 | layer 3 | layer 2 | **tensor 3** | **tensor 2** |
|---|---|---|---|---|---|
| **Break-even acceptance** | **29.9%** | **29.9%** | **28.9%** | **38.1%** | **31.9%** |
| Measured acceptance | **32.0%** | **32.0%** | **32.0%** | 29.2% | 27.8% |
| Verdict | **gain** | **gain** | **gain** | loss | loss |

**All three layer cells accept at exactly 32.0%**, with identical draft and accepted
counts — layer split is deterministic regardless of N (METHOD 6-2).

⚠️ Acceptance reads low here because `ignore_eos` forces 400 tokens and pushes
generation past where the model would stop (METHOD 1-5).

### 3-3. On the same card the split mode decides the sign

```
layer   cost multiple 1.89,  break-even 29.9%,  acceptance 32.0%   ->  gain
tensor  cost multiple 2.13,  break-even 38.1%,  acceptance 29.2%   ->  loss
```

**Two things move in the same direction.**

**(a) A higher cost multiple.** Tensor parallelism **makes single-token decode
cheaper** — per token cost drops from 35.9 to 23.3 ms while the cost of verifying four
does not drop as much, so **the denominator shrinks and the multiple grows.**

**(b) Lower acceptance.** Tensor parallelism changes the generated content as N
changes (METHOD 6-2). Layer split is fixed at `2440 / 780` while tensor gives
`2547/744` and `2599/723`. **Do not read that difference as an effect of GPU count.**

### 3-4. Against P104

| | P104 layer 4 | **V100 layer 4** |
|---|---|---|
| **Cost multiple** | 2.25 | **1.891** |
| Break-even acceptance | 42.3% | **29.9%** |
| Measured acceptance | 29.9% | 32.0% |
| Verdict | **−16.3%** | **+3.5%** |

**Acceptance goes 29.9% → 32.0%, essentially unchanged, and the sign flips.** What
moved is **the break-even line.** The mechanism is in [FINDINGS 4](../FINDINGS.md).

---

## 4. Roofline

```
effective bandwidth = (model size / N) / time the card actually spends
```

The model is 12.81 GiB and is read once per token. Spec bandwidth is 900 GB/s (HBM2).

| Configuration | Time the card spends | Effective bandwidth | Against spec |
|---|---|---|---|
| **1 card (measured)** | 36.00 ms | **382 GB/s** | **42%** |
| layer 4 cards | 1/4 of 36.56 ms = 9.14 ms | 376 GB/s | **42%** |
| tensor 2 cards | 23.96 ms (two at once) | 287 GB/s | 32% |
| tensor 3 cards | 23.31 ms (three at once) | 197 GB/s | 22% |
| tensor 4 cards | 21.40 ms (four at once) | 161 GB/s | **18%** |

**All below half of spec. Decode is not bound by memory bandwidth.**

**One card and layer split at four cards both land on 42%, exactly as the structure
predicts.** Layer split only serializes the same work, so per-card utilization should
match a single card. **The measurement confirms the structure.**

**Tensor parallelism loses utilization as cards are added** (32% → 22% → 18%), because
communication waits get interleaved. Total throughput is still highest at four cards
**because four cards work at once.** This is the efficiency decay of section 2 seen
from another angle.

### Against P104

| Configuration | P104 (spec 320 GB/s) | **V100 (spec 900 GB/s)** |
|---|---|---|
| layer 4 cards | 108 GB/s (**34%**) | 376 GB/s (**42%**) |
| tensor 4 cards | 62 GB/s (19%) | 161 GB/s (**18%**) |

**V100 uses more of its bandwidth** (34% → 42%). That shares a root with the cost
multiple dropping from 2.25 to 1.89 in section 3 — **being less compute bound makes
verifying a batch of four cheaper.** **It does not mean it became bandwidth bound.**
It is still below half.

---

## 5. SM clock sweep — what is decode actually bound by?

The roofline in section 4 shows **how much bandwidth is used**, not what causes what.
Lowering the clock directly measures the causation.

**V100 supports exactly one memory clock, 877 MHz, so bandwidth cannot be constrained.**
The inverse works instead — **lower only the core clock and pin the bandwidth.**

`layer-nomtp-1` was run on a single GPU with the clock fixed by `nvidia-smi -lgc`.
With no split and no communication, only the card's own behaviour remains.

| Requested | Actual clock | Clock ratio | **tg** | tg ratio | Time per token |
|---|---|---|---|---|---|
| 1530 | **1507** | 1.000 | **27.67** | 1.000 | 36.14 ms |
| 1230 | 1230 | 0.816 | 23.78 | 0.859 | 42.05 ms |
| 930 | 930 | 0.617 | 18.74 | 0.677 | 53.36 ms |
| 630 | 630 | 0.418 | 12.69 | 0.459 | 78.80 ms |
| 330 | 330 | 0.219 | **4.74** | 0.171 | 210.97 ms |

**Lowering the clock 4.57x lowered tg 5.84x.**

### 5-1. Slope by interval

Log-log slope, `d(ln tg) / d(ln clock)`. **1.0 means tg is proportional to clock;
0 means tg is independent of it.**

| Interval | Slope |
|---|---|
| 1507 → 1230 | **0.75** |
| 1230 → 930 | 0.85 |
| 930 → 630 | **1.00** |
| 630 → 330 | **1.52** |

### 5-2. 330 MHz behaves unlike the rest

A slope of 1.52 is **different in kind from the three intervals above it.** Cold
prefill bends at the same point — 125.0 / 752.9 = **0.166** against a clock ratio of
0.219.

**Two metrics deviating together is not measurement noise.** The cause was not
investigated.

### 5-3. The status of this measurement

**This is a side measurement outside the frozen methodology.** `nvidia-smi -lgc` is not
in the list of permitted variations in METHOD section 3. It is not placed in the same
table as the 12-cell matrix.

Clocks were restored with `-rgc` afterwards. Raw data is in
[`../results/clocksweep.jsonl`](../results/clocksweep.jsonl), one line per clock,
each carrying `validity: side` and **`locked_clock_mhz` / `actual_clock_mhz`**. The
actual clock is the modal value in the telemetry CSV over samples where GPU
utilization exceeded 50%.
