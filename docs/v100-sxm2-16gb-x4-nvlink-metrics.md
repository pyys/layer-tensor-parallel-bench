# V100-SXM2 16GB x4 (NVLink, DGX-1) — derived metrics

Companion to [`v100-sxm2-16gb-x4-nvlink.md`](v100-sxm2-16gb-x4-nvlink.md). The formulas
are in [METHOD section 5](../METHOD.md).

Source figures are in
[`../results/v100-sxm2-16gb-x4-nvlink.jsonl`](../results/v100-sxm2-16gb-x4-nvlink.jsonl).
Every value below is **the mean of the four warm turns**; cold is a single run.
**`tg_1` is measured** (`layer-nomtp-1`, 27.13 t/s).

"Earlier V100" means the same card model adapted from SXM2 to PCIe
([`v100-sxm2-16gb-x4-metrics.md`](v100-sxm2-16gb-x4-metrics.md)).

---

## 1. Session time and crossover

Session = one cold turn + N warm turns, wall time.

| Configuration | Time |
|---|---|
| layer / off / 4 cards | 30.26 + **16.49N** s |
| layer / off / 2 cards | 35.96 + **15.85N** s |
| layer / off / 1 card | 46.62 + **16.34N** s |
| tensor / off / 4 cards | **18.94** + **7.91N** s |
| tensor / off / 2 cards | 28.91 + **10.54N** s |
| tensor / on / 4 cards | 21.04 + **8.67N** s |

```
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | **NVLink** | Earlier V100 |
|---|---|---|
| tensor 4 > layer 4 | **−1.32 turns** | 0.87 |
| tensor 2 > layer 4 | **−0.23 turns** | 0.12 |
| tensor 4 > tensor 2 | **−3.78 turns** | 5.4 |

**All three are negative**, meaning A is already faster on the cold turn — there is no
crossover to wait for. Four-card tensor's cold turn takes 18.94s, 11.3s less than
four-card layer (30.26) and 10.0s less than two-card tensor (28.91).

**The earlier V100 conclusion "for sessions of five turns or fewer, two-card tensor beats
four-card tensor" does not hold here,** because what made four cards lose the cold turn —
tensor parallelism's cold prefill — is gone.

⚠️ These figures assume **the warm turn is the mean of four turns.**

---

## 2. Tensor-parallel scaling efficiency

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

**`tg_1` = 27.13 t/s (`layer-nomtp-1`, measured).**

| N | tensor tg | Speedup | **Efficiency** |
|---|---|---|---|
| 1 | 27.17 | 1.00x | — |
| 2 | 42.40 | 1.56x | **78%** |
| 3 | 48.31 | 1.78x | **59%** |
| 4 | 56.73 | 2.09x | **52%** |

### 2-1. Against the earlier V100

| N | Earlier V100 speedup | Earlier efficiency | **NVLink speedup** | **NVLink efficiency** |
|---|---|---|---|---|
| 2 | 1.50x | 75% | 1.56x | **78%** |
| 3 | 1.54x | 51% | 1.78x | **59%** |
| 4 | 1.68x | 42% | **2.09x** | **52%** |

**Efficiency rose at every N, and more so with more cards** (+3 / +8 / +10 points). The
two-to-four-card gain went from 1.12x to **1.34x**, close to the 1.37x of P104 at Gen1 x4.

### 2-2. The single-card stand-in — confirmed again

| | Value |
|---|---|
| **1 card, measured** | **27.13** |
| layer, 2 cards | 27.97 (+3.1%) |
| layer, 3 cards | 27.39 (+1.0%) |
| layer, 4 cards | 26.90 (−0.8%) |

**Within 3.1%.** A slightly wider range than the earlier V100 (within 1.5%), but METHOD
section 5's premise that the layer-split tg equals the single-card speed holds here too.

---

## 3. MTP batch economics

```
steps            = n_predict - draft_n_accepted
cost per step    = predicted_ms(MTP on) / steps
cost per token   = predicted_ms(MTP off) / n_predict
cost multiple    = cost per step / cost per token
tokens per step  = n_predict / steps
```

**If tokens per step is below the cost multiple, MTP is a loss.**

### 3-1. All eight cells

Sums over the four warm turns.

| | layer 4 | layer 3 | layer 2 | layer 1 | tensor 4 | tensor 3 | tensor 2 | tensor 1 |
|---|---|---|---|---|---|---|---|---|
| Drafted / accepted (4 turns) | 2440 / 780 | 2440 / 780 | 2440 / 780 | 2440 / 780 | 2616 / 719 | 2458 / 773 | 2599 / 723 | 2440 / 780 |
| Steps per turn | 205 | 205 | 205 | 205 | 220.25 | 206.75 | 219.25 | 205 |
| Draft length | 2.976 | 2.976 | 2.976 | 2.976 | 2.969 | 2.972 | 2.964 | 2.976 |
| MTP off, per token | 37.10 ms | 36.42 ms | 35.66 ms | 36.77 ms | **17.58 ms** | 20.65 ms | 23.52 ms | 36.72 ms |
| MTP on, **per step** | 67.36 ms | 66.53 ms | 64.85 ms | 67.53 ms | 33.48 ms | 38.41 ms | 44.63 ms | 69.26 ms |
| **Cost multiple** | **1.816** | **1.827** | **1.819** | **1.837** | **1.904** | **1.860** | **1.897** | **1.886** |
| Tokens per step | 1.951 | 1.951 | 1.951 | 1.951 | 1.816 | 1.935 | 1.824 | 1.951 |
| **Predicted** | +7.5% | +6.8% | +7.3% | +6.2% | −4.6% | +4.0% | −3.8% | +3.4% |
| **Measured** | **+7.6%** | **+7.0%** | **+7.6%** | **+6.6%** | **−4.5%** | **+4.1%** | **−3.6%** | **+3.7%** |

**Prediction and measurement agree within 0.4 points in all eight.** The cost model holds
on this platform too.

**All eight cost multiples fall between 1.82 and 1.90,** nearly regardless of split mode
and card count. On the earlier V100 they spread to 1.86~1.89 for layer against 1.94~2.13
for tensor — **on NVLink batch verification under tensor parallelism is no longer
expensive.**

### 3-2. Break-even acceptance

```
draft length d = draft_n / steps
1 + d x acc = cost multiple   ->   acc = (cost multiple - 1) / d
```

| | layer 4 | layer 3 | layer 2 | layer 1 | tensor 4 | tensor 3 | tensor 2 | tensor 1 |
|---|---|---|---|---|---|---|---|---|
| **Break-even acceptance** | **27.4%** | **27.8%** | **27.5%** | **28.1%** | **30.4%** | **28.9%** | **30.3%** | **29.8%** |
| Measured acceptance | 32.0% | 32.0% | 32.0% | 32.0% | 27.5% | 31.4% | 27.8% | 32.0% |
| Verdict | gain | gain | gain | gain | loss | gain | loss | gain |

**Break-even sits at 27~30% and measured acceptance at 27~32%,** so the sign turns on
small differences in acceptance. Tensor-cell acceptance differs by N because the generated
content differs (METHOD 6-2), so **do not read the three-card tensor gain as a property of
three cards.**

⚠️ Acceptance in this run is **lower than in real use**, because `ignore_eos` forces 400
tokens past where the model would stop (METHOD 1-5).

### 3-3. Against the earlier V100

| | Earlier V100 layer 4 | **NVLink layer 4** | Earlier V100 tensor 3 | **NVLink tensor 3** |
|---|---|---|---|---|
| **Cost multiple** | 1.891 | **1.816** | 2.132 | **1.860** |
| Break-even acceptance | 29.9% | **27.4%** | 38.1% | **28.9%** |
| Measured acceptance | 32.0% | 32.0% | 29.2% | 31.4% |
| Verdict | +3.5% | **+7.6%** | −12.1% | **+4.1%** |

**For layer, acceptance is unchanged and the multiple fell, doubling the gain. For tensor,
the multiple fell sharply from 2.13 to 1.86 and the sign flipped.** This fits the
all-reduce cost added to batch verification under tensor parallelism shrinking on NVLink.

---

## 4. Roofline

```
effective bandwidth = (model size / N) / time the card actually spends
```

Model 12.81 GiB, read once per token. Spec bandwidth is 900 GB/s (HBM2).

| Configuration | Time the card spends | Effective bandwidth | Against spec | Earlier V100 |
|---|---|---|---|---|
| **1 card (measured)** | 36.86 ms | **373 GB/s** | **41%** | 42% |
| layer 4 | 1/4 of 37.17 ms = 9.29 ms | 370 GB/s | **41%** | 42% |
| tensor 2 | 23.58 ms (both at once) | 292 GB/s | 32% | 32% |
| tensor 3 | 20.70 ms (all three at once) | 221 GB/s | 25% | 22% |
| tensor 4 | 17.63 ms (all four at once) | 195 GB/s | **22%** | 18% |

**One card and layer split match the earlier V100** (41% against 42%) — the card itself
behaves the same. **Tensor at three and four cards rises by 3~4 points** — less waiting on
communication, so the cards work that much more. This is the scaling-efficiency gain of
section 2 seen from another angle.

**Everything is below half of spec. Decode is not bound by memory bandwidth** — the same
conclusion as the earlier V100.
