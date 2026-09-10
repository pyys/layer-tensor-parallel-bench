# P104-100 8GB ×4 — derived metrics

Companion to [`p104-100-x4.md`](p104-100-x4.md). Formulas in
[METHOD section 5](../METHOD.md).

> ⚠️ **May be revised once the V100 16GB PCIe ×4 measurement lands.** The scaling
> efficiency in particular comes from a single platform, and its 1-GPU baseline is a
> substitution rather than a measurement.
>
> *[한국어: `p104-100-x4-metrics.ko.md`](p104-100-x4-metrics.ko.md) — English is
> authoritative.*

Source numbers: [`../results/p104-100-x4.jsonl`](../results/p104-100-x4.jsonl).
All values below are **means over the four warm turns**; cold is a single turn.

---

## 1. Session time and break-even

Session = one cold turn plus N warm turns.

| Config | Time |
|---|---|
| layer / off / 4 GPUs | 119.8 + **59.7N** s |
| layer / off / 2 GPUs | 157.7 + **57.9N** s |
| tensor / off / 4 GPUs | 156.6 + **28.1N** s |
| tensor / off / 2 GPUs | 174.2 + **36.9N** s |

```
break-even N = (cold_A − cold_B) / (warm_B − warm_A)
```

| Break-even | Turns |
|---|---|
| tensor 4 GPUs > layer 4 GPUs | **1.2** |
| **tensor 2 GPUs > layer 4 GPUs** | **2.4** |

Tensor parallelism loses the first turn because its cold prefill is slower. But it
**flips by the second or third message**, and the gap widens with every turn after
that.

The headline here: **two GPUs with tensor parallelism catch four with layer split in
about three turns.**

---

## 2. Tensor scaling efficiency

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

⚠️ **`tg_1` is not measured.** The model (13,115 MiB) does not fit on one 8GB card.
The **layer-split `tg` stands in for it** — layer split computes on one card at a
time (report section 3-1), so that value *is* single-card speed. **This is a
substitution.**

| N | tensor tg | layer tg (=1 GPU) | Speedup | **Efficiency** |
|---|---|---|---|---|
| 2 | 13.06 | 8.08 | 1.62x | **81%** |
| 3 | 14.14 | 7.97 | 1.77x | **59%** |
| 4 | 17.89 | 7.82 | 2.29x | **57%** |

Scaling holds well to two GPUs and degrades sharply beyond. N=3 is unusually poor,
plausibly because NCCL's ring/butterfly algorithms favour powers of two — **not
verified**.

Since memory bandwidth is not the bottleneck (section 4), the likely cause of the
falloff is the **PCIe Gen1 x4 interconnect**.

---

## 3. MTP batch economics

Whether speculative decoding pays is decided by the **cost multiplier of the
verification batch**.

```
steps        = n_predict − draft_n_accepted
cost/step    = predicted_ms(MTP on) / steps
cost/token   = predicted_ms(MTP off) / n_predict
multiplier   = cost/step ÷ cost/token
yield/step   = n_predict / steps
```

**If yield < multiplier, MTP is a loss.**

| | layer, 4 GPUs | tensor, 4 GPUs |
|---|---|---|
| MTP off, per token | 127.9 ms | 55.9 ms |
| MTP on, **per step** | 288.3 ms | 135.5 ms |
| **Cost multiplier** | **2.25** | **2.42** |
| Draft length | 2.97 | 2.98 |
| Yield per step | 1.89 tokens | 1.99 tokens |
| **Predicted loss** | **−16.2%** | **−18.1%** |
| **Measured loss** | **−16.3%** | **−17.2%** |

**Prediction and measurement agree to within 0.1–0.9 percentage points**, which is
the evidence that the model is sound.

### Break-even acceptance

```
draft length d = draft_n / steps
1 + d × acc = multiplier   ->   acc = (multiplier − 1) / d
```

| | layer, 4 GPUs | tensor, 4 GPUs |
|---|---|---|
| **Break-even acceptance** | **42.3%** | **47.8%** |
| Measured here (Mujeong fixture) | 29.9% | 33.1% |
| Observed in real usage | 47.9% | 47.9% |

In this measurement acceptance falls well short of break-even, hence the 16–18% loss.
Substituting the real-usage acceptance of 47.9% puts it **at best at break-even**.

⚠️ Acceptance is low here because `ignore_eos` forces 400 tokens past the point where
the model would stop (METHOD 1-5). So **"break-even in real usage" is an
extrapolation from the break-even formula, not a direct measurement.**

### Why this number matters

The speculative decoding literature assumes **batch 1 is memory-bound, so
verification is nearly free**. A cost multiplier of 2.25–2.42 means **that assumption
fails at batch 1 on this card**, which is direct evidence that it is **compute-bound**.

The same calculation applies to any llama.cpp run. All it needs is `draft_n`,
`draft_n_accepted` and `predicted_ms`.

---

## 4. Roofline

```
effective bandwidth = bytes the card reads ÷ the time that card spends reading them
```

The model is 12.81 GiB and is read once per token.

| Config | Derivation | Effective bandwidth | vs spec (320 GB/s) |
|---|---|---|---|
| layer, 4 GPUs | 3.20 GiB per card ÷ 32.0 ms (active for 1/4 of the 128 ms token) | 108 GB/s | **34%** |
| tensor, 4 GPUs | 3.20 GiB per card ÷ 55.9 ms (all four at once) | 62 GB/s | **19%** |

Both are under a third of spec. **Decode is not bound by memory bandwidth.**

Note that **layer split achieves the higher per-card bandwidth utilization** (34% vs
19%), because tensor parallelism inserts communication stalls. Total throughput is
still 2.3x higher with tensor parallelism, **because all four cards work at once.**

### Compared against Q6

| | Model size | tensor 4-GPU tg | Effective bandwidth |
|---|---|---|---|
| Q6_K | 20.88 GiB | 18.39 | 96 GB/s (30%) |
| Q3_K_M | 12.81 GiB | 17.89 | 62 GB/s (19%) |

**39% fewer bytes read, same speed, and lower bandwidth utilization.** That cannot
happen if bandwidth is the bottleneck.

⚠️ The Q6 value comes from llama-bench (short context) and the Q3 value from the
server (23k context), so the harnesses differ. This comparison alone is weak; the
batch economics in section 3 are the independent evidence.
