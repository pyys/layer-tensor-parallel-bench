# P104-100 8GB x4 at PCIe Gen1 x1 — derived metrics

Companion to [`p104-100-x4-gen1x1.md`](p104-100-x4-gen1x1.md). The formulas are in
[METHOD section 5](../METHOD.md).

Source figures are in
[`../results/p104-100-x4-gen1x1.jsonl`](../results/p104-100-x4-gen1x1.jsonl), rows with
`validity: ok`. Every value below is the mean of four warm turns; cold is a single run.

**Each section puts Gen1 x4 alongside** — the same card model, the same model file and
the same llama.cpp commit ([`p104-100-x4-metrics.md`](p104-100-x4-metrics.md)). The host
changed as well, so the comparison is not a single-variable one
([Notes and Caveats 11)](../FINDINGS.md#notes-and-caveats)).

---

## 1. Session time and crossover

A session is one cold turn plus N warm turns.

| Configuration | Gen1 x1 | Gen1 x4 |
|---|---|---|
| layer / off / 4 cards | 128.8 + **64.7N** s | 119.8 + 59.7N s |
| layer / off / 2 cards | 166.1 + **62.9N** s | 157.7 + 57.9N s |
| tensor / off / 4 cards | 415.5 + **51.4N** s | 156.6 + 28.1N s |
| tensor / off / 2 cards | 398.1 + **56.8N** s | 174.2 + 36.9N s |

```
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | Gen1 x1 | Gen1 x4 |
|---|---|---|
| tensor 4 > layer 4 | **21.5** turns | 1.2 |
| tensor 2 > layer 4 | **34.0** turns | 2.4 |

**Tensor parallelism still wins every warm turn** — 51.4s against 64.7s. What it no
longer does is win quickly. Its cold turn is **286.7 seconds behind** (against 36.8 at
Gen1 x4), because a 23k-token cold prefill takes 380s under tensor parallelism and 76s
under layer split, while the per-turn advantage it has to repay that with shrinks from
31.6s to 13.3s.

At Gen1 x4 "two tensor-parallel cards beat four layer-split cards by the third
message". **At Gen1 x1 that takes 34 messages.** For a workload that keeps one long
conversation going it still pays; for anything shorter, layer split is ahead.

---

## 2. Tensor scaling efficiency

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

⚠️ **`tg_1` is derived.** The model does not fit on one 8GB card, so the layer-split
`tg` at the same N stands in for it, as at Gen1 x4
([Notes and Caveats 2)](../FINDINGS.md#notes-and-caveats)).

| N | tensor tg | layer tg (=1 card) | Speedup | **Efficiency** | Gen1 x4 efficiency |
|---|---|---|---|---|---|
| 2 | 9.58 | 7.61 | 1.26x | **63%** | 81% |
| 3 | 10.70 | 7.62 | 1.40x | **47%** | 59% |
| 4 | 11.16 | 7.40 | 1.51x | **38%** | 57% |

| | Gen1 x1 | Gen1 x4 |
|---|---|---|
| tensor, 2 → 4 cards | **1.16x** | 1.37x |

**Cutting the link to a quarter lowered efficiency at every N, and adding cards buys
less.** This is the first comparison in the repository where the link changed and the
card did not.

### 2-1. How this sits with the V100 result

[FINDINGS 5](../FINDINGS.md) showed that a 16x faster link (P104 Gen1 x4 → V100 Gen3
x16) did **not** improve tensor scaling. This platform shows that a 4x slower link on
the same card **does** worsen it. The two do not contradict each other — **bandwidth is
not what limits scaling above Gen1 x4, but below it, it becomes one of the limits.**
Where between x1 and x4 that changes was not measured.

---

## 3. MTP batch economics

```
steps          = n_predict - draft_n_accepted
cost per step  = predicted_ms(MTP on) / steps
cost per token = predicted_ms(MTP off) / n_predict
cost multiple  = cost per step / cost per token
tokens/step    = n_predict / steps
```

| | layer 4 | layer 3 | tensor 4 | tensor 3 | tensor 2 |
|---|---|---|---|---|---|
| MTP off, per token | 134.9 ms | 130.9 ms | 89.4 ms | 93.2 ms | 104.1 ms |
| MTP on, **per step** | 304.8 ms | 302.1 ms | 259.6 ms | 266.8 ms | 283.4 ms |
| **Cost multiple** | **2.26** | **2.31** | **2.90** | **2.86** | **2.72** |
| Draft length | 2.97 | 2.97 | 2.98 | 2.98 | 2.97 |
| Tokens per step | 1.89 | 1.89 | 1.99 | 2.08 | 1.86 |
| **Break-even acceptance** | **42.4%** | **44.0%** | **63.9%** | **62.6%** | **57.9%** |
| Measured acceptance | 29.9% | 29.9% | 33.1% | 36.2% | 29.1% |
| **Predicted** | −16.4% | −18.1% | −31.6% | −27.4% | −31.5% |
| **Measured** | **−16.4%** | **−18.1%** | **−30.6%** | **−23.0%** | **−31.4%** |

**Prediction and measurement agree within 1 percentage point in four of five cells.**
`tensor-mtp-3` misses by 4.4 points; its warm spread is 66.9%, the widest in the run.

### 3-1. Against Gen1 x4

| Cost multiple | Gen1 x1 | Gen1 x4 |
|---|---|---|
| layer 4 | 2.26 | 2.25 |
| tensor 4 | **2.90** | 2.42 |

**Layer split pays the same multiple** — verifying a batch of four does not send more
over the link. **Tensor parallelism pays 20% more**, because the verification batch adds
all-reduce traffic on a link that is already the bottleneck. The break-even line for
tensor 4 cards moves from 47.8% to 63.9%, while acceptance stays at 33.1%.

---

## 4. Roofline

```
effective bandwidth = bytes the card reads / time that card spends
```

The model is 12.81 GiB and is read once per token. Spec bandwidth is about 320 GB/s.

| Configuration | Calculation | Effective bandwidth | Against spec | Gen1 x4 |
|---|---|---|---|---|
| layer 4 cards | 3.20 GiB per card / 33.8 ms (active for 1/4 of 135 ms) | 102 GB/s | **32%** | 34% |
| tensor 4 cards | 3.20 GiB per card / 89.6 ms (four at once) | 38 GB/s | **12%** | 19% |
| tensor 2 cards | 6.41 GiB per card / 104.3 ms | 66 GB/s | 21% | 28% |

**Layer split uses its memory bandwidth as before; tensor parallelism uses less.** More
of each token's time goes to waiting on the link, so the card reads its weights more
slowly. Decode was not memory bound at Gen1 x4 and is further from it here.

---

## 5. Sensitivity to the link (Gen1 x4 → x1)

Every cell measured on both links.

| Cell | Cold pp | Warm pp | tg |
|---|---|---|---|
| layer-nomtp-4 | −7.5% | −19.6% | −5.4% |
| layer-nomtp-3 | −5.2% | −18.2% | −4.4% |
| layer-nomtp-2 | −4.8% | −19.2% | −5.8% |
| layer-mtp-4 | −11.9% | −26.4% | −5.5% |
| layer-mtp-3 | −9.1% | −26.2% | −6.1% |
| **tensor-nomtp-4** | **−64.6%** | **−63.4%** | **−37.6%** |
| tensor-nomtp-3 | −57.7% | −55.8% | −24.3% |
| tensor-nomtp-2 | −59.7% | −58.3% | −26.6% |
| tensor-mtp-4 | −64.8% | −63.4% | −47.8% |
| tensor-mtp-3 | −59.0% | −57.5% | −38.3% |
| tensor-mtp-2 | −60.5% | −59.4% | −37.3% |

**Layer split loses 4~6% of decode and 5~12% of cold prefill; tensor parallelism loses
24~48% of decode and 58~65% of prefill.** Layer split sends one activation across
each card boundary, tensor parallelism an all-reduce per layer per token.

**Warm prefill is the exception for layer split** — it loses 18~26%, three times its
cold-prefill loss. A likely reason is that a short prompt fills only one or two
micro-batches, so the transfer at each card boundary is no longer hidden behind
computation. **This was not tested.**

**Even so, tensor parallelism still decodes 1.51x faster than layer split at four
cards** (11.16 against 7.40), down from 2.29x. The link costs tensor parallelism most of
its lead, not the lead itself.
