# NVIDIA GTX 1080 Ti x4 on mixed PCIe Gen3 x4/x8 — derived metrics

Companion to [`gtx1080ti-x4.md`](gtx1080ti-x4.md). The formulas are in
[METHOD section 5](../METHOD.md).

Source figures are in [`../results/gtx1080ti-x4.jsonl`](../results/gtx1080ti-x4.jsonl), rows with
`validity: ok`. Every value below is the mean of four warm turns; cold is a single run.
**Main runs only** (two cards on GPUs 1,3 and three cards on 1,2,3; four cards on 0-3). ★ marks the
two invalid cells (layer / MTP off / 3 and 4 cards); where a formula needs them, the result is marked too.

**Each section puts P104 Gen1 x1 alongside** — the same host, driver and binary
([`p104-100-x4-gen1x1-metrics.md`](p104-100-x4-gen1x1-metrics.md)). The cards and the link changed
together, so card and link effects are mixed.

---

## 1. Session time and crossover

A session is one cold turn plus N warm turns.

| Configuration | 1080 Ti | P104 Gen1 x1 |
|---|---|---|
| layer / off / 4 cards ★ | 77.0 + 38.5N s | 128.8 + 64.7N s |
| layer / off / 2 cards | 98.0 + **37.0N** s | 166.1 + 62.9N s |
| tensor / off / 4 cards | **63.8** + **16.0N** s | 415.5 + 51.4N s |
| tensor / off / 2 cards | 87.5 + 22.2N s | — |

| Crossover | 1080 Ti | P104 Gen1 x1 |
|---|---|---|
| tensor 4 > layer 4 ★ | **−0.58** | 21.5 |
| tensor 2 > layer 4 ★ | 0.65 | 34.0 |
| tensor 4 > layer 2 | **−1.63** | — |
| tensor 2 > layer 2 | **−0.71** | — |

**Negative means tensor is already ahead on the cold turn.** Tensor four cards finishes the cold turn
(prefill plus 400 generated tokens) in 63.8 s against 77.0 s for layer four cards, even though layer's cold
**prefill** is faster (551.6 against 460.6 t/s) — the 400 generated tokens decide it. The comparisons
against layer four cards use an invalid cell; the comparisons against layer two cards do not, and agree.

---

## 2. Tensor scaling efficiency

`tg_1` = **12.45, layer / off / 2 cards, as a proxy.** METHOD section 5 prescribes the layer-split
average; layer three and four cards are invalid, so the one valid layer cell is used. The invalid cells
(12.20, 11.92) would give 12.19 as the average — a 2% lower baseline.

⚠️ **Different from the P104 reports**, which used the layer-split `tg` **at the same N** as each N's
baseline. Here the same N is invalid at three and four cards, so the two-card value serves every N.

| N | Tensor tg | Speedup | Efficiency | P104 Gen1 x1 efficiency | P104 Gen1 x4 efficiency |
|---|---|---|---|---|---|
| 2 | 20.83 | 1.67x | **83.6%** | 63% | 81% |
| 3 | 25.59 | 2.05x | **68.5%** | 47% | 59% |
| 4 | 28.94 | 2.32x | **58.1%** | 38% | 57% |

Four cards include the x4 device. The auxiliary runs give an idea of its cost: three cards with the x4
(0,1,2) decode 24.25 against 25.59 on 1,2,3 — 5% lower.

### 2-1. How this sits with the other platforms

Against P104 at Gen1 x4 the link is 4~8x wider and the four-card efficiency is the same (57% → 58%).
Against P104 at Gen1 x1 on the same host it rises from 38% to 58%. A link far wider than Gen1 x4 buys
nothing more, which fits Finding 5: past a point, scaling is not set by PCIe bandwidth.

---

## 3. MTP batch economics

```
steps          = n_predict - draft_n_accepted
cost per step  = predicted_ms(MTP) / steps
cost per token = predicted_ms(no MTP) / n_predict
cost multiple  = cost per step / cost per token
output / step  = n_predict / steps
```

| Cell | Acceptance | Draft length | Cost multiple | Output / step | Payoff | Break-even acceptance |
|---|---|---|---|---|---|---|
| layer / 4 | 29.1% | 2.97 | 2.230 | 1.865 | **−16.4%** | 41.5% |
| layer / 3 | 29.1% | 2.97 | 2.256 ★ | 1.865 | −17.3% | 42.3% |
| layer / 2 | 29.1% | 2.97 | 2.103 | 1.865 | −11.3% | 37.2% |
| tensor / 4 | 30.5% | 2.97 | 2.051 | 1.905 | **−7.1%** | 35.4% |
| tensor / 3 | 29.5% | 2.97 | 2.177 | 1.878 | −13.7% | 39.6% |
| tensor / 2 | 26.3% | 2.97 | 2.148 | 1.782 | −17.1% | 38.7% |

★ layer / 4 and layer / 3 divide by an invalid no-MTP cell.

### 3-1. Against P104

| | 1080 Ti | P104 Gen1 x1 | P104 Gen1 x4 |
|---|---|---|---|
| Cost multiple, layer | 2.10~2.26 | 2.26 | 2.25 |
| Cost multiple, tensor | 2.05~2.18 | 2.90 | 2.42 |

Layer split pays the same multiple on both cards. **Tensor pays less on the 1080 Ti** — the wider link
makes the extra draft step's all-reduce cheaper — but not enough for the 29% acceptance that `ignore_eos`
allows (METHOD 1-5).

---

## 4. Roofline

```
effective bandwidth = (model size / N) / time the card is actually busy
```

| Cell | Calculation | Share of 484 GB/s |
|---|---|---|
| tensor / off / 4 | 13,115 MiB / 4 × 28.94 t/s = 99.5 GB/s | **21%** |
| tensor / off / 2 | 13,115 MiB / 2 × 20.83 t/s = 143 GB/s | 30% |
| layer / off / 2 (each card busy half the time) | 6.88 GB × 12.45 × 2 = 171 GB/s | 35% |

**Not bandwidth bound**, as on every other platform.

---

## 5. Sensitivity to the link (auxiliary runs)

Same cell, x4 device included (first N) against left out (main run).

| Cell | Warm pp, x4 in → out | tg, x4 in → out |
|---|---|---|
| tensor / off / 2 | 235.8 → 250.9 (**+6.4%**) | 20.62 → 20.83 (+1.0%) |
| tensor / off / 3 | 267.1 → 286.2 (**+7.2%**) | 24.25 → 25.59 (+5.5%) |
| tensor / on / 2 | 208.8 → 222.2 (+6.4%) | 17.21 → 17.30 (+0.5%) |
| tensor / on / 3 | 232.4 → 249.8 (+7.5%) | 21.99 → 22.09 (+0.5%) |
| layer / off / 2 | 150.0 → 150.5 (+0.3%) | 12.97 → 12.45 (−4.0%) |
| layer / on / 2 | 140.7 → 139.7 (−0.7%) | 11.17 → 11.05 (−1.1%) |

Tensor prefill gains 6~8% from leaving out the one x4 link; layer split does not react. This is the same
direction as Finding 8, at a much smaller step (x4 → x8 on one card, not Gen1 x4 → x1 on all).
