# Findings — what we found, and what we did not

Five runs on three kinds of card have been measured. The methodology is frozen in
[`METHOD.md`](METHOD.md); each platform's environment, results and
interpretation live in its own report.

| Platform | Interconnect | Measured | Report | Raw data |
|---|---|---|---|---|
| P104-100 8GB x4 | PCIe Gen1 x4 (~1 GB/s) | 2026-09-10 | [docs/p104-100-x4.md](docs/p104-100-x4.md) | [results/](results/) |
| V100-SXM2 16GB x4 | PCIe Gen3 x16 (~15.75 GB/s) | 2026-09-14~15 | [docs/v100-sxm2-16gb-x4.md](docs/v100-sxm2-16gb-x4.md) | [results/](results/) |
| P104-100 8GB x4 | **PCIe Gen1 x1** (~250 MB/s) | 2026-10-01~02 | [docs/p104-100-x4-gen1x1.md](docs/p104-100-x4-gen1x1.md) | [results/](results/) |
| V100-SXM2 16GB x4 | **NVLink** (DGX-1, 25.78 GB/s per link, 1~2 links per pair) | 2026-10-19 | [docs/v100-sxm2-16gb-x4-nvlink.md](docs/v100-sxm2-16gb-x4-nvlink.md) | [results/](results/) |
| GTX 1080 Ti 11GB x4 | **PCIe Gen3, mixed x4/x8** (one x4, three x8) | 2026-11-12 | [docs/gtx1080ti-x4.md](docs/gtx1080ti-x4.md) | [results/](results/) |

**Both platforms were measured with the same binary**
(`CMAKE_CUDA_ARCHITECTURES=61;70`). Everything except the GPUs was reused, so the
difference comes from the cards — but card, link, VRAM and power limit all changed
at once, and we did not decompose how much each contributed
[Notes and Caveats 6)](#notes-and-caveats).

**The Gen1 x1 run is the same P104 card model on a quarter of the link**, with the
same model file and the same llama.cpp commit (built for `61` only). The host CPU and
RAM changed as well, so the link is not the only difference there either
[Notes and Caveats 11)](#notes-and-caveats).

**The NVLink run is the same V100 card model in a factory SXM2 system (DGX-1).** The
cards in the earlier V100 run were SXM2 modules adapted to PCIe, with no NVLink. The
model file (sha256 identical) and the llama.cpp commit are the same, but the build was
separate (`70`) and the whole system — power delivery, cooling, host — changed with it
[Notes and Caveats 12)](#notes-and-caveats). Four GPUs of a shared eight-GPU system
were used [Notes and Caveats 13)](#notes-and-caveats).

**The GTX 1080 Ti run swaps the cards on the Gen1 x1 host** — same host, driver and binary
(`61`) as the Gen1 x1 run, and the same model file. The cards link at Gen3, x8 in three
slots and x4 in one, so devices were chosen under the asymmetric-device clause of METHOD
v1.5 and two layer-split cells failed the warm-spread check
[Notes and Caveats 15)](#notes-and-caveats).

**On all five runs the fastest token generation came from `tensor parallelism +
MTP off`** — including PCIe Gen1 x1, the slowest PCIe link there is. Conventional
advice is to use layer split with MTP on a slow interconnect; depending on the
conditions, that is not always right. The evidence is below.
[V100 results](docs/v100-sxm2-16gb-x4.md#2-results) ·
[P104 results](docs/p104-100-x4.md#2-results) ·
[P104 Gen1 x1 results](docs/p104-100-x4-gen1x1.md#2-results) ·
[V100 NVLink results](docs/v100-sxm2-16gb-x4-nvlink.md#2-results) ·
[1080 Ti results](docs/gtx1080ti-x4.md#2-results)

**There is also a supplementary run with a different model.** Gemma 4 31B was measured
the same way on the same NVLink system — tensor scaling efficiency did not depend on the
model, and the MTP result came out the opposite way
([Supplementary finding S1](#supplementary-findings--outside-the-frozen-method)).

---

## Where to start

| If you want to know | Start here |
|---|---|
| **Whether to use layer split or tensor parallelism** | [Finding 1](#1-layer-split-buys-no-decode-speed), [Finding 2](#2-two-cards-with-tensor-parallelism-beat-four-with-layer-split) |
| **Why your tensor-parallel numbers look wrong** | [METHOD 1-2 — NCCL](METHOD.md) |
| **Whether speculative decoding pays off on your card** | [Finding 4](#4-speculative-decoding-flips-sign-with-the-card-and-the-split-mode) |
| **How much a fourth card actually buys** | [Finding 5](#5-interconnect-bandwidth-and-tensor-parallel-scaling) |
| **Whether a slow link (x1 riser) rules out tensor parallelism** | [Finding 8](#8-split-modes-respond-oppositely-to-link-bandwidth) |
| **What changes on an NVLink system** | [Finding 3](#3-cold-and-warm-prefill-rank-the-split-modes-in-opposite-order--except-on-nvlink-and-at-gen1-x1), [Finding 5](#5-interconnect-bandwidth-and-tensor-parallel-scaling) |
| **Whether the conclusions hold for another model** | [Supplementary finding S1](#supplementary-findings--outside-the-frozen-method) |
| **How to measure this yourself** | [METHOD.md](METHOD.md) |
| **How to add a platform to this repository** | [docs/TEMPLATE.md](docs/TEMPLATE.md) |
| **What not to trust** | [Limits](#limits), [Notes and Caveats](#notes-and-caveats) |

---

## 1. Layer split buys no decode speed

The official guide says layer split is "tolerant of slow interconnects" while tensor
mode "prioritizes token generation" [{ref 1.}](#references). Here is what layer split
actually returns as cards are added.

| N | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **P104** tg | — | 8.08 | 7.97 | 7.82 |
| **P104 Gen1 x1** tg | — | 7.61 | 7.62 | 7.40 |
| **V100** tg | **27.71** | 27.97 | 27.78 | 27.29 |
| **V100 NVLink** tg | **27.13** | 27.97 | 27.39 | 26.90 |
| **1080 Ti** tg | — | 12.45 | 12.20 ★ | 11.92 ★ |

**All stay within about 4%. From two cards up, more cards is slightly slower.** The 1080 Ti
spans 4.3%, but its three- and four-card cells are invalid (★, warm spread over 5%
[Notes and Caveats 15)](#notes-and-caveats)).
Switching the interconnect from PCIe to NVLink changes nothing here — layer split
passes one activation across each card boundary, so the link has almost nothing to do.

**The single-card figure on V100 is measured.** The model (13,115 MiB) fits on one
16GB card. One card gives 27.71 and four give 27.29 — a 1.5% difference.
**Layer split uses four cards to deliver what one card does.**

The telemetry shows why — **one card at 100%, the rest at 0%, rotating in order.**
Token generation walks the layers in sequence, so only one card computes at a time.

### Prefill is different

| N | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| **V100 cold pp** | 747.4 | 1,091.1 | 1,435.3 | **1,555.5** |
| **V100 warm pp** | 488.4 | 490.9 | 492.7 | 487.6 |
| **V100 NVLink cold pp** | 720.7 | 1,062.7 | 1,412.9 | **1,533.9** |
| **V100 NVLink warm pp** | 472.5 | 485.9 | 488.5 | 473.8 |

Cold prefill throughput rises by up to **2.08x** (2.13x on NVLink), while warm prefill
barely moves.
A cold prompt has dozens of micro-batches, enough to fill the pipeline; a warm turn
(637~790 tokens) has one or two and cannot.

→ **What extra cards buy under layer split is (a) more VRAM and (b) faster handling
of long prefills.** "Long" here is about prompt token count, not cache state — we
measured 2.08x at 23k tokens and 0x at 700 tokens. Where in between the gain starts,
we did not measure.

---

## 2. Two cards with tensor parallelism beat four with layer split

Warm turn wall time.

| Configuration | P104 | V100 | P104 Gen1 x1 | V100 NVLink | 1080 Ti |
|---|---|---|---|---|---|
| layer, 4 cards | 59.7s | 16.2s | 64.7s | 16.5s | 38.5s ★ |
| **tensor, 2 cards** | **36.9s** | **10.8s** | **56.8s** | **10.5s** | **22.2s** |
| tensor, 4 cards | 28.1s | 9.9s | 51.4s | 7.9s | 16.0s |

**Half the cards, 1.6x faster on P104 and 1.5x on V100** — still 1.14x faster at
Gen1 x1, and 1.57x faster on NVLink.

Tensor parallelism has a slower cold prefill, so it loses the first turn. It does not
stay behind for long.

```
session = cold + warm x N
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | P104 | V100 | P104 Gen1 x1 | V100 NVLink | 1080 Ti |
|---|---|---|---|---|---|
| tensor 4 > layer 4 | 1.2 turns | **0.87** | **21.5** | **−1.32** | **−0.58** ★ |
| tensor 2 > layer 4 | 2.4 turns | **0.12** | **34.0** | **−0.23** | 0.65 ★ |

**On V100 tensor wins from the very first warm turn. At Gen1 x1 it takes 21.5 and 34
turns** — tensor parallelism still wins every warm turn, but its cold prefill is almost
five minutes behind (Finding 8). For a workload that keeps one long conversation going
it still pays; for shorter sessions layer split is ahead.

**On NVLink the crossover is negative** — tensor is already faster on the cold turn, so
there is nothing to wait for (Finding 3).

**On the 1080 Ti tensor four cards is also ahead from the cold turn** — not because its
cold prefill is faster (it is not, Finding 3) but because the 400 generated tokens of the
cold turn favour it. The layer four-card cell is invalid (★); against the valid layer
two-card cell the crossovers are −1.63 (tensor 4) and −0.71 (tensor 2), the same answer
([1080 Ti metrics section 1](docs/gtx1080ti-x4-metrics.md)).

### It also uses less memory

Measured under the author's production settings on P104 (Q6_K, `-c 90000`)
[Notes and Caveats 3)](#notes-and-caveats).

| | Per card (MiB) | Spread | Smallest headroom |
|---|---|---|---|
| Layer split, hand-tuned `-ts` | 7565 / 7253 / 7583 / 7699 | 446 | **410** |
| **Tensor parallelism** | **6823 x 4** | **0** | **1,286** |

**2.8GB less, and even on the first load.** The layer-split figures above took
**six restarts** to find, and the attempt before them failed to load at all. Once
found they stay put, so this is not a daily cost — it is **the cost of initial setup
and of every upgrade**, paid again whenever the model changes or a card is added.

---

## 3. Cold and warm prefill rank the split modes in opposite order — except on NVLink and at Gen1 x1

| | Cold (23,037 tok) | Warm (637~790 tok) |
|---|---|---|
| **P104** layer 4 | **327.3** | 85.3 |
| **P104** tensor 4 | 171.4 | **128.8** |
| **V100** layer 4 | **1,555.5** | 487.6 |
| **V100** tensor 4 | 887.4 | **594.1** |
| **P104 Gen1 x1** layer 4 | **302.6** | **68.6** |
| **P104 Gen1 x1** tensor 4 | 60.7 | 47.2 |
| **V100 NVLink** layer 4 | 1,533.9 | 473.8 |
| **V100 NVLink** tensor 4 | **1,968.2** | **903.8** |
| **1080 Ti** layer 4 ★ | **551.6** | 149.6 |
| **1080 Ti** tensor 4 | 460.6 | **343.4** |

| Ratio | P104 | V100 | P104 Gen1 x1 | V100 NVLink | 1080 Ti |
|---|---|---|---|---|---|
| Cold | layer ahead 1.91x | layer ahead 1.75x | **layer ahead 4.99x** | **tensor ahead 1.28x** | layer ahead 1.20x |
| Warm | tensor ahead 1.51x | tensor ahead 1.22x | **layer ahead 1.45x** | **tensor ahead 1.91x** | tensor ahead 2.30x |

**The direction holds on P104 at Gen1 x4, on V100 and on the 1080 Ti; only the magnitude
changes.**
**At Gen1 x1 warm prefill reverses as well** — the all-reduce per batch costs more than
tensor parallelism gains from splitting the work, even for a 700-token prompt.

**On NVLink it reverses the other way — tensor wins the cold prefill too.** On the same
V100 card model the four-card tensor cold pp went from 887.4 to 1,968.2 (**2.22x**), while
four-card layer split stayed where it was (1,555.5 → 1,533.9). We read this as the
per-batch all-reduce becoming cheaper on NVLink than the advantage of the layer pipeline.
**As the link widens from Gen1 x1 to Gen1 x4 to Gen3 x16 to NVLink, the cold layer lead
shrinks from 4.99x to 1.91x to 1.75x and then turns into a tensor lead** (the first two
steps are P104 and the last two V100, so the cards are mixed
[Notes and Caveats 12)](#notes-and-caveats)).

⚠️ This reversal **did not hold for another model.** On the same NVLink system Gemma 4
31B ran the cold prefill at the same rate under both splits (1,589.7 against 1,598.7) —
[Supplementary finding S1](#supplementary-findings--outside-the-frozen-method).

23k tokens is dozens of micro-batches, enough to fill the layer pipeline, while tensor
mode issues an all-reduce per batch. At 700 tokens there are one or two batches and
layer split cannot fill the pipeline — which is why **warm pp is flat in N**
(Finding 1).

→ **Do not pick a split mode from a synthetic prefill benchmark.** On the same
hardware `llama-bench` `pp 512` favours tensor (190 against 152) while the measured
23k cold prefill favours layer (171 against 327). **The two disagree completely**
[Notes and Caveats 4)](#notes-and-caveats).

### And warm turns are what users feel

This judgement assumes the author's usage pattern
[Notes and Caveats 9)](#notes-and-caveats).

Time split within a warm turn.

| | P104 layer 4 | V100 layer 4 | P104 Gen1 x1 layer 4 | V100 NVLink layer 4 |
|---|---|---|---|---|
| Prefill | 8.5s (14%) | 1.5s (10%) | 10.7s (17%) | 1.5s (9%) |
| **Generation** | **51.2s (86%)** | **14.7s (90%)** | **54.0s (83%)** | **14.8s (91%)** |

When the prompt cache hits, prefill drops from 23,037 tokens to roughly 700.
**Prefill goes from 91% of a cold turn to 10~14% of a warm one, and the workload
becomes generation dominated.**

Layer split is exactly the mode that cannot use extra cards during generation.

---

## 4. Speculative decoding flips sign with the card and the split mode

The standard account of speculative decoding runs like this
[{ref 7.}](#references) [{ref 8.}](#references) — **at batch size 1 the GPU is memory
bandwidth bound and mostly idle, so verifying a few draft tokens is nearly free.**
That premise is what makes speculative decoding work.

**The premise fails in all twenty-four measured conditions** (five more at Gen1 x1 in
4-1-1, eight more on NVLink in 4-1-2, six more on the 1080 Ti in 4-1-3). Even the lowest
cost multiple is **1.82x**. And in **seventeen of the twenty-four, MTP is a net loss.**

### 4-1. Measurements

| | P104 layer 4 | P104 tensor 4 | V100 layer 4 | V100 tensor 3 | V100 tensor 2 |
|---|---|---|---|---|---|
| MTP off, per token | 127.9 ms | 55.9 ms | 36.56 ms | 23.31 ms | 23.96 ms |
| MTP on, **per step** | 288.3 ms | 135.5 ms | 69.14 ms | 49.70 ms | 46.59 ms |
| **Cost multiple** | **2.25** | **2.42** | **1.891** | **2.132** | **1.944** |
| Draft length | 2.97 | 2.98 | 2.976 | 2.975 | 2.964 |
| Tokens per step | 1.89 | 1.99 | 1.951 | 1.869 | 1.824 |
| **Break-even acceptance** | **42.3%** | **47.8%** | **29.9%** | **38.1%** | **31.9%** |
| Measured acceptance | 29.9% | 33.1% | **32.0%** | 29.2% | 27.8% |
| **Predicted** | −16.2% | −18.1% | **+3.2%** | −12.3% | −6.2% |
| **Measured** | **−16.3%** | **−17.2%** | **+3.5%** | **−12.1%** | **−5.9%** |

**Prediction and measurement agree within 1 percentage point in all five.** The cost
model survives a change of platform and a change of split mode.

The formulas are in [METHOD section 5](METHOD.md). Full per-platform tables are in
each report.

#### 4-1-1. At Gen1 x1

| | layer 4 | layer 3 | tensor 4 | tensor 3 | tensor 2 |
|---|---|---|---|---|---|
| **Cost multiple** | **2.26** | **2.31** | **2.90** | **2.86** | **2.72** |
| **Break-even acceptance** | 42.4% | 44.0% | **63.9%** | 62.6% | 57.9% |
| Measured acceptance | 29.9% | 29.9% | 33.1% | 36.2% | 29.1% |
| **Predicted** | −16.4% | −18.1% | −31.6% | −27.4% | −31.5% |
| **Measured** | **−16.4%** | **−18.1%** | **−30.6%** | **−23.0%** | **−31.4%** |

**Four of five within 1 percentage point.** `tensor-mtp-3` misses by 4.4 points; its
warm spread is 67%, the widest in the run. Details in
[Gen1 x1 metrics section 3](docs/p104-100-x4-gen1x1-metrics.md).

#### 4-1-2. On NVLink

| | layer 4 | layer 3 | layer 2 | layer 1 | tensor 4 | tensor 3 | tensor 2 | tensor 1 |
|---|---|---|---|---|---|---|---|---|
| **Cost multiple** | **1.816** | **1.827** | **1.819** | **1.837** | **1.904** | **1.860** | **1.897** | **1.886** |
| **Break-even acceptance** | 27.4% | 27.8% | 27.5% | 28.1% | 30.4% | 28.9% | 30.3% | 29.8% |
| Measured acceptance | 32.0% | 32.0% | 32.0% | 32.0% | 27.5% | 31.4% | 27.8% | 32.0% |
| **Predicted** | +7.5% | +6.8% | +7.3% | +6.2% | −4.6% | +4.0% | −3.8% | +3.4% |
| **Measured** | **+7.6%** | **+7.0%** | **+7.6%** | **+6.6%** | **−4.5%** | **+4.1%** | **−3.6%** | **+3.7%** |

**All eight within 0.4 percentage points.** The cost multiple sits between 1.82 and 1.90
almost regardless of split mode and card count, so the sign turns on small differences
in acceptance — all four layer cells gain, and tensor loses at two and four cards, where
acceptance came out lower. Details in
[NVLink metrics section 3](docs/v100-sxm2-16gb-x4-nvlink-metrics.md).

#### 4-1-3. On the GTX 1080 Ti

| | layer 4 ★ | layer 3 ★ | layer 2 | tensor 4 | tensor 3 | tensor 2 |
|---|---|---|---|---|---|---|
| **Cost multiple** | **2.23** | **2.26** | **2.10** | **2.05** | **2.18** | **2.15** |
| **Break-even acceptance** | 41.5% | 42.3% | 37.2% | 35.4% | 39.6% | 38.7% |
| Measured acceptance | 29.1% | 29.1% | 29.1% | 30.5% | 29.5% | 26.3% |
| **Predicted** | −16.4% | −17.3% | −11.3% | −7.1% | −13.7% | −17.1% |
| **Measured** | **−16.4%** | **−17.5%** | **−11.2%** | **−7.0%** | **−13.7%** | **−16.9%** |

**All six lose, and all six within 0.2 percentage points.** ★ divides by an invalid
no-MTP cell. The tensor cost multiple (2.05~2.18) is lower than P104 at Gen1 x4 (2.42)
and x1 (2.90) — the wider link makes the draft step's all-reduce cheaper — but still above
V100. Details in [1080 Ti metrics section 3](docs/gtx1080ti-x4-metrics.md).

### 4-2. What sets the regime is not batch size

The rule of thumb in the literature — "below 50% acceptance it is a loss" — is usually
framed as a property of **batch size**. Here batch size is pinned at 1 and the sign
still flips. Three things decide it.

**(a) The card.** On P104 a batch of four costs **2.25x** a single token. On a memory
bandwidth bound card the weights are read once, so it should be nearly free.
**Cost rising with batch size is direct evidence of a compute bound.** On V100 the
multiple falls to 1.89x.

**(b) The split mode.** On the same V100, layer is **1.89x** and tensor with three
cards is **2.13x**. Tensor parallelism **makes single-token decode cheaper** — per
token cost drops from 36.6 to 23.3 ms while the cost of verifying four does not drop
as much, so **the denominator shrinks and the multiple grows.**

```
P104  acceptance 29.9%,  break-even 42.3%   ->  loss
V100  acceptance 32.0%,  break-even 29.9%   ->  gain     (layer)
V100  acceptance 29.2%,  break-even 38.1%   ->  loss     (tensor, 3 cards)
```

**(c) The link.** On the same P104, cutting the link from Gen1 x4 to x1 leaves the
layer-split multiple where it was (2.25 → 2.26) and raises the tensor-parallel one from
**2.42 to 2.90**. Verifying a batch adds all-reduce traffic, and on a narrow link that
traffic is what costs.

**The other direction holds too.** Moving the same V100 card model from PCIe to NVLink
drops the three-card tensor multiple from **2.13 to 1.86** and flips its sign (−12.1% →
+4.1%). The four-card layer multiple moves only a little, 1.89 → 1.82 (+3.5% → +7.6%).
**The link moves the tensor-parallel multiple and barely touches the layer-split one** —
whether it widens or narrows.

**Acceptance sits around 30% in all three. What moved is the break-even line.**

### 4-3. On other cards the same feature doubles throughput

| Source | Hardware | Acceptance | Speed |
|---|---|---|---|
| PR #22673 [{ref 2.}](#references) | — | — | about **2x** (1.2~2x range) |
| RTX PRO 6000 [{ref 3.}](#references) | RTX PRO 6000 | — | **1.73x** |
| sudoingX/qwen38-mtp [{ref 4.}](#references) | consumer GPUs | 0.797 / 0.732 | **2.07 / 2.33x** |
| **This work (P104)** | P104-100 x4 | 0.299 / 0.331 | **0.84 / 0.83x** |
| **This work (V100, layer)** | V100 16GB x4 | **0.320** | **1.035x** |
| **This work (V100, tensor)** | V100 16GB x3 | 0.292 | **0.879x** |
| **This work (P104, Gen1 x1)** | P104-100 x4 | 0.299 / 0.331 | **0.84 / 0.69x** |
| **This work (V100 NVLink, layer)** | V100 16GB x4 | **0.320** | **1.076x** |
| **This work (V100 NVLink, tensor)** | V100 16GB x4 | 0.275 | **0.955x** |

**The same feature doubles throughput on one card and loses on another.** That
contrast also lets the logic be read backwards — if 75% acceptance yields 2x, then
**on those cards verifying at batch 1 really is nearly free**, which is to say they
are **bandwidth bound** [Notes and Caveats 1)](#notes-and-caveats).

There is a public report in the other direction too — on an RTX 3090 with
Qwen3.6-35B-A3B every speculative mode was slower than baseline
[{ref 10.}](#references). That model is an MoE, so the cause may differ, and no batch
cost decomposition was done.

### 4-4. The method transfers

```
steps = n_predict - draft_n_accepted
```

Given `draft_n`, `draft_n_accepted` and `predicted_ms` you can compute the cost
multiple from **any llama.cpp run** and decide which regime your own card is in.

---

## 5. Interconnect bandwidth and tensor-parallel scaling

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

| N | P104 speedup | P104 efficiency (=speedup/N) | **V100 speedup** | **V100 efficiency (=speedup/N)** |
|---|---|---|---|---|
| 2 | 1.62x | 81% | 1.49x | **75%** |
| 3 | 1.77x | 59% | 1.54x | **51%** |
| 4 | 2.29x | 57% | **1.68x** | **42%** |

**Interconnect bandwidth went up 16x from P104 to V100 and tensor scaling efficiency
fell across the whole range.**

### This refutes our own earlier guess

With only P104 measured, this repository wrote:

> Memory bandwidth is not the bottleneck (section 4), so the decay is presumed to come
> from the **PCIe Gen1 x4 interconnect**.

**That was wrong.** Going from Gen1 x4 (~1 GB/s) to Gen3 x16 (~15.75 GB/s) is a 16x
increase and the decay did not ease. Bandwidth improved **2.8x more than compute did**
(5.7x) and it still did not help.

**The cause is undetermined.** Per-step synchronization latency or a fixed cost in the
all-reduce itself are candidates, but this run did not test them.

### Cutting the link on the same card does hurt

The Gen1 x1 run takes the other direction — **the same card model, a quarter of the
link.**

| N | Gen1 x4 efficiency | **Gen1 x1 efficiency** |
|---|---|---|
| 2 | 81% | **63%** |
| 3 | 59% | **47%** |
| 4 | 57% | **38%** |

| | Gen1 x4 | Gen1 x1 |
|---|---|---|
| tensor, 2 → 4 cards | 1.37x | **1.16x** |

**Efficiency fell at every N, and the fourth card buys less.** The two results do not
contradict each other: **more bandwidth than Gen1 x4 did not improve scaling, but less
than Gen1 x4 does worsen it.** Bandwidth is not what limits scaling on these cards at
Gen1 x4 and above; below that it becomes one of the limits. Where between x1 and x4
that changes was not measured. The host changed along with the link
[Notes and Caveats 11)](#notes-and-caveats). What the link does to each split mode as
a whole is in Finding 8.

> The same conclusion was reached independently for diffusion models — the cost of
> row split in sd.cpp did not improve across the same card swap. Different engine,
> different workload, different split implementation.
> → [`image-model-split-bench`](https://github.com/pyys/image-model-split-bench)

### Moving the same card to NVLink does improve scaling

**The same V100 card model, PCIe Gen3 x16 → NVLink.**

| N | PCIe V100 efficiency | **NVLink efficiency** | NVLink tg |
|---|---|---|---|
| 2 | 75% | **78%** | 42.40 |
| 3 | 51% | **59%** | 48.31 |
| 4 | 42% | **52%** | 56.73 |

| | PCIe V100 | NVLink |
|---|---|---|
| tensor, 2 → 4 cards | 1.12x | **1.34x** |

**Efficiency rose at every N, and more so with more cards.** `tg_1` is measured on both
(27.71 / 27.13). The two-card cell uses an NV1 pair (one link) and differs little from
PCIe; the gap opens at three and four cards.

**Read together with the two results above:** raising PCIe bandwidth 16x (Gen1 x4 →
Gen3 x16) did not improve scaling, cutting it to a quarter (→ Gen1 x1) worsened it, and
**moving to NVLink improved it.** That fits the reading that what decides scaling is not
the size of the PCIe bandwidth but **a direct GPU-to-GPU path that does not cross a host
bridge.** Of the candidates listed under "the cause is undetermined" above, it puts
weight on **per-step synchronization latency.** The whole system changed along with the
link, so **this is not settled** [Notes and Caveats 12)](#notes-and-caveats).

On the same NVLink system, switching the model to Gemma 4 31B left efficiency almost
unchanged at 77 / 62 / 53% —
[Supplementary finding S1](#supplementary-findings--outside-the-frozen-method).

### A much wider PCIe link than Gen1 x4 leaves efficiency where it was — on another card

The 1080 Ti run links at Gen3 x8 (four cards include one x4).

| N | P104 Gen1 x1 | P104 Gen1 x4 | **1080 Ti Gen3 x8** |
|---|---|---|---|
| 2 | 63% | 81% | **84%** |
| 3 | 47% | 59% | **69%** |
| 4 | 38% | 57% | **58%** |

Four-card efficiency is the same as P104 at Gen1 x4 on a link 4~8x wider — the same
reading as the PCIe V100 above. `tg_1` here is the layer two-card value standing in for
every N, because the same-N layer cells are invalid at three and four cards
[Notes and Caveats 15)](#notes-and-caveats). The cards changed along with the link.

### Whether tensor prefill rises or falls with the card count depends on the platform

| Tensor / MTP off, cold pp | 2 cards | 3 cards | 4 cards |
|---|---|---|---|
| P104, Gen1 x1 | 64.6 | 64.1 | 60.7 |
| P104, Gen1 x4 | 160.2 | 151.5 | 171.4 |
| V100, PCIe Gen3 x16 | 1,140.5 | 961.3 | **887.4** |
| V100, NVLink | 1,188.2 | 1,458.4 | **1,968.2** |
| **1080 Ti, Gen3 x8 (four cards include one x4)** | 337.0 | 398.8 | **460.6** |

**More cards slow tensor prefill on the PCIe V100 and speed it up on the 1080 Ti and on
NVLink**; P104 is flat. Adding a card cuts each card's share of the compute and adds
all-reduce traffic. We read the direction as set by the balance of the two — the V100 is
fast enough that communication dominates on PCIe, while the 1080 Ti computes far more
slowly on a link far wider than P104's. **This is a hypothesis; card speed and link were
not separated.** Warm prefill follows the same direction (1080 Ti 250.9 → 286.2 → 343.4).

---

## 6. On these cards decode is not bandwidth bound

```
effective bandwidth = (model size / N) / time the card actually spends
```

| Configuration | Effective bandwidth | Against spec |
|---|---|---|
| P104 layer 4 | 108 GB/s | **34%** (320) |
| P104 tensor 4 | 62 GB/s | 19% (320) |
| **V100 1 card (measured)** | **382 GB/s** | **42%** (900) |
| V100 layer 4 | 376 GB/s | 42% (900) |
| V100 tensor 4 | 161 GB/s | 18% (900) |
| P104 Gen1 x1 layer 4 | 102 GB/s | 32% (320) |
| P104 Gen1 x1 tensor 4 | 38 GB/s | **12%** (320) |
| V100 NVLink 1 card (measured) | 373 GB/s | 41% (900) |
| V100 NVLink layer 4 | 370 GB/s | 41% (900) |
| V100 NVLink tensor 4 | 195 GB/s | 22% (900) |
| 1080 Ti layer 2 | 171 GB/s | 35% (484) |
| 1080 Ti tensor 4 | 100 GB/s | 21% (484) |

**All of them sit below half of spec.** V100 at one card and at four under layer split
land on the same 42%, which is what the structure predicts — layer split only
serializes the same work, so per-card utilization should match a single card.
At Gen1 x1 tensor parallelism drops to 12%: more of each token's time goes to waiting
on the link. On NVLink it goes the other way, from 18% to 22%. One card and layer split
stay at the same 41~42% as on PCIe V100 — the card itself behaves the same, and what the
link changes is only how long tensor parallelism waits.

### Confirmed directly by lowering the clock

A roofline tells you **how much bandwidth is used**, not what causes what. On a single
V100 we lowered **only the core clock, leaving the memory clock pinned at 877 MHz**
[Notes and Caveats 3)](#notes-and-caveats).

| Actual clock | 1507 | 1230 | 930 | 630 | 330 |
|---|---|---|---|---|---|
| Clock ratio | 1.000 | 0.816 | 0.617 | 0.418 | 0.219 |
| **tg** | **27.67** | 23.78 | 18.74 | 12.69 | **4.74** |
| tg ratio | 1.000 | 0.859 | 0.677 | 0.459 | 0.171 |

Log-log slope by interval. **1.0 means tg is proportional to clock; 0 means tg is
independent of it.**

| Interval | 1507→1230 | 1230→930 | 930→630 | 630→330 |
|---|---|---|---|---|
| Slope | **0.75** | 0.85 | **1.00** | **1.52** |

**Lowering the clock 4.57x lowered tg 5.84x.**

⚠️ The 330 MHz interval behaves unlike the others. Cold prefill bends at the same
point (0.166 against a clock ratio of 0.219). **Two metrics deviating together is not
measurement noise.** The cause was not investigated. The full table is in
[V100 metrics section 5](docs/v100-sxm2-16gb-x4-metrics.md).

### Which is why quantization was not a speed lever on P104

| | Model size | tensor 4 tg |
|---|---|---|
| Q6_K | 20.88 GiB | 18.39 |
| Q3_K_M | 12.81 GiB | **17.89** |

**39% fewer bytes read, same speed.** Decode is normally memory bandwidth bound, which
is why dropping the quantization usually makes it faster. **On this card that premise
does not hold.** Quantization here is a capacity lever, not a speed lever.

⚠️ The two figures come from different harnesses
[Notes and Caveats 4)](#notes-and-caveats), but Finding 4 reaches the same conclusion
by an independent route.

⚠️ **No Q6 comparison was run on V100.** The quantization conclusion in this section
is **specific to P104.** On a bandwidth bound card, quantization is still a speed
lever.

---

## 7. One particular combination knocked GPUs off the bus

**On V100, `tensor parallelism + MTP + 4 cards` died both times it was run.**

```
NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
NVRM: Xid (PCI:0000:01:00): 154, GPU recovery action ... 0x1 (GPU Reset Required)
```

The other three four-card cells all completed.

| Cell | P104 | V100 | P104 Gen1 x1 | V100 NVLink |
|---|---|---|---|---|
| `layer-nomtp-4` | ok | ok | ok | ok |
| `layer-mtp-4` | ok | ok | ok | ok |
| `tensor-nomtp-4` | ok | ok | ok | ok |
| **`tensor-mtp-4`** | **ok (14.82 t/s)** | **crashed twice** | **ok (7.74 t/s)** | **ok (54.16 t/s)** |

**It is not "because it was four cards".** What is distinctive about this combination
is neither arithmetic nor power but **transaction frequency** — tensor parallelism
issues an all-reduce per token, and MTP draft verification rides on top, so **small
peer-to-peer transfers happen more often here than anywhere else.**

**And the same combination completed on P104**, where the link is 16x slower and
generation runs at a third the speed, so transactions per second were far lower.
**It completed again at Gen1 x1**, where they are lower still. Consistent with the
hypothesis.

**It also completed on the same V100 card model in the NVLink system** — generation is
faster than on PCIe V100 (54.16 t/s), but the small transfers between GPUs travel over
NVLink rather than PCIe. Also consistent with the hypothesis. That system is a factory
unit with different power delivery and cooling, though, and Xid could not be checked in
the kernel log [Notes and Caveats 13)](#notes-and-caveats).

### Causes ruled out

| Cause | Verdict |
|---|---|
| Overheating | **Ruled out.** 76 °C peak across 12 cells; 50 °C in the crashing cell |
| Sustained power | **Ruled out.** 60% of the cap. A separate workload held ~1,120W across four cards without incident |
| ECC / memory | **Ruled out.** Zero retired pages, no uncorrectable errors |
| A single faulty card | **Weak.** Three cards were lost at once on the second crash |

**The cause remains undetermined** [Notes and Caveats 5)](#notes-and-caveats).

### What it means in practice

**Do not use `tensor + MTP + 4 cards` on the PCIe-adapted V100 setup.** The fastest
configuration, `tensor-nomtp-4` at 46.62 t/s, has no MTP, and MTP is a loss in tensor
mode on that setup anyway (Finding 4). **Avoiding it costs nothing.**

---

## 8. Split modes respond oppositely to link bandwidth

**The same P104 card model, Gen1 x4 → Gen1 x1** — a quarter of the bandwidth.

| 4 cards | Cold pp | Warm pp | tg |
|---|---|---|---|
| **Layer split** | 327.3 → 302.6 (**−7.5%**) | 85.3 → 68.6 (−19.6%) | 7.82 → 7.40 (**−5.4%**) |
| **Tensor parallelism** | 171.4 → 60.7 (**−64.6%**) | 128.8 → 47.2 (−63.4%) | 17.89 → 11.16 (**−37.6%**) |

Across all eleven cells, **layer split loses 4~6% of decode and 5~12% of cold prefill;
tensor parallelism loses 24~48% of decode and 58~65% of prefill.** Layer split sends
one activation across each card boundary; tensor parallelism does an all-reduce per
layer per token, so its cost scales directly with the link. Warm prefill is the
exception for layer split — it loses 18~26%. The reason was not tested.

### Tensor parallelism still decodes faster at the slowest link

| 4 cards, tg | Gen1 x4 | Gen1 x1 |
|---|---|---|
| Layer split | 7.82 | 7.40 |
| Tensor parallelism | 17.89 | 11.16 |
| **Tensor / layer** | **2.29x** | **1.51x** |

Gen1 x1 is the narrowest link the PCIe specification defines. **The lead shrinks but
does not reverse.** Decode reads every weight once per token — gigabytes — while the
all-reduce traffic per token is far smaller, so splitting the reads across cards still
wins even when the link is cut to the minimum.

**What the link takes away is prefill.** At Gen1 x1 layer split reads a 23k-token cold
prompt **5.0x faster** (76s against 380s), and warm prefill reverses too (Finding 3).
That is why the session crossover moves from 1.2 turns to 21.5 (Finding 2).

→ **On x1 risers, tensor parallelism still gives the fastest generation, but only
pays off over long conversations that keep the prompt cache warm.** For many short or
cold prompts, layer split is ahead. MTP should stay off either way — at Gen1 x1 it
costs tensor parallelism 31% (Finding 4-1-1).

The full per-cell table is in
[Gen1 x1 metrics section 5](docs/p104-100-x4-gen1x1-metrics.md).

---

## Supplementary findings — outside the frozen method

This section reports observations from **supplementary runs that change a fixed item of
METHOD.** They are not used as evidence for Findings 1~8 and are numbered separately
with S. The status and rules of supplementary runs are in
[`supplements/README.md`](supplements/README.md) [Notes and Caveats 14)](#notes-and-caveats).

### S1. The model decides whether MTP pays; it does not decide tensor scaling

**Same NVLink system, same GPUs 0~3, same fixture text — only the model changed:**
Qwen3.8-27B Q3_K_M from the main run against Gemma 4 31B Q6_K from the supplementary run.
→ [Supplementary report](supplements/gemma4-31b-v100-sxm2-16gb-x4-nvlink.md)

**Tensor scaling efficiency — nearly identical**

| N | Qwen (main) | Gemma (supplementary) |
|---|---|---|
| 2 | 78% | 77% |
| 3 | 59% | 62% |
| 4 | 52% | 53% |

Model size (13 against 24 GiB), architecture (three quarters of Qwen's layers are
DeltaNet) and quantization all differ, yet the two stay within 3 points. Read with
Finding 5: **on these systems what set tensor scaling was the interconnect, not the
model.**

**MTP — opposite results**

| | Qwen cost multiple | **Qwen result** | Gemma cost multiple | **Gemma result** |
|---|---|---|---|---|
| layer 4 | 1.82 | **+7.6%** | **3.69** | **−41.2%** |
| layer 2 | 1.82 | +7.6% | 2.18 | −0.6% |
| tensor 4 | 1.90 | **−4.5%** | **1.77** | **+26.9%** |
| tensor 2 | 1.90 | −3.6% | **1.69** | **+27.0%** |

**Gemma's acceptance is nearly the same, 38.7~42.0%, both under tensor where it gains
and under layer where it loses.** What flips the sign is the **cost multiple** — Finding
4's "what moved is the break-even line" shows up between models as well. Qwen's multiple
stays at 1.82~1.90 regardless of split mode or card count; Gemma's spreads from 1.69 to
3.69.

**Hypothesis (not tested)** — the MTP structures differ. Qwen uses a one-layer head inside
the model; Gemma uses a separate four-layer draft model that shares the main model's KV.
Under layer split Gemma's per-step cost jumps from 94.9 to 158.1 ms between two and three
cards while the per-token cost stays put, which suggests that the cost of the draft path
crossing card boundaries grows with the card count.

**In practice** — **measure whether to enable MTP for each model.** A conclusion drawn on
one model (such as "on for layer, off for tensor") can be exactly reversed on another.
The cost multiple needs only `draft_n`, `draft_n_accepted` and `predicted_ms`
(Finding 4-4).

**Also observed** — Finding 3's "on NVLink tensor wins the cold prefill too" did not hold
for Gemma (cold pp 1,589.7 tensor against 1,598.7 layer). And Gemma's `tensor + MTP + 3
cards` aborted during load with a `GGML_ASSERT` — it occurs only when an uneven three-way
split meets draft KV sharing, and is described in section 6 of the report.

---

## Relation to prior work

This repository does not contradict the sources below; it fills a gap in them.

| Source | What it covers | What is missing |
|---|---|---|
| llama.cpp `docs/multi-gpu.md` [{ref 1.}](#references) | What each split mode is for, the NCCL requirement, the conceptual pipeline-vs-tensor trade-off | **No numbers.** No mention that the built-in AllReduce is two-GPU only |
| SharedLLM [{ref 5.}](#references), knightli [{ref 6.}](#references) | Practical multi-GPU setup, some two-card consumer figures | Two cards and fast PCIe only. No scaling curve, no cold/warm split, no MTP |
| Speculative decoding literature [{ref 7.}](#references) [{ref 8.}](#references) | Batch 1 is bandwidth bound so verification is nearly free; below 0.5 acceptance it is a loss | Frames the regime as **batch size**. Finding 4 is a counterexample where **the card and the split mode** set the regime at batch 1 |
| llama.cpp MTP measurements [{ref 2.}](#references) [{ref 3.}](#references) [{ref 4.}](#references) [{ref 9.}](#references) | MTP numbers on cards where it works (0.73~0.80 acceptance, 1.2~2.3x) | Only the winning side. Nothing decomposes **why it loses on some cards** |
| RTX 3090 speculative sweep [{ref 10.}](#references) | **A public case where MTP was slower than baseline** | An MoE model, so the cause may differ, and no batch cost decomposition |

If you know of prior measurements this duplicates, please open an issue.

---

## Limits

1. **One run per cell.** Four warm turns show within-cell variation, but run-to-run
   reproducibility was not measured
2. **One model, one quantization.** A different architecture — particularly a
   different full-attention ratio — may behave differently. A supplementary run did
   find the MTP result reversed on another model (Supplementary finding S1)
3. **`ignore_eos` depresses MTP acceptance.** Generation continues past where the
   model would stop, so acceptance reads lower than in real use
   [Notes and Caveats 7)](#notes-and-caveats)
4. **`tensor-mtp-4` is permanently missing on the PCIe-adapted V100** (Finding 7). It
   was measured on the NVLink system (−4.5%), but that is a different system and is not
   used to fill the gap
5. **The single-card baseline on P104 is still derived**, not measured. V100 only
   confirmed that the derivation is sound
   [Notes and Caveats 2)](#notes-and-caveats)
6. **No Q6 comparison on V100.** The quantization conclusion in Finding 6 is specific
   to P104
7. **The Gen1 x1 run changed the host along with the link**, so Findings 5 and 8 are
   not single-variable comparisons [Notes and Caveats 11)](#notes-and-caveats)
8. **Nine of the eleven Gen1 x1 cells were measured concurrently**, two cells at a
   time on separate GPUs [Notes and Caveats 10)](#notes-and-caveats)
9. **The NVLink run changed the whole system along with the interconnect.** The NVLink
   comparisons in Findings 3 and 5 are not single-variable comparisons
   [Notes and Caveats 12)](#notes-and-caveats)
10. **The NVLink run was made on a shared system, and Xid was not checked**
    [Notes and Caveats 13)](#notes-and-caveats)
11. **The 1080 Ti run's single-card baseline comes from one cell, two of its layer cells
    are invalid, and its devices were chosen under METHOD v1.5** — only the four-card cells
    include the x4 link, and cooling differs from the P104 runs
    [Notes and Caveats 15)](#notes-and-caveats)
12. Per-platform limits are in the last section of each report

---

## Notes and Caveats

**1)** The external figures in Finding 4-3 **were not measured with this
methodology.** Model, quantization, context and settings all differ, and some entries
do not publish an acceptance rate. **They are a reference, not a controlled
comparison.** Only the rows marked "This work" share a methodology, a model and a
binary.

**2)** The single-card baseline differs in kind between platforms. **On V100 it is
measured** — the 13,115 MiB model fits on one 16GB card. **On P104 it is derived** —
the model does not fit on one 8GB card, so **the layer-split tg stands in for it**
(the premise being that layer split computes on one card at a time, so its figure is
the single-card speed). **V100 confirmed that premise to within 1%** (measured 27.71
at one card against 27.97 / 27.78 / 27.29 for layer split at two, three and four).
Until that check, the P104 efficiency figures in Finding 5 rested on the derivation.

**3)** Two measurements sit **outside the frozen methodology**. Neither is placed in
the same table as the 12-cell matrix.

**The VRAM table and session re-check in Finding 2** — the matrix ran at Q3_K_M /
`-c 27000` with no `-ts`, while that table uses the author's production settings of
**Q6_K / `-c 90000`** and a hand-tuned `-ts`. The point is to check whether the matrix
conclusion carries over to production conditions.

**The clock sweep in Finding 6** — `nvidia-smi -lgc` is not in the list of permitted
variations in METHOD section 3. It was run on a single V100 and clocks were restored
with `-rgc` afterwards. Raw data is in
[`results/clocksweep.jsonl`](results/clocksweep.jsonl), one line per clock, each
marked `validity: side`.

**4)** Findings 3 and 6 each contain **a comparison across different harnesses.**
`llama-bench` is a low-context synthetic benchmark while this work reads the server's
`/completion` response at 23k context. **Neither comparison stands on its own.**
Finding 6 is backed independently by Finding 4, and Finding 3 by the measured
cold/warm figures.

**5)** The Xid 79 in Finding 7 has **no confirmed cause.** The remaining hypothesis is
PCIe transaction frequency; it fits the circumstances but was not tested.
`NCCL_P2P_DISABLE=1` would separate the cases, but doing so means running the same
workload again on a configuration that has already died twice. **We judged the risk
not worth the information and stopped.**
⚠️ **Microsecond-scale current transients are not visible to `nvidia-smi`.** Low
sustained power does not fully rule out a power delivery problem.

**6)** Between the two platforms, **card, link, VRAM and power limit all changed at
once.** We did not decompose them — what this repository measures is "what changes
when you physically swap the cards", not a controlled factor analysis. The V100 units
are also **SXM2 modules adapted to PCIe**, so power delivery and cooling differ from
factory PCIe cards.

**7)** The acceptance rates in Finding 4 come from `ignore_eos: true` with
`n_predict 400` (METHOD 1-5). Generation runs past the point where the model would
stop, so acceptance is **lower than in real use.** Any statement about real-world
behaviour is therefore **an extrapolation from the break-even calculation**, not a
direct measurement.

**8)** **Throttle data could not be collected on V100.** The telemetry query in
METHOD v1.1 did not include `clocks_throttle_reasons.active`. Only temperatures
survive (42~76 °C). The field was added in v1.2, so later measurements are unaffected.

**9)** "Warm turns dominate" in Finding 3 means **they dominate in the author's usage
pattern** — long context held across many turns, with the prompt cache hitting.
For workloads where the cache does not hit — one-shot queries, RAG that inserts a
different document each time, batch processing — **cold prefill keeps dominating**, and
the conclusion shifts toward layer split. The prompt design is in METHOD 1-3.

**10)** **Nine of the eleven Gen1 x1 cells were measured concurrently** — two cells at a
time on disjoint GPUs of the six installed. **The same cells were first run both alone
and concurrently, the difference was confirmed to be small, and only then was the rest
of the matrix run concurrently.** The largest difference was 2.1% (cold prefill of
`layer-nomtp-2`); decode differed by 1.7% at most. The comparison, the pairing and the
GPUs used are in the [Gen1 x1 report section 5-1](docs/p104-100-x4-gen1x1.md); the
alone runs are kept in the raw data as `validity: side`. METHOD v1.3 permits this,
conditional on that check.

**11)** **The Gen1 x1 run is not a single-variable comparison with Gen1 x4.** The card
model, the model file (sha256 identical) and the llama.cpp commit are the same, but the
host changed with the link — EPYC 7232 / 32GB to i7-5820K / 16GB — and so did the
driver (580.173 → 580.178) and the CUDA architecture list (`61;70` → `61`). Prefill and
decode are GPU bound and the CPU only tokenizes and samples, so the link is taken as the
main factor, **but it was not isolated.** The earlier report did not record card UUIDs,
so the physical units cannot be shown to be the same.

**12)** **The NVLink run is not a single-variable comparison with the PCIe V100 run.**
The card model, the model file (sha256 identical) and the llama.cpp commit are the same,
but the physical cards differ — SXM2 modules adapted to PCIe on one side, factory SXM2
(DGX-1) on the other. Power delivery and cooling, the host (EPYC 7232 / 32GB → Xeon
E5-2698 v4 x2), the driver (580.173 → 580.178) and the build (`61;70` → `70`) changed
with it. Reading the differences in Findings 3 and 5 as NVLink is **a judgement that the
interconnect is the largest change**, not an isolated result. GPUs 0~3 of the DGX-1 also
have one or two NVLinks depending on the pair. The two-card cells use a one-link pair; a
supplementary run of the same cell on a two-link pair differed by 0.1% in decode.

**13)** **The NVLink run used four GPUs of a shared eight-GPU system** (METHOD v1.4
section 3). Devices were selected by UUID and telemetry was limited to those four.
**The kernel log (`dmesg`) records the whole system, including events on other users'
devices, so it was not read — the Xid item of the section 4 checklist is therefore
"not checked".** The server logs of all sixteen cells show no GPU errors and every cell
completed. Other work may have been running on the remaining four GPUs, and the host
CPU, memory and PCIe root are shared. Also, the warm spread of `layer-nomtp-4` is 5.01%,
0.01 points over the METHOD limit. It is reported as a borderline value rather than
re-measured — its tg is in the same range as the other layer cells on that platform.

**14)** Supplementary finding S1 **sets a supplementary run against a main run.** The
supplementary run changed the model (METHOD 1-1) and the context (1-4, `-c 28000`),
because the same text is about 6% longer under Gemma's tokenizer and does not fit in
27000. Quantization (Q6_K against Q3_K_M), MTP structure (separate draft model against
built-in head) and the sampling defaults (top_k 64 against 20) also differ. **"The model
decides whether MTP pays" is an observation that two models gave opposite results; which
property of the model is responsible was not isolated.** The twelve supplementary cells
were measured in two sessions under the same conditions; how is set out in section 7 of
the report.

**15)** **The GTX 1080 Ti run.** (a) **Device selection** — the system mixes one Gen3 x4
link and three x8 links. Under the METHOD v1.5 asymmetric-device clause the main runs use
the x8 devices at two cards (GPUs 1,3) and three cards (1,2,3); four cards cannot avoid
the x4. The first-N runs (0,1 and 0,1,2) are published as auxiliary runs; leaving out the
x4 raised tensor prefill by 6~8% and left layer split unchanged. (b) **Invalid cells** —
`layer-nomtp-4` (warm spread 6.7%, re-run 7.2%) and `layer-nomtp-3` (12.6%) fail the 5%
check. The same cell passed at 1.5% on the auxiliary device set, so we read it as a
run-to-run effect of idle cards' clocks moving between 1,480 and ~1,800 MHz under layer
split, **not established.** The values are shown but excluded from rankings and derived
metrics. (c) **Baseline** — `tg_1` is the layer two-card value for every N, not the same-N
layer value used on P104. (d) **Comparisons** — the host, driver and binary match the Gen1
x1 run, but the link changed with the cards; against Gen1 x4 the host, driver and build
differ as well. A user fan-curve daemon kept thermal throttling at about 0%, unlike the
P104 runs on the default curve.

---

## References

{ref 1.} *llama.cpp — Multi-GPU support* (`docs/multi-gpu.md`).
https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md (accessed 2026-09-10)

{ref 2.} *llama.cpp Pull Request #22673 — MTP speculative decoding*.
https://github.com/ggml-org/llama.cpp/pull/22673 (accessed 2026-09-10)

{ref 3.} *Qwen3.6 MTP on llama.cpp with RTX PRO 6000*.
https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000 (accessed 2026-09-10)

{ref 4.} *sudoingX/qwen38-mtp*. https://github.com/sudoingX/qwen38-mtp (accessed 2026-09-10)

{ref 5.} *SharedLLM — llama.cpp tensor-split*.
https://sharedllm.org/blog/llama-cpp-tensor-split.html (accessed 2026-09-10)

{ref 6.} *llama.cpp multi-GPU offload performance*.
https://knightli.com/en/2026/05/09/llama-cpp-multi-gpu-offload-performance/ (accessed 2026-09-10)

{ref 7.} *Speculative decoding — LLM Inference Optimization Handbook* (BentoML).
https://bentoml.com/llm/inference-optimization/speculative-decoding (accessed 2026-09-10)

{ref 8.} *arXiv 2508.08192*. https://arxiv.org/pdf/2508.08192 (accessed 2026-09-10)

{ref 9.} *llama.cpp — Speculative decoding* (`docs/speculative.md`).
https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md (accessed 2026-09-10)

{ref 10.} *RTX 3090 + Qwen3.6-35B-A3B — speculative mode sweep*.
https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw (accessed 2026-09-10)
