# Findings — what we found, and what we did not

Two platforms have been measured. The methodology is frozen in
[`METHOD.md`](METHOD.md); each platform's environment, results and
interpretation live in its own report.

| Platform | Interconnect | Measured | Report | Raw data |
|---|---|---|---|---|
| P104-100 8GB x4 | PCIe Gen1 x4 (~1 GB/s) | 2026-09-10 | [docs/p104-100-x4.md](docs/p104-100-x4.md) | [results/](results/) |
| V100-SXM2 16GB x4 | PCIe Gen3 x16 (~15.75 GB/s) | 2026-09-14~15 | [docs/v100-sxm2-16gb-x4.md](docs/v100-sxm2-16gb-x4.md) | [results/](results/) |

**Both platforms were measured with the same binary**
(`CMAKE_CUDA_ARCHITECTURES=61;70`). Everything except the GPUs was reused, so the
difference comes from the cards — but card, link, VRAM and power limit all changed
at once, and we did not decompose how much each contributed
[Notes and Caveats 6)](#notes-and-caveats).

**On both platforms the fastest token generation came from `tensor parallelism +
MTP off`.** Conventional advice is to use layer split with MTP on a slow interconnect;
depending on the conditions, that is not always right. The evidence is below.
[V100 results](docs/v100-sxm2-16gb-x4.md#2-results) ·
[P104 results](docs/p104-100-x4.md#2-results)

---

## Where to start

| If you want to know | Start here |
|---|---|
| **Whether to use layer split or tensor parallelism** | [Finding 1](#1-layer-split-buys-no-decode-speed), [Finding 2](#2-two-cards-with-tensor-parallelism-beat-four-with-layer-split) |
| **Why your tensor-parallel numbers look wrong** | [METHOD 1-2 — NCCL](METHOD.md) |
| **Whether speculative decoding pays off on your card** | [Finding 4](#4-speculative-decoding-flips-sign-with-the-card-and-the-split-mode) |
| **How much a fourth card actually buys** | [Finding 5](#5-interconnect-bandwidth-is-not-what-limits-tensor-parallel-scaling) |
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
| **V100** tg | **27.71** | 27.97 | 27.78 | 27.29 |

**Both platforms stay within 3%. From two cards up, more cards is slightly slower.**

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

Cold prefill throughput rises by up to **2.08x**, while warm prefill barely moves.
A cold prompt has dozens of micro-batches, enough to fill the pipeline; a warm turn
(637~790 tokens) has one or two and cannot.

→ **What extra cards buy under layer split is (a) more VRAM and (b) faster handling
of long prefills.** "Long" here is about prompt token count, not cache state — we
measured 2.08x at 23k tokens and 0x at 700 tokens. Where in between the gain starts,
we did not measure.

---

## 2. Two cards with tensor parallelism beat four with layer split

Warm turn wall time.

| Configuration | P104 | V100 |
|---|---|---|
| layer, 4 cards | 59.7s | 16.2s |
| **tensor, 2 cards** | **36.9s** | **10.8s** |
| tensor, 4 cards | 28.1s | 9.9s |

**Half the cards, 1.6x faster on P104 and 1.5x on V100.**

Tensor parallelism has a slower cold prefill, so it loses the first turn. It does not
stay behind for long.

```
session = cold + warm x N
crossover N = (cold_A - cold_B) / (warm_B - warm_A)
```

| Crossover | P104 | V100 |
|---|---|---|
| tensor 4 > layer 4 | 1.2 turns | **0.87** |
| tensor 2 > layer 4 | 2.4 turns | **0.12** |

**On V100 tensor wins from the very first warm turn.**

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

## 3. Cold and warm prefill rank the split modes in opposite order

| | Cold (23,037 tok) | Warm (637~790 tok) |
|---|---|---|
| **P104** layer 4 | **327.3** | 85.3 |
| **P104** tensor 4 | 171.4 | **128.8** |
| **V100** layer 4 | **1,555.5** | 487.6 |
| **V100** tensor 4 | 887.4 | **594.1** |

| Ratio | P104 | V100 |
|---|---|---|
| Cold — layer ahead | 1.91x | 1.75x |
| Warm — tensor ahead | 1.51x | 1.22x |

**The direction holds on both platforms; only the magnitude shrinks.**

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

| | P104 layer 4 | V100 layer 4 |
|---|---|---|
| Prefill | 8.5s (14%) | 1.5s (10%) |
| **Generation** | **51.2s (86%)** | **14.7s (90%)** |

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

**The premise fails in all five measured conditions.** Even the lowest cost multiple
is **1.86x**. And in **four of the five, MTP is a net loss.**

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

### 4-2. What sets the regime is not batch size

The rule of thumb in the literature — "below 50% acceptance it is a loss" — is usually
framed as a property of **batch size**. Here batch size is pinned at 1 and the sign
still flips. Two things decide it.

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

## 5. Interconnect bandwidth is not what limits tensor-parallel scaling

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

> The same conclusion was reached independently for diffusion models — the cost of
> row split in sd.cpp did not improve across the same card swap. Different engine,
> different workload, different split implementation.
> → [`image-model-split-bench`](https://github.com/pyys/image-model-split-bench)

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

**All of them sit below half of spec.** V100 at one card and at four under layer split
land on the same 42%, which is what the structure predicts — layer split only
serializes the same work, so per-card utilization should match a single card.

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

| Cell | P104 | V100 |
|---|---|---|
| `layer-nomtp-4` | ok | ok |
| `layer-mtp-4` | ok | ok |
| `tensor-nomtp-4` | ok | ok |
| **`tensor-mtp-4`** | **ok (14.82 t/s)** | **crashed twice** |

**It is not "because it was four cards".** What is distinctive about this combination
is neither arithmetic nor power but **transaction frequency** — tensor parallelism
issues an all-reduce per token, and MTP draft verification rides on top, so **small
peer-to-peer transfers happen more often here than anywhere else.**

**And the same combination completed on P104**, where the link is 16x slower and
generation runs at a third the speed, so transactions per second were far lower.
Consistent with the hypothesis.

### Causes ruled out

| Cause | Verdict |
|---|---|
| Overheating | **Ruled out.** 76 °C peak across 12 cells; 50 °C in the crashing cell |
| Sustained power | **Ruled out.** 60% of the cap. A separate workload held ~1,120W across four cards without incident |
| ECC / memory | **Ruled out.** Zero retired pages, no uncorrectable errors |
| A single faulty card | **Weak.** Three cards were lost at once on the second crash |

**The cause remains undetermined** [Notes and Caveats 5)](#notes-and-caveats).

### What it means in practice

**Do not use `tensor + MTP + 4 cards` on this setup.** The fastest configuration,
`tensor-nomtp-4` at 46.62 t/s, has no MTP, and MTP is a loss in tensor mode anyway
(Finding 4). **Avoiding it costs nothing.**

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
   different full-attention ratio — may behave differently
3. **`ignore_eos` depresses MTP acceptance.** Generation continues past where the
   model would stop, so acceptance reads lower than in real use
   [Notes and Caveats 7)](#notes-and-caveats)
4. **`tensor-mtp-4` is permanently missing on V100** (Finding 7). Do not extrapolate
   the tensor-mode MTP result to four cards
5. **The single-card baseline on P104 is still derived**, not measured. V100 only
   confirmed that the derivation is sound
   [Notes and Caveats 2)](#notes-and-caveats)
6. **No Q6 comparison on V100.** The quantization conclusion in Finding 6 is specific
   to P104
7. Per-platform limits are in the last section of each report

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
