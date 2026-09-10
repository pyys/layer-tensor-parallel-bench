# layer-tensor-parallel-bench

### Layer split vs tensor parallelism in llama.cpp — measured across GPUs and interconnects, starting at PCIe Gen1 x4 (~1 GB/s)

**Numbers where the official documentation has none, under one frozen method.**

llama.cpp's own [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
describes what each split mode is for. It contains **no benchmark numbers**. This
repository supplies them, starting from an extreme case: **four mining GPUs on PCIe
Gen1 x4, no NVLink.**

Conventional wisdom says an interconnect this slow calls for **layer split rather than
tensor parallelism**, and that **turning MTP on buys you generation speed**. The author
was running exactly that — **layer split with MTP on**. It turned out to be the
**slowest of the 12 configurations measured.**

The workload is what actually happens in a chat session — one cold prefill of tens of
thousands of tokens, then warm turns where the prompt cache hits. **Which split mode
wins reverses between those two**, so a synthetic benchmark that measures only one of
them gives the wrong answer.

*[한국어 문서: `README.ko.md`](README.ko.md) — English is authoritative in this
repository. (Note: in the companion repo [local-ai-cost-cutting](https://github.com/pyys/local-ai-cost-cutting),
Korean is authoritative. The two repos differ on this point.)*

---

## ⚠️ Before you measure anything: install NCCL

`GGML_CUDA_NCCL` **defaults to ON**, but **NCCL does not ship with CUDA.** You have to
install it yourself. The official guide says this. What it does not say is what the
omission costs, or that the built-in fallback is **2-GPU only**:

```
W NCCL not compiled in; falling back to internal AllReduce.
W internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```

cmake only warns (`Could NOT find NCCL`). The server only warns. Everything keeps
running — at a third of the speed.

| 4 GPUs, Qwen3.6-27B Q6_K | No NCCL | NCCL |
|---|---|---|
| tensor pp512 | 61.15 | **190.48** (+211%) |
| tensor tg128 | 13.19 | **18.39** (+39%) |
| layer (control) | unchanged | unchanged |

Measured in the fallback state, the conclusion comes out as *"tensor parallelism
trades prefill for decode."* That is an artifact. **The first measurement in this
repository fell into exactly that trap and the conclusion had to be reversed.**

---

## Findings

*(One platform so far. Each finding gets re-checked for generality as platforms are
added. Evidence: [docs/p104-100-x4.md](docs/p104-100-x4.md))*

### 1. "Batch 1 is memory-bound, so verification is free" — not on every card

The standard account of speculative decoding assumes that at batch 1 the GPU is
memory-bandwidth-bound and mostly idle, so verifying a handful of draft tokens costs
almost nothing. That assumption is what makes speculative decoding profitable.

**On this hardware it is false at batch 1.** Verifying a batch of 4 costs
**2.25–2.42x** a single token, because the card is compute-bound even at batch 1.

| | layer, 4 GPUs | tensor, 4 GPUs |
|---|---|---|
| Cost multiplier for a batch of ~4 | 2.25 | 2.42 |
| Tokens yielded per step | 1.89 | 1.99 |
| **Break-even acceptance** | **42.3%** | **47.8%** |
| Measured acceptance | 29.9% | 33.1% |
| Measured loss | **−16.3%** | **−17.2%** |

Predicted loss from the cost model matched measurement within 0.9 percentage points.

The literature's rule of thumb — "below ~50% acceptance it hurts" — is usually framed
as a property of *batch size*. Here it is a property of *the card*: the regime is set
by the compute-to-bandwidth ratio, not by how many sequences are in flight.

**The method is reusable.** `steps = n_predict − accepted` gives the cost multiplier
directly from any llama.cpp run; see METHOD section 5.

For contrast, here are published measurements of the same llama.cpp MTP feature:

| Source | Hardware | Acceptance | Speedup |
|---|---|---|---|
| [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673) | — | — | ~**2x** (spread 1.2–2x) |
| [RTX PRO 6000](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000) | RTX PRO 6000 | — | **1.73x** |
| [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp) | consumer GPUs | 0.797 / 0.732 | **2.07x / 2.33x** |
| **This measurement** | **P104-100 ×4** | **0.299 / 0.331** | **0.84x / 0.83x** |

**The same feature doubles speed on one card and loses money on another.** And the
contrast lets the logic be read backwards — getting 2x at 75% acceptance means
**verification at batch 1 really is nearly free on those cards**, which is to say they
are **bandwidth-bound**.

⚠️ Those figures were not measured under this repository's methodology, and the model,
quantization and settings all differ. They are a reference point, not a control.

### 2. Layer split gains nothing on decode — quantified

The official guide says layer split is "tolerant of slow interconnects" and tensor
mode "prioritizes fast token generation." Here is the size of that difference.

| GPUs | 2 | 3 | 4 |
|---|---|---|---|
| tg (t/s) | 8.08 | 7.97 | 7.82 |

Within 3%, and **slightly slower with more cards**. Telemetry shows why: one GPU at
100%, the rest at 0%, rotating. Token generation walks the layers in order, so only
one card computes at any instant.

→ **In layer split, extra cards buy VRAM and long prefill. Nothing else.**

### 3. Two GPUs with tensor parallelism beat four with layer split

| Config | Warm turn |
|---|---|
| layer, 4 GPUs | 59.7 s |
| **tensor, 2 GPUs** | **36.9 s** |
| tensor, 4 GPUs | **28.1 s** |

**1.6x faster on half the cards.** Session break-even is **1.2 turns** (tensor 4) and
**2.4 turns** (tensor 2) — that is, from the second or third message onward.

And it uses less memory. At the production settings (Q6_K, `-c 90000`):

| | Per card (MiB) | Spread | Min free |
|---|---|---|---|
| Layer split, hand-tuned `-ts` | 7565 / 7253 / 7583 / 7699 | 446 | **410** |
| **Tensor parallelism** | **6823 × 4** | **0** | **1,286** |

**2.8 GB less, and balanced on the first load.** The layer-split figure above took six
reloads to find, and one earlier attempt failed to load at all. That is a **setup and
upgrade cost**, not a daily one — but it repeats every time the model changes or a
card is added.

Re-running the whole comparison at those production settings against the private
fixture: **warm turns went 58.4 s → 30.4 s (1.92x), generation 8.22 → 16.24 t/s.**
Break-even is 0.23 turns — B wins from the first warm turn onward. Details in
[the report](docs/p104-100-x4.md#2-6-re-checked-at-production-settings-q6_k--c-90000).

### 4. Cold and warm prefill reverse the ranking

| | Cold, 23k tokens | Warm, ~700 tokens |
|---|---|---|
| layer, 4 GPUs | **327.3 t/s** | 85.3 t/s |
| tensor, 4 GPUs | 171.4 t/s | **128.8 t/s** |

Long prefill favours layer split by 1.9x; short prefill favours tensor by 1.5x. On
the same hardware `llama-bench`'s `pp512` reports the **opposite** ranking, because
one or two micro-batches cannot fill a layer-split pipeline.

**Do not use a synthetic prefill benchmark to choose a split mode.**

And once the prompt cache hits, **86% of a warm turn is decode**. Warm turns, not
cold ones, decide what a user feels.

### 5. Tensor scaling efficiency falls off after two GPUs

| N | vs 1 GPU | Efficiency |
|---|---|---|
| 2 | 1.62x | **81%** |
| 3 | 1.77x | **59%** |
| 4 | 2.29x | **57%** |

Adding cards stops paying somewhere around four. N=3 is unusually poor, plausibly
because NCCL's ring/butterfly algorithms favour powers of two — **not verified**.

⚠️ The 1-GPU baseline here is a substitution, not a measurement — see Limitations.

### 6. Quantization is not a speed lever on a compute-bound card

| | Model size | tensor 4-GPU tg |
|---|---|---|
| Q6_K | 20.88 GiB | 18.39 |
| Q3_K_M | 12.81 GiB | **17.89** |

Reading 39% fewer bytes gives the same speed, and the roofline confirms it — only
19–34% of spec memory bandwidth is used. **Here, quantization buys capacity, not
speed.** This reaches the same conclusion as finding 1 by an independent route.

---

## Related work

This repository is additive to, not a correction of, the following.

| Source | What it covers | What it lacks |
|---|---|---|
| [llama.cpp `docs/multi-gpu.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) | Split-mode semantics, NCCL requirement, "`row` is deprecated", pipeline-vs-tensor tradeoff in concept | **No numbers.** No mention that the built-in AllReduce is 2-GPU only |
| [SharedLLM: llama.cpp tensor-split](https://sharedllm.org/blog/llama-cpp-tensor-split.html), [knightli multi-GPU guide](https://knightli.com/en/2026/05/09/llama-cpp-multi-gpu-offload-performance/) | Practical multi-GPU setup, some 2-GPU consumer numbers (e.g. 2×RTX 4090: 1.72x tg on 8B F16, 1.57x on Gemma4 31B Q8_0) | Two GPUs, fast PCIe. No scaling curve, no cold/warm split, no MTP |
| Speculative decoding literature ([BentoML handbook](https://bentoml.com/llm/inference-optimization/speculative-decoding), [arXiv 2508.08192](https://arxiv.org/pdf/2508.08192)) | Batch 1 is memory-bound so verification is nearly free; profitability collapses at large batch; acceptance below ~0.5 hurts | Regime is framed by **batch size**. Finding 1 here is a counterexample where the **card** sets the regime at batch 1 |
| llama.cpp MTP measurements ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [RTX PRO 6000](https://jarvislabs.ai/blog/qwen36-mtp-llamacpp-rtxpro6000), [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)) | MTP numbers on cards where it works (acceptance 0.73–0.80, 1.2–2.3x) | Only the winning side. **No breakdown of why it loses on some cards** |
| [All speculative modes tested on RTX 3090 + Qwen3.6-35B-A3B](https://hackmd.io/ODXuOQNzSiyUITz7g9mtBw) | **A published case where MTP was slower than baseline** | MoE model, so the cause may differ; no batch-cost decomposition |

If you know of prior measurements this repository duplicates, please open an issue —
being scooped is more useful than being wrong.

---

## Platforms measured

| Platform | Interconnect | Report | Raw data |
|---|---|---|---|
| P104-100 8GB ×4 | PCIe Gen1 x4, no NVLink | [docs/p104-100-x4.md](docs/p104-100-x4.md) | [results/](results/) |
| *(planned)* V100 16GB PCIe ×4 | | | |

---

## Method

[METHOD.md](METHOD.md) is a **frozen contract**. If any item marked "fixed" changes,
the results cannot go in the same table as another platform's.

| Item | Value |
|---|---|
| Model | Qwen3.8-27B-Heretic **Q3_K_M** (13,115 MiB) |
| Cells | `{layer, tensor} × {MTP off, on} × {2, 3, 4 GPUs}` = 12 |
| Prompt | ~23k-token cold + 4 warm turns (~700-800 tokens added per turn) |
| Context | `-c 27000`, fixed |
| Generation | `n_predict 400`, `ignore_eos` — identical work per cell |
| Measurement | the `timings` object in the `/completion` response |

Q3_K_M is chosen not for quality but because it is **the largest quantization that
fits on two 8GB cards**. Anything larger and the N=2 cells drop out, leaving a hole
in the scaling curve.

---

## Reproducing

```bash
# 0) Install NCCL first. It does not come with CUDA.
#    Match the CUDA version — the default candidate may target a newer CUDA.
apt-get install -y libnccl2=<ver>+cuda<your-cuda> libnccl-dev=<ver>+cuda<your-cuda>

# 1) Build llama.cpp. Confirm cmake prints "Found NCCL", not "Could NOT find NCCL".
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
cmake --build llama.cpp/build --config Release -j 8

# 2) Quantize the model (source on HuggingFace, see METHOD 1-1)
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M

# 3) Build the fixture — fetched from Wikisource at pinned revisions
python3 harness/make_fixture.py --pin          # once
python3 harness/make_fixture.py --port 8099 --out fixtures/mujeong.json

# 4) Verify token counts (do not skip — an oversized prompt is silently truncated)
python3 harness/server_bench.py --check

# 5) Measure
python3 harness/server_bench.py
```

Re-running skips cells that already completed.

---

## The fixture

*Mujeong* (무정) by Yi Kwang-su, 1917 — **public domain** — fetched from Korean
Wikisource and reassembled into a dialogue of short `user` turns and long
`assistant` turns.

The text is not committed. `harness/fixture.lock.json` **pins the revision IDs**, so
anyone rebuilding gets a byte-identical fixture.

⚠️ **The fixture is Korean, in 1917 orthography.** Tokens per character and
predictability both vary by language and era, so **do not compare these numbers
directly against English benchmarks.** MTP acceptance in particular is sensitive to
the fixture — see Appendix A of the P104 report, where the same cells were run
against a second fixture.

---

## Limitations

1. **The single-GPU baseline is not measured on every platform.** If the model
   (13,115 MiB) does not fit on one card, the layer-split `tg` stands in for it.
   Finding 5 depends on this substitution. Reports say so explicitly
2. **One run per cell.** The four warm turns show within-cell variation, but
   run-to-run reproducibility was not measured
3. **One model, one quantization.** A different architecture — particularly a
   different ratio of full-attention layers — could behave differently
4. **`ignore_eos` depresses MTP acceptance.** Generation is forced past the point
   where the model would stop, so acceptance reads lower than in real use.
   Finding 1's verdict rests on a break-even calculation, not a direct measurement
5. Platform-specific limitations are in section 7 of each report

---

## License

- Code (`harness/`) and raw data (`results/`): MIT — see [LICENSE](LICENSE)
- Documents (`README*`, `METHOD*`, `docs/`): CC BY 4.0 — see [LICENSE-docs](LICENSE-docs)
- Fixture source text: public domain (Yi Kwang-su, *Mujeong*, 1917). Not included in
  this repository

---

## Related

- [local-ai-cost-cutting](https://github.com/pyys/local-ai-cost-cutting) — building a
  low-cost local AI stack on heterogeneous GPUs. This repository drills into the
  multi-GPU LLM part of it with measurements
