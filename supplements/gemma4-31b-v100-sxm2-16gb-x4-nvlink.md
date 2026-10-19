# [Supplementary] Gemma 4 31B — V100-SXM2 16GB x4 (NVLink, DGX-1)

Measured 2026-10-19. **A supplementary run outside the frozen methodology**
([`README.md`](README.md)).
Raw data [`../results/supplements/gemma4-31b-v100-sxm2-16gb-x4-nvlink.jsonl`](../results/supplements/gemma4-31b-v100-sxm2-16gb-x4-nvlink.jsonl).

**Two fixed items were changed — the model (METHOD 1-1) and the context (1-4).** The MTP
method that comes with the model changed too. Everything else follows METHOD v1.4.

**Same system, same GPUs (0~3), same harness structure, same fixture text — so it can be
set against the main run ([V100 NVLink](../docs/v100-sxm2-16gb-x4-nvlink.md), Qwen3.8).**
That comparison is the purpose of this report.

**In short**

- **Tensor scaling efficiency did not depend on the model** — Gemma 77 / 62 / 53%, Qwen
  78 / 59 / 52%
- **The MTP result was opposite between the models** — under layer split Qwen +7%,
  Gemma −40%; under tensor Qwen −4%, Gemma +27%
- What decided the latter was **the cost multiple, not acceptance** (section 4)

---

## 1. What changed — METHOD fixed items

| METHOD | Main (Qwen) | **This run (Gemma)** | Reason |
|---|---|---|---|
| 1-1 model | Qwen3.8-27B-Heretic Q3_K_M, 13,115 MiB | **Gemma 4 31B it heretic Q6_K, 24,034 MiB** | the subject of the run |
| 1-1 MTP | MTP head inside the model (one layer, `blk.64`) | **separate draft model** (gemma4-assistant, four layers, shares the main model's KV), given with `-md` | structure of the model family |
| 1-3 fixture | Mujeong, targets 23,079 … 25,470 tokens | **same text** (byte-identical). 24,492 … 27,022 tokens under Gemma's tokenizer (+6%) | below |
| 1-4 `-c` | 27000 | **28000** | warm 4 + 400 = 27,422 does not fit in 27000 |
| 1-5 sampling | unset → GGUF defaults (top_k 20) | unset → GGUF defaults (**top_k 64**, temp 1.0, top_p 0.95) | values that come with the model |
| 1-6 cells | 12 cells + N=1 | **12 cells.** N=1 impossible | 24,034 MiB does not fit on one 16GB card |
| everything else | — | same | — |

**The fixture matches the text, not the token count.** There were two options — (A) keep
the same text and raise `-c`, or (B) drop messages from the front to meet METHOD's token
targets and keep `-c 27000`. **We chose A.** MTP acceptance is sensitive to the generated
content (METHOD 1-3), so keeping the content identical serves a comparison between models
better. The cost is about 3.7% more KV (by `-c`) and about 6% longer prompts.

The KV format is the default (f16) in both runs.

---

## 2. Environment

The system, GPUs, driver, CUDA, NCCL and llama.cpp build are **the same as the main run**
([V100 NVLink report section 1](../docs/v100-sxm2-16gb-x4-nvlink.md)). Measured on the
same day on the same GPUs 0~3.

### Model

| File | Source | Size | sha256 |
|---|---|---|---|
| `gemma-4-31B-it-uncensored-heretic-Q6_K.gguf` | downloaded as is from HF `llmfan46/gemma-4-31B-it-uncensored-heretic-GGUF` (commit eee61b81) | 25,201,484,928 B | `ec331ca8d40bef30bc00312b506a78e01bd9bdc6aa98d449d79b2942eca6efc1` |
| `gemma-4-31B-it-assistant-Q8_0.gguf` (draft) | HF `google/gemma-4-31B-it-assistant` (commit 627c5ec1), converted with the command below | 514,687,008 B | `9146f2fac8262724207eceaa388665286e7413b5661ffaca72e0e23835dde1ce` |

```
python convert_hf_to_gguf.py <gemma-4-31B-it-assistant> --outtype q8_0 --outfile gemma-4-31B-it-assistant-Q8_0.gguf
```

The conversion used the script from the same llama.cpp (465e49b9c). The main model is a
derivative of Gemma 4 31B it with refusals reduced by Heretic, quantized by its publisher
as a static Q6_K (no imatrix trace in the file header). **The Qwen in the main run is a
heretic derivative too.**

### Harness

[`../harness/supplements/server_bench_gemma4.py`](../harness/supplements/server_bench_gemma4.py).
Its header lists the differences from the original harness — prompt assembly (Gemma 4
chat format), the draft `-md`, `-c 28000`, twelve cells.

---

## 3. Results

Definitions are those of the main report — tg, warm pp and warm turn are means of the four
warm turns, cold is one run, and percentages are relative to the slowest cell in this run.

| Metric | Baseline cell (100%) | Value |
|---|---|---|
| Cold pp | layer / MTP on / 2 cards | 835.6 t/s |
| Warm pp | layer / MTP on / 4 cards | 221.7 t/s |
| tg | layer / MTP on / 4 cards | 13.24 t/s |

### 3-1. Twelve cells

| Cell | Cold pp | Warm pp | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|
| layer / off / 4 | 1,598.7 | 320.9 | 22.53 | 170% | — | 19.7s |
| layer / off / 3 | 1,283.0 | 319.6 | 22.68 | 171% | — | 19.6s |
| layer / off / 2 | 1,033.4 | 292.4 | 22.89 | 173% | — | 19.6s |
| layer / on / 4 | 1,061.2 | 221.7 | **13.24** | **100%** | 39.2% | 32.9s |
| layer / on / 3 | 965.6 | 222.2 | 13.66 | 103% | 39.2% | 32.0s |
| layer / on / 2 | 835.6 | 223.1 | 22.76 | 172% | 39.2% | 20.3s |
| tensor / off / 4 | 1,589.7 | 350.7 | 48.56 | 367% | — | 10.0s |
| tensor / off / 3 | 1,193.7 | 336.0 | 41.99 | 317% | — | 11.4s |
| tensor / off / 2 | 969.6 | 331.8 | 35.05 | 265% | — | 13.3s |
| **tensor / on / 4** | 1,489.3 | 316.6 | **61.61** | **465%** | 42.0% | **8.5s** |
| tensor / on / 3 | — | — | **failed** | — | — | — |
| tensor / on / 2 | 926.9 | 279.1 | 44.52 | 336% | 38.7% | 11.2s |

**First place is `tensor / MTP on / 4 cards`.** In the main run (Qwen) it was
`tensor / MTP off / 4 cards`. **Last is `layer / MTP on / 4 cards`, at 21% of first.**

### 3-2. Validation checklist (METHOD section 4)

| Item | Result |
|---|---|
| No `NCCL not compiled in` / `internal AllReduce init failed` | ✅ 11 cells + topology cell |
| `prompt_n` | 24,492 / 24,726 / 25,386 / 26,197 / 27,022 (not the METHOD targets but **the Gemma token counts of the same text**, identical in every cell) |
| Cold `cache_n` = 0, warm `cache_n` ≥ 22,000 | ✅ warm 24,486 → 26,193 |
| `predicted_n` = 400 | ✅ |
| Warm tg spread within 5% (MTP off) | ✅ 0.5~3.1% |
| Thermal and power throttling | thermal 0%. Power-cap bit in 10~31% of samples in the two-card cells, 0~5% elsewhere (section 5) |
| No kernel Xid during the run | **not checked** — same reason as the main report (the kernel log of a shared system was not read) |

---

## 4. Against the main run (Qwen) on the same system

### 4-1. Tensor scaling efficiency — independent of the model

| N | Qwen speedup | **Qwen efficiency** | Gemma speedup | **Gemma efficiency** |
|---|---|---|---|---|
| 2 | 1.56x | **78%** | 1.54x | **77%** |
| 3 | 1.78x | **59%** | 1.85x | **62%** |
| 4 | 2.09x | **52%** | 2.14x | **53%** |

`tg_1` is 27.13 measured for Qwen; **for Gemma it is derived as the mean of layer split at
two to four cards, 22.70** (METHOD section 5, N=1 impossible). The layer cells span only
1.6%, so the error of the derivation is small.

**Model size (13 against 24 GiB), architecture (three quarters of Qwen's layers are
DeltaNet; Gemma uses ordinary attention) and quantization all differ, yet efficiency
agrees within 3 points.** On this system what sets tensor-parallel scaling is the system,
not the model — which is how it reads alongside the main-run result that the same Qwen
scaled at 42% on PCIe V100 and 52% on NVLink.

### 4-2. MTP — opposite results

| | **Qwen** cost multiple | tokens/step | **result** | **Gemma** cost multiple | tokens/step | **result** |
|---|---|---|---|---|---|---|
| layer 4 | 1.816 | 1.951 | **+7.6%** | **3.685** | 2.165 | **−41.2%** |
| layer 3 | 1.827 | 1.951 | +7.0% | **3.594** | 2.165 | **−39.7%** |
| layer 2 | 1.819 | 1.951 | +7.6% | 2.178 | 2.165 | −0.6% |
| tensor 4 | 1.904 | 1.816 | **−4.5%** | **1.769** | 2.244 | **+26.9%** |
| tensor 3 | 1.860 | 1.935 | +4.1% | — | — | **failed** |
| tensor 2 | 1.897 | 1.824 | −3.6% | **1.694** | 2.151 | **+27.0%** |

**Prediction and measurement agree within 0.1 points for the five Gemma cells as well** —
the cost model of METHOD section 5 holds for a separate draft model unchanged.

**Gemma's acceptance is nearly the same, 38.7~42.0%, both where it gains (tensor) and
where it loses (layer, three and four cards)** — and higher than Qwen's (27.5~32.0%).
**What flips the sign is not acceptance but the cost multiple** — FINDINGS 4's "what moved
is the break-even line" appears between models as well.

| | Qwen | Gemma |
|---|---|---|
| Range of the cost multiple | **1.82~1.90 (independent of split and card count)** | **1.69 (tensor 2) ~ 3.69 (layer 4)** |
| Break-even acceptance | 27~30% | 23~26% (tensor) / 40% (layer 2) / **87~90% (layer 3, 4)** |

**Qwen's multiple is nearly the same everywhere; Gemma's spreads more than twofold with
the split mode and card count.**

### 4-3. Why — hypothesis (not tested)

**The MTP structures differ.** Qwen's MTP is a one-layer head inside the model. Gemma's
draft is a separate four-layer model that shares the main model's KV. The numbers
suggest the following.

- **Under layer split Gemma's per-step cost jumps from 94.9 to 158.1 ms between two and
  three cards** (per-token cost stays at 43.6 → 44.0 ms). From three cards the draft path
  crosses one more card boundary. Under layer split drafting and verification pass through
  the cards in turn; because the separate draft shares the main model's KV, we presume
  that round trip happens on every step
- **Under tensor parallelism the draft is split across the four cards and runs with them**
  — VRAM rose evenly by about 0.6GB per card (N=4, tensor off 8.6GB → on 9.2GB). The cost
  per step stays at 1.7x the cost per token

**Not tested.** Draft placement across cards and the KV-sharing path were not
instrumented.

### 4-4. Other metrics

| 4 cards | Qwen tensor / off | Gemma tensor / off |
|---|---|---|
| Cold pp (tensor / layer) | 1,968.2 / 1,533.9 = **tensor 1.28x** | 1,589.7 / 1,598.7 = **equal** |
| Warm pp (tensor / layer) | 903.8 / 473.8 = tensor 1.91x | 350.7 / 320.9 = tensor 1.09x |
| tg (tensor / layer) | 2.11x | 2.16x |

**The tensor lead on cold prefill disappears with Gemma.** The main run's "on NVLink tensor
wins the cold prefill too" does not hold across models. Why Gemma's warm pp is low in every
cell (221~351) was not checked — the large head dimension (512) of its global-attention
layers is a candidate.

---

## 5. Topology cell — NV1 pair and NV2 pair

GPUs 0~3 of the DGX-1 have different numbers of NVLinks per pair. The matrix's two-card
cells used 0 and 1 (NV1), following METHOD 1-6. **The same cell was run once more on 2
and 3 (NV2).**

| `tensor-nomtp-2` | Cold pp | Warm pp | **tg** | Warm spread |
|---|---|---|---|---|
| GPUs 0, 1 (**NV1**, one link) | 969.6 | 331.8 | **35.05** | 0.7% |
| GPUs 2, 3 (**NV2**, two links) | 1,005.0 (+3.7%) | 331.2 | **35.09 (+0.1%)** | 0.9% |

**Decode does not depend on the link count.** Only the cold prefill was 3.7% faster. We
read this as decode's all-reduce being tens of KB per token, which one link handles, while
prefill's larger batches let bandwidth show a little.

This cell is published as `validity: side`.

**The power-cap bit** was set in 21.6% (NV2) / 31.4% (NV1) of samples in the two-card
tensor cells and 0% in the four-card cells. We read this as two cards computing without
pause and reaching the cap, but without a single-card cell it cannot be compared the way
the main run does.

---

## 6. Failure — `tensor-mtp-3`

**It aborted on an internal check during loading.** Not out of VRAM.

```
ggml-backend-meta.cpp:1098: GGML_ASSERT(split_state.ne[j]*split_state.nr[0] * tensor->src[i]->ne[src_ss[i].axis] == sum * tensor->ne[split_state.axis]) failed
```

Just before it, the log records the draft layers sharing the KV of the main model's
layer 58.

| Same three cards | Result |
|---|---|
| `tensor-nomtp-3` | ok |
| `layer-mtp-3` | ok |
| **`tensor-mtp-3`** | **failed (twice)** |
| `tensor-mtp-2`, `tensor-mtp-4` | ok |

**It occurs only when a three-way tensor split meets draft KV sharing.** Every head count
in Gemma 4 is a multiple of four (32 query, 16 sliding-window KV, 4 global KV), so three
cards cannot split them evenly. Tensor splitting itself handles the uneven split
(`tensor-nomtp-3` is fine); we presume the draft's KV-sharing graph does not. **As of
llama.cpp 465e49b9c.** Not retried (deterministic, reproduced twice).

**The practical impact is small** — there is little reason to run three-card tensor, and
tensor + MTP works at two and four cards.

---

## 7. How it was measured

This report discloses the following.

1. **The twelve cells were measured in two sessions.** The first measured ten (six layer,
   three tensor off, `tensor-mtp-4`); the second `tensor-mtp-3`, `tensor-mtp-2` and the
   topology cell. Conditions were the same. The `run` field in the raw data tells them apart
2. **`tensor-mtp-3` also stalled during loading in the first session.** Its log was
   overwritten when the second session started, so the cause cannot be seen directly, but
   the symptom (no progress at the loading stage) was the same
3. **The first session ran on a harness from before telemetry was limited.** Its telemetry
   CSVs contained rows for GPUs not used in this run; those rows were removed before
   publication. The measured values (`timings`) are unaffected
4. **A pilot** (`-c 30000`, four cells at four cards) was run on the same day on the same
   GPUs. Warm four-turn tg agreed with this run within 0.4% for the two MTP-off cells and
   within 3.8% for the two MTP-on cells (`layer-nomtp-4` 22.61 / 22.53, `tensor-nomtp-4`
   48.49 / 48.56, `layer-mtp-4` 12.76 / 13.24, `tensor-mtp-4` 61.04 / 61.61). Because `-c`
   differs, its raw data is not published

---

## 8. Limits

1. **A supplementary run.** The model and context differ from METHOD. It is not used as
   evidence for the main findings
2. **A comparison of two models.** "The model decides whether MTP pays" is the observation
   that two models gave opposite results; which property of the model is responsible was
   not isolated (4-3)
3. **The quantization differs** (Q6_K against Q3_K_M). Whether quantization changes the MTP
   result on this system was not measured
4. **The MTP structure differs.** This report treats it as a property of the model family
5. **The sampling defaults differ** (top_k 64 against 20), which may affect acceptance
6. **The fixture is 6% longer in tokens** (section 1)
7. **There is no N=1.** `tg_1` is derived from layer split
8. **Xid was not checked**
9. Tensor parallelism changes the output as N changes (METHOD 6-2), so acceptance
   differences between tensor cells are not controlled

---

## 9. Not included

- **Four cells of the same model on P104-100 x4** — the tensor cells had to run with KV
  q8_0 (f16 ran out of VRAM), and the layer + MTP cell ran short of VRAM with reduced
  compute buffers. The conditions are not comparable, so they are not published. In
  direction only: layer + MTP lost heavily (−71%), and tensor + MTP did not load
- The pilot cells (section 7)
