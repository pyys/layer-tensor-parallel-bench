# METHOD — measurement methodology (frozen, v1.1)

Frozen 2026-09-10.

This document is a **contract that makes results comparable across platforms.**
If any item marked "fixed" here changes, the results **cannot go in the same table**
as another platform's. If something has to change, bump the version and keep results
measured under the old version in a separate table.

Each platform report (`docs/<platform>.md`) refers to this document rather than
restating the method.

*[한국어: `METHOD.ko.md`](METHOD.ko.md) — English is authoritative.*

---

## 1. Fixed — do not change

### 1-1. Model

| Item | Value |
|---|---|
| Source | `RVN-BF16-mtp.gguf` from `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` |
| Quantization | **Q3_K_M** |
| Size | 13,752,763,616 bytes (13,115 MiB) |
| Architecture | `qwen35`, 65 blocks (64 layers + 1 MTP head), `full_attention_interval = 4` |

Q3_K_M is chosen not for quality but because it is **the largest quantization that
fits on two 8GB cards**. Anything larger and low-VRAM platforms cannot run the N=2
cells, leaving a hole in the scaling curve.

To reproduce the quantization:

```
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M
```

`--tensor-type blk.64=q8_0` is required. The MTP head in the source file is already
q8_0, and without this the run fails with
`requantizing from type q8_0 is disabled`.

**Record `sha256sum` before measuring and put it in the report.** Re-quantizing on
another machine can produce a different file if the build differs.

### 1-2. Build

```
cmake -S <src> -B <src>/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
```

- **NCCL must actually be present.** `GGML_CUDA_NCCL` defaults to ON, but
  [NCCL does not ship with CUDA](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md) —
  install `libnccl2` / `libnccl-dev` yourself, matching your CUDA version.
  **Passing the flag proves nothing**; cmake merely warns
  (`Could NOT find NCCL`) and builds anyway.
  Without the library, llama.cpp's built-in AllReduce — which is **2-GPU only** —
  falls back on 3+ GPUs and tensor-parallel numbers come out ~3x wrong.
  Section 4 catches this at runtime; check there, not at build time
- `CMAKE_CUDA_ARCHITECTURES` varies by platform (allowed variation, section 3)
- Record the llama.cpp commit hash in the report. If it differs between platforms,
  say so

### 1-3. Prompt fixture

*Mujeong* (무정) by Yi Kwang-su, 1917, public domain, fetched from Korean Wikisource.
The text is not committed; `harness/make_fixture.py` fetches it at **pinned revision
IDs**, so anyone rebuilding gets a byte-identical fixture.

The structure imitates a real chat session — short `user` turns alternating with long
`assistant` turns, accumulating across warm turns.

**Target token counts** (measured with `/tokenize`, tolerance ±1%):

| Prompt | Tokens |
|---|---|
| Cold | **23,079** |
| Warm 1 | 23,177 |
| Warm 2 | 23,890 |
| Warm 3 | 24,688 |
| Warm 4 | 25,470 |

These numbers come from a real (unpublishable) chat log. Matching them means a
second fixture built from that log differs **only in the text**, which makes it a
controlled cross-check (see the appendix in the P104 report).

⚠️ The fixture is Korean, in 1917 orthography. **Tokens per character and
predictability both vary by language and era, so these numbers must not be compared
directly against English benchmarks.**

### 1-4. Server arguments

```
-ngl 99 -c 27000 -np 1 --split-mode <layer|tensor> [--spec-type draft-mtp]
```

- **`-c 27000`** — the ceiling at which two 8GB cards still leave room for NCCL's P2P
  buffers. **Do not raise it on platforms with more VRAM.** Raising it changes the KV
  size and breaks comparability
- **`-np 1`** — one slot. Multiple slots replicate SSM state per sequence, changing
  memory behaviour
- **Do not pass `-ts`.** Default even split. Per-platform tuning destroys comparability
- Do not pass chat-template arguments. Measurement goes through `/completion`, which
  does not apply a chat template

### 1-5. Request parameters

```json
{"prompt": "...", "n_predict": 400, "ignore_eos": true, "cache_prompt": true, "seed": 1234, "stream": false}
```

- **`ignore_eos: true` with `n_predict 400`** makes the generation work identical
  across all 12 cells. ⚠️ It also forces generation past the point where the model
  would stop, so **MTP acceptance reads lower than in real use.** Every report must
  state this
- **Do not send sampling parameters.** The GGUF metadata defaults apply
  (`temp 1.0 / top_k 20 / top_p 0.95`). The model is fixed, so this is deterministic
- `cache_prompt: true` — prompt-cache hits on warm turns are part of what is measured

### 1-6. Cell definition

`{layer, tensor} × {MTP off, on} × {2, 3, 4 GPUs}` = **12 cells**

- GPUs are selected with `CUDA_VISIBLE_DEVICES=0,1,...`, the first N
- Cell names are `<split>-<nomtp|mtp>-<N>`, e.g. `tensor-nomtp-4`
- Run order is 4 → 3 → 2 GPUs. The combinations most likely to fail go last, so
  earlier results survive
- **N=1 is an optional addition** on platforms where 13,115 MiB fits on one card.
  The comparable core is always 2, 3 and 4

### 1-7. Measurement source

**Do not parse server logs.** Take the `timings` object from the `/completion`
response:

`prompt_n` / `prompt_ms` / `prompt_per_second` / `cache_n` /
`predicted_n` / `predicted_ms` / `predicted_per_second` /
`draft_n` / `draft_n_accepted`

Record wall-clock time (`wall_s`) alongside.

### 1-8. Telemetry

Sampled once per second, per cell:

```
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader -l 1
```

**Whether utilization sums to more than 100%** is the first-order indicator of
whether parallelism is doing anything.

---

## 2. Procedure

```
1. Record the model sha256
2. python3 server_bench.py --check      # token counts within ±1% of 1-3
3. Manual probe: bring up the heaviest cell (N=2, tensor, MTP on) and confirm it loads
4. python3 server_bench.py              # all 12 cells
5. Verify results (section 4)
```

Do not skip step 3. If the tightest cell fails to load, the whole run has to be redone.

The harness **skips completed cells** on re-run, and retries cells that errored.

---

## 3. Allowed variation per platform

| Item | Note |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | matched to the card |
| Driver / CUDA Toolkit version | record in the report |
| NCCL version | record. **Absent means the measurement is void** |
| Number of GPUs available | leave cells empty if fewer than 4 |
| CPU / RAM / PCIe generation / topology | record in the report's environment section |

**Changing anything not on this list means it is not a v1.1 measurement.**

---

## 4. Verification checklist

Run through this after measuring. Any failure invalidates that cell.

- [ ] Startup log does **not** contain `NCCL not compiled in`
- [ ] Startup log does **not** contain `internal AllReduce init failed`
- [ ] `prompt_n` matches the targets in 1-3 for every cell
- [ ] Cold turn `cache_n` is 0
- [ ] Warm turn `cache_n` is at least 22,000 (cache hit)
- [ ] `predicted_n` is 400 on every turn
- [ ] **For MTP-off cells**, the spread of `predicted_per_second` across the four warm
      turns is within 5%

⚠️ **The 5% rule does not apply to MTP-on cells.** Acceptance swings with the content
being generated; that is intrinsic to speculative decoding, not a measurement error
(up to 67% spread on P104-100). **Record and report the spread instead** — a cell can
have a respectable average and still feel bad if response times jump around, so the
spread is itself a result.

---

## 5. Derived metrics

Every report uses these formulas.

**Warm turn time** = mean of `wall_s` over the four warm turns

**Session time** = `cold wall_s + warm turn time × N`

**Session break-even** — where configuration A starts beating B

```
N = (cold_A - cold_B) / (warm_B - warm_A)
```

**Tensor scaling efficiency**

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

Use a measured `tg_1` where possible. Where the model does not fit on one card,
substitute the **layer-split `tg`** (which is constant in N, hence equal to
single-card speed) and **state that it is a substitution**.

**MTP batch economics** — what decides whether speculative decoding pays

```
steps        = n_predict - draft_n_accepted
cost/step    = predicted_ms(MTP) / steps
cost/token   = predicted_ms(no MTP) / n_predict
multiplier   = cost/step ÷ cost/token
yield/step   = n_predict / steps
```

**If yield < multiplier, MTP is a loss.**

**Break-even acceptance**

```
draft length d = draft_n / steps
1 + d × acc = multiplier   ->   acc = (multiplier - 1) / d
```

**Roofline effective bandwidth**

```
effective bandwidth = bytes the card reads ÷ the time that card spends reading them
```

A low fraction of spec bandwidth means the workload is **not bandwidth-bound**.

---

## 6. Known pitfalls

| Symptom | Cause / response |
|---|---|
| `-sm row` reports `does not support split buffers` | That path was removed from the CUDA backend. **Use `-sm tensor`** |
| Tensor-parallel numbers look oddly low | Built without NCCL. Section 4 catches this |
| NCCL `Cuda failure 2 'out of memory'` in `transport/p2p.cc` | The model loaded, but there is no VRAM left for NCCL's P2P buffers. Independent of topology. `NCCL_P2P_DISABLE=1` gets it running but routes through host memory, **which breaks comparability.** Shrinking the context is the real fix, but `-c 27000` is fixed in v1.1, so record the cell as failed |
| `layer` + MTP on few GPUs: compute-buffer OOM | The MTP draft context does not fit. Record as failed |
| `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | Warning only. Irrelevant, since `-ngl` and `-c` are given explicitly |
| `backend sampling not supported with SPLIT_MODE_TENSOR; using CPU` | Intrinsic to tensor parallelism. Treat as a controlled variable |
| Prompt exceeds `n_ctx` | llama.cpp **silently truncates**, corrupting the measurement. Never skip `--check` |

---

## 6-2. ⚠️ Tensor parallelism changes the generated output

With the same seed and the same prompt, **tensor parallelism produces different
tokens when the split mode or GPU count changes**, because the all-reduce ordering
changes the floating-point result slightly.

Observed on P104-100:

- `layer-mtp-4` and `layer-mtp-3` have **identical** `draft_n` / `draft_n_accepted`
  on every turn → layer split is deterministic regardless of N
- `tensor-mtp-4 / 3 / 2` are **all different** (675 / 642 / 660 ...)

So **MTP acceptance is not controlled across tensor-parallel cells.** Do not read a
difference in acceptance as an effect of GPU count. MTP-off metrics (pp, tg) do not
depend on the generated content and are unaffected.

---

## 7. What must not be reported

- **Do not use a synthetic benchmark (`llama-bench`'s `pp512` and friends) as a stand-in
  for real cold prefill.** The batch is small enough that the ranking between split
  modes reverses. Use it as a side note and say so
- **Do not put results measured under different methodology versions in the same table**

---

## 8. Version history

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-09-10 | Initial freeze, based on the P104-100 ×4 measurement |
| **v1.1** | 2026-09-10 | Section 4 spread check now applies to **MTP-off cells only** (spread is intrinsic to MTP). **Section 6-2 added** — tensor parallelism changes output when N changes |

v1.0 and v1.1 have **identical measurement conditions**. Only verification and
interpretation rules changed, so results measured under v1.0 remain valid.
