# METHOD — measurement methodology (frozen, v1.2)

Frozen 2026-09-15.

This document is **a contract that keeps platforms comparable.**
If any item marked "fixed" changes, **the result cannot go in the same table as
another platform's.** If it must change, bump the version and split the older
results into their own table.

Each platform report (`docs/<platform>.md`) references this document rather than
restating the method. Results and interpretation live in
[`FINDINGS.md`](FINDINGS.md).

---

## 1. Fixed — do not change

### 1-1. Model

| Item | Value |
|---|---|
| Source | `RVN-BF16-mtp.gguf` from `0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF` [{ref 11.}](#references) |
| Quantization | **Q3_K_M** |
| Size | 13,752,763,616 bytes (13,115 MiB) |
| sha256 | `727e7d8b3ab3af1a6eeb166f3d7aac93d51ac0cdc06200e8f1cbac9074591ca4` |
| Architecture | `qwen35`, 65 blocks (64 real + 1 MTP head), `full_attention_interval = 4` |

Q3_K_M was chosen not for quality but because it is **the largest size that fits on
two 8GB cards.** Anything larger makes the N=2 cell unmeasurable on low-end platforms
and breaks the scaling curve.

To reproduce the quantization:

```
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M
```

`--tensor-type blk.64=q8_0` is required. The MTP head in the source is already q8_0,
so a plain run fails with `requantizing from type q8_0 is disabled`.

**Record `sha256sum` before measuring and put it in the report.** Re-quantizing on a
different machine can produce a different file.

### 1-2. Build — without NCCL the measurement is void

```
cmake -S <source> -B <source>/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
```

- `CMAKE_CUDA_ARCHITECTURES` varies by platform (permitted variation, section 3).
  If you plan to alternate between cards, list them together as in `61;70` —
  **measuring two platforms with one binary removes build differences as a variable**
- Record the llama.cpp commit hash in the report

> ## ⚠️ Install NCCL first
>
> `GGML_CUDA_NCCL` **defaults to ON.** But **NCCL is not shipped with CUDA**
> [{ref 1.}](FINDINGS.md#references). You have to install it yourself.
> **Passing the flag guarantees nothing** — cmake emits `Could NOT find NCCL` as a
> warning and builds anyway.
>
> Without the library, tensor parallelism on three or more cards does this:
>
> ```
> W NCCL not compiled in; falling back to internal AllReduce.
> W internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
> ```
>
> **The built-in AllReduce is two-GPU only.** With three or more it fails to
> initialize and falls back. The official guide does not mention this limit.
> **Everything keeps running** — with the tensor path three times slower.
>
> | 4 cards, Qwen3.6-27B Q6_K | No NCCL | NCCL |
> |---|---|---|
> | tensor **pp 512** | 61.15 | **190.48** (+211%) |
> | tensor **tg 128** | 13.19 | **18.39** (+39%) |
> | layer (control) | unchanged | unchanged |
>
> Measured in the fallback state, the conclusion reads *"tensor parallelism trades
> prefill for decode"*. **That is a measurement error.** The first run in this
> repository fell into exactly this trap; **the experiment was stopped and redone
> from scratch.**
>
> *(Those llama-bench figures are a side measurement outside the frozen methodology.
> `pp 512` is a 512-token prompt, `tg 128` is 128 generated tokens.)*

The install must match your CUDA version — the default candidate package may target
a newer one.

```
apt-get install -y libnccl2=<version>+cuda<your CUDA> libnccl-dev=<version>+cuda<your CUDA>
```

Check for `Found NCCL` at build time, but **make the final call from the runtime
checklist in section 4.**

### 1-3. Prompt fixture

Assembled from *Mujeong* by Yi Kwang-su (1917, **public domain**) via Korean
Wikisource [{ref 12.}](#references). Short `user` turns alternate with long
`assistant` turns and each warm turn accumulates the ones before it, mirroring
**the structure of a real conversation.**

**The text is not committed.** `harness/make_fixture.py` **pins a revision ID** in
`harness/fixture.lock.json`, so anyone who runs it gets a byte-identical fixture.

**Target token counts** (measured via `/tokenize`, tolerance ±1%):

| Prompt | Tokens |
|---|---|
| Cold | **23,079** |
| Warm 1 | 23,177 |
| Warm 2 | 23,890 |
| Warm 3 | 24,688 |
| Warm 4 | 25,470 |

These numbers come from a real (private) conversation log. Matching the fixture to
them gives **a controlled comparison where only the character of the text differs.**

⚠️ **The fixture is Korean and uses 1917 orthography.** Characters per token and
predictability vary by language and era, so **do not compare these directly against
English benchmarks.** MTP acceptance is especially fixture-sensitive — the P104
report has an appendix running the same cells against a different fixture.

### 1-4. Server arguments

```
-ngl 99 -c 27000 -np 1 --split-mode <layer|tensor> [--spec-type draft-mtp]
```

- **`-c 27000`** — the ceiling at which two 8GB cards still leave room for NCCL P2P
  buffers. **Do not raise it on platforms with headroom.** A different KV size breaks
  the comparison
- **`-np 1`** — one slot. Multiple slots replicate SSM state per sequence and change
  the memory picture
- **Do not pass `-ts`.** Default even split. Per-platform tuning breaks comparability
- Do not pass template arguments such as `--chat-template-kwargs`. Measurement uses
  `/completion`, which does not go through the chat template

### 1-5. Request parameters

```json
{"prompt": "...", "n_predict": 400, "ignore_eos": true, "cache_prompt": true, "seed": 1234, "stream": false}
```

- **`ignore_eos: true` + `n_predict 400`** equalizes the generation workload across
  all 12 cells.
  ⚠️ In exchange, generation runs past where the model would stop, so **MTP
  acceptance reads lower than in real use.** State this limit in the report
- **Do not send sampling parameters.** The GGUF metadata defaults apply
  (`temp 1.0 / top_k 20 / top_p 0.95`). The model is fixed, so this is deterministic
- `cache_prompt: true` — the prompt cache hit on warm turns is what is being measured

### 1-6. Cell definition

`{layer, tensor} x {MTP off, on} x {2, 3, 4 cards}` = **12 cells**

- GPU selection is `CUDA_VISIBLE_DEVICES=0,1,...`, first N devices
- Cell names are `<split>-<nomtp|mtp>-<N>`, e.g. `tensor-nomtp-4`
- Run order is 4 → 3 → 2 cards, so the fragile combinations come last and earlier
  results survive a failure

**N=1 must also be measured on any platform where the model fits on one card.** The
13,115 MiB model fits in 16GB — it was measured on V100 16GB.
**With that cell, `tg_1` in the derived metrics becomes a measurement rather than a
derivation** (section 5). The core comparison is still 2, 3 and 4 cards; N=1 serves
as the baseline.

### 1-7. Measurement source

**Do not parse the server log.** Take the values from the `timings` object in the
`/completion` response.

`prompt_n` / `prompt_ms` / `prompt_per_second` / `cache_n` /
`predicted_n` / `predicted_ms` / `predicted_per_second` /
`draft_n` / `draft_n_accepted`

Record wall-clock time (`wall_s`) alongside.

### 1-8. Telemetry

Recorded alongside each cell at one-second intervals. **Clock and throttle reasons
were added in v1.2.**

```
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.current.sm,clocks_throttle_reasons.active --format=csv,noheader -l 1
```

**Whether utilization sums above 100%** is the first-order check that parallelism is
working.

> **`clocks_throttle_reasons.active` is a bitmask and `0x1` is GpuIdle — not
> throttling.** Every idle card sets it, so counting "non-zero" over-reports badly.
> This matters most under layer split, which **computes on one card at a time, so
> more cards means more idle samples** — counting those as throttling produces the
> nonsensical pattern "throttling decreases as cards are added".
>
> The bits that actually hold the clock down are:
>
> ```
> therm = 0x08 HwSlowdown | 0x20 SwThermal | 0x40 HwThermal
> pcap  = 0x04 SwPowerCap | 0x80 HwPowerBrake
> ```
>
> **Aggregate thermal and power separately or the cause is indistinguishable.** To
> check an aggregation, read the bitmask distribution straight from the raw CSV:
>
> ```
> awk -F, '{gsub(/ /,"",$8); c[$8]++} END {for (k in c) print k, c[k]}' <cell>.smi.csv
> ```

---

## 2. Reproduction

```bash
# 0) Install NCCL first. It does not come with CUDA (section 1-2).
apt-get install -y libnccl2=<version>+cuda<your CUDA> libnccl-dev=<version>+cuda<your CUDA>

# 1) Build. Check that cmake prints "Found NCCL".
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<arch> -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA_NCCL=ON
cmake --build llama.cpp/build --config Release -j 8

# 2) Quantize the model (source on HuggingFace, section 1-1)
llama-quantize --tensor-type blk.64=q8_0 RVN-BF16-mtp.gguf qwen38-heretic-Q3KM-mtp.gguf Q3_K_M
sha256sum qwen38-heretic-Q3KM-mtp.gguf        # record in the report

# 3) Build the fixture — pinned revision from Wikisource
python3 harness/make_fixture.py --pin          # once
python3 harness/make_fixture.py --port 8099 --out fixtures/mujeong.json

# 4) Verify token counts (do not skip — an overflow is truncated silently)
python3 harness/server_bench.py --check

# 5) Manual probe: bring up the heaviest cell (N=2, tensor, MTP on) and confirm it loads
# 6) Measure
python3 harness/server_bench.py

# 7) Validate (checklist in section 4)
```

Do not skip step 5. If the tightest cell will not load you have to run all 12 again.

The harness **skips completed cells** when restarted. Cells that ended in an error are
retried.

---

## 3. Permitted variation between platforms

| Item | Note |
|---|---|
| `CMAKE_CUDA_ARCHITECTURES` | Match the card |
| Driver / CUDA Toolkit version | Record in the report |
| NCCL version | Record. **Without it the measurement is void** |
| Number of available GPUs | Leave the corresponding cells empty if fewer than four |
| **Power limit (W)** | **A card specification — record it in the environment section.** If it was adjusted during measurement, state that and the value |
| CPU / RAM / PCIe generation / topology | Record in the environment section |

**Changing anything not on this list means it is not a v1.2 measurement.**

---

## 4. Validation checklist

Check after every run. A cell that fails any of these is void.

- [ ] No `NCCL not compiled in` in the startup log
- [ ] No `internal AllReduce init failed` in the startup log
- [ ] `prompt_n` matches the targets in 1-3 in every cell
- [ ] `cache_n` is 0 on the cold turn
- [ ] `cache_n` is at least 22,000 on warm turns (cache hit)
- [ ] `predicted_n` is 400 on every turn
- [ ] Warm 4-turn `predicted_per_second` spread is within 5% — **MTP off cells only**
- [ ] **Thermal and power throttle ratios aggregated and recorded** (criteria in 1-8)
- [ ] **No Xid in the kernel log during the run** (`dmesg -T | grep -i xid`)

⚠️ **The 5% rule does not apply to MTP on cells.** Acceptance swinging with generated
content is intrinsic to speculative decoding, not a measurement error (up to 67% on
P104-100). **Record and report the spread instead** — identical averages with
different jitter feel different to a user, so the spread is itself a reportable
metric.

⚠️ **Cells run immediately after a crash can be contaminated.** On V100, the two cells
that ran right after `tensor-mtp-4` died with Xid 79 showed prefill cut in half and
carried NCCL messages in the server log. **Reboot and re-measure, and do not discard
the contaminated originals** — they are the evidence for the judgement.

---

## 5. Derived metric formulas

Every report uses the same formulas.

**Warm turn time** = mean `wall_s` over the four warm turns

**Session time** = `cold wall_s + warm turn time x N`

**Session crossover** — the point at which configuration A starts beating B

```
N = (cold_A - cold_B) / (warm_B - warm_A)
```

**Tensor scaling efficiency**

```
efficiency(N) = (tg_tensor(N) / tg_1) / N
```

`tg_1` is **a measured value** (1-6). On platforms where the model does not fit on one
card, **substitute the layer-split tg and state that it is derived** — the premise
being that layer split computes on one card at a time, so its figure is the
single-card speed. **V100 confirmed that premise to within 1%** (measured 27.71 at one
card against 27.97 / 27.78 / 27.29 for layer split at two, three and four).

**MTP batch cost multiple** — what decides whether speculative decoding pays

```
steps          = n_predict - draft_n_accepted
cost per step  = predicted_ms(MTP) / steps
cost per token = predicted_ms(no MTP) / n_predict
cost multiple  = cost per step / cost per token
tokens/step    = n_predict / steps
```

**If tokens per step is below the cost multiple, MTP is a loss.**

**Break-even acceptance**

```
draft length d = draft_n / steps
1 + d x acc = cost multiple  ->  acc = (cost multiple - 1) / d
```

**Roofline effective bandwidth**

```
effective bandwidth = (model size / N) / time the card actually spends
```

Under layer split a card is active for only 1/N of the total time, so use that time.
A low ratio against the spec figure means **it is not bandwidth bound.**

---

## 6. Known traps

| Symptom | Cause and response |
|---|---|
| `-sm row` reports `does not support split buffers` | A path removed from the CUDA backend. **Use `-sm tensor`.** The same change also broke row split in sd.cpp — the source-level account is in the METHOD of [`image-model-split-bench`](https://github.com/pyys/image-model-split-bench) |
| Tensor-parallel numbers look oddly low | NCCL not built in. Caught by the section 4 checklist |
| NCCL `Cuda failure 2 'out of memory'` (`transport/p2p.cc`) | The model loaded but **there is no VRAM left for NCCL P2P buffers.** Unrelated to topology. `NCCL_P2P_DISABLE=1` routes through the host and **breaks comparability.** Reducing context is the real fix, but `-c 27000` is fixed, so record the cell as failed |
| Compute buffer OOM with `layer` + MTP + few GPUs | The MTP draft context does not fit on top. Record as failed |
| **`Xid 79 — GPU has fallen off the bus`** | **Reproduced on `tensor` + MTP + 4 cards** (V100, twice). Record the cell as failed and **do not retry.** For contamination of following cells see section 4 |
| `llama_params_fit is not implemented for SPLIT_MODE_TENSOR` | A warning only. `-ngl` and `-c` are given explicitly, so it does not apply |
| `backend sampling not supported with SPLIT_MODE_TENSOR; using CPU` | Intrinsic to tensor parallelism. Treat as a controlled variable |
| Prompt exceeds `n_ctx` | llama.cpp **truncates silently** and the measurement is contaminated. Do not skip `--check` |

### 6-2. ⚠️ Tensor parallelism changes the generated output itself

With the same seed and the same prompt, **changing the split mode or the GPU count
changes the output tokens under tensor parallelism.** The all-reduce ordering differs,
so floating-point results diverge slightly.

Confirmed on both platforms.

| | Layer split | Tensor parallelism |
|---|---|---|
| P104 `draft_n` | 4 and 3 cards **identical** | 675 / 642 / 660 … **all different** |
| V100 `draft_n` / `accepted` | **2440 / 780** at 4, 3, 2 and 1 cards | 2547/744, 2599/723 **all different** |

So **MTP acceptance comparisons between tensor-parallel cells are not controlled for
generated content.** Do not read an acceptance difference as an effect of GPU count.
MTP-off metrics (pp, tg) are independent of generated content and unaffected.

---

## 7. What must not be reported

- **Do not use a synthetic benchmark (`llama-bench` `pp512` and friends) as a stand-in
  for real prefill.** The batch is small enough to reverse the ranking of the split
  modes. Use it as a reference figure only, and say so
- **Do not put values measured under a different methodology version in the same
  table**

---

## 8. Version history

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-09-10 | Initial freeze, based on the P104-100 x4 run |
| v1.1 | 2026-09-10 | Section 4 spread check applies to **MTP off cells only** (MTP jitter is intrinsic). **Section 6-2 added** — tensor parallelism changes the output as N changes |
| **v1.2** | **2026-09-15** | **`clocks.current.sm` and `clocks_throttle_reasons.active` added to telemetry in 1-8** (throttle aggregation was impossible under the v1.1 harness) · **NCCL warning absorbed into 1-2** · **N=1 moved from recommendation to requirement in 1-6** · power limit added to section 3 · throttle, Xid and crash contamination added to section 4 · **Xid 79 added to section 6** · reproduction procedure added as section 2 |

**v1.0, v1.1 and v1.2 share the same measurement conditions.** Only recorded fields
and validation rules were added, so **results from earlier versions can go in the same
table.** Note that measurements before v1.2 carry no throttle record.

---

## References

{ref 1.} through {ref 10.} are in [`FINDINGS.md`](FINDINGS.md#references).

{ref 11.} *0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF*.
https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF
(accessed 2026-09-10)

{ref 12.} *Yi Kwang-su, "Mujeong" (1917)* — Korean Wikisource. Public domain.
https://ko.wikisource.org/wiki/무정 (accessed 2026-09-10)
