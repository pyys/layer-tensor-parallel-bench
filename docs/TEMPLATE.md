# <platform> — layer split vs tensor parallelism

Measured YYYY-MM-DD. Methodology **METHOD v1.1** ([`../METHOD.md`](../METHOD.md)).
Raw data: [`../results/<platform>.jsonl`](../results/).
Derived metrics live in [`<platform>-metrics.md`](<platform>-metrics.md).

> Measurement conditions live in METHOD. This document covers only this platform's
> **environment, results and interpretation**.

One or two sentences on why this platform is worth measuring — what is unusual about
it, and what question it answers that other platforms do not.

---

## 1. Environment

### Hardware

| Item | Value |
|---|---|
| GPU | |
| VRAM | per card and total, plus usable |
| Spec memory bandwidth | |
| CPU / RAM | |

**PCIe link** — paste the raw output. **State whether the reported generation and
width are the ceiling or an idle downshift**, and confirm under load.

```
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
```

**Topology** — paste raw `nvidia-smi topo -m` output, plus only the legend lines that
appear in it.

```

```

**P2P** — `nvidia-smi topo -p2p r`.

### Software

| Item | Value |
|---|---|
| OS | |
| Driver / CUDA | |
| llama.cpp | commit hash + build number |
| `CMAKE_CUDA_ARCHITECTURES` | |
| **NCCL** | version. **Absent means this measurement is void** |
| Model | name and quantization (see below) |

### Model

Source repository, source file and size; the exact `llama-quantize` command; the
resulting size and sha256; and why this quantization was chosen.

⚠️ Do not use a markdown footnote here — GitHub breaks footnotes that contain code
blocks. Keep this as a plain subsection.

---

> ## ⚠️ Without the NCCL library this measurement is void
>
> Keep this box. Adapt only if the platform's behaviour differs from what the P104
> report describes. See [`p104-100-x4.md`](p104-100-x4.md) for the full text and the
> before/after numbers.

---

## 2. Results

- **prefill (pp)** = how fast the prompt is read. `prompt_per_second`
- **generation (tg)** = final output token rate. `predicted_per_second`.
  Accepted output tokens, not drafts — `predicted_n` is 400 on every turn
- Cold is a single turn; warm is the mean of four
- **Percentages** are normalized **per metric**, against the slowest cell for that
  metric on this platform. Name the three baselines here.

| Metric | Baseline cell (100%) | Value |
|---|---|---|
| Cold pp | | |
| Warm pp | | |
| tg | | |

### 2-1. Layer split, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | | |
| 3 | | | | | | | |
| 2 | | | | | | | |

### 2-2. Layer split, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | | | | | | | | |
| 3 | | | | | | | | |
| 2 | | | | | | | | |

### 2-3. Tensor parallelism, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| 4 | | | | | | | |
| 3 | | | | | | | |
| 2 | | | | | | | |

### 2-4. Tensor parallelism, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | | | | | | | | |
| 3 | | | | | | | | |
| 2 | | | | | | | | |

*(Add an N=1 row to 2-1 and 2-3 if the platform can measure it. That also removes the
substitution in the metrics document's scaling section.)*

### 2-5. Generation speed, all cells ranked

| Rank | Cell | tg | % |
|---|---|---|---|
| 1 | | | |

State plainly whether the two split modes separate cleanly or interleave, and
**whether the cold-pp ranking and the tg ranking disagree.**

### Verification checklist (METHOD section 4)

| Check | Result |
|---|---|
| No `NCCL not compiled in` | |
| No `internal AllReduce init failed` | |
| `prompt_n` matches targets | |
| Cold `cache_n` = 0 | |
| Warm `cache_n` ≥ 22,000 | |
| `predicted_n` = 400 | |
| Warm `tg` spread within 5% (**MTP-off cells only**) | |

List failed cells with their cause.

---

## 3. Interpretation

> ⚠️ If this is the only platform measured so far, say so here and mark the
> conclusions as specific to this hardware until another platform confirms them.

Numbers and derivations are in [`<platform>-metrics.md`](<platform>-metrics.md).

Cover at least these, and say explicitly where this platform **differs** from others
already measured:

- Does layer split gain anything on decode as N rises?
- Does the cold/warm prefill ranking reverse?
- What share of a warm turn is generation?
- Is decode bandwidth-bound? (roofline + the Q6/Q3 comparison if available)
- Does speculative decoding pay? (batch cost multiplier)

Separate what holds only here from what transfers.

---

## 4. MTP variance

Spread of `tg` across the four warm turns, MTP-off cells alongside MTP-on ones.
Per METHOD 4, the 5% rule does not apply to MTP-on cells — **report the spread as a
result**, not as a failure.

| Cell | Warm turns | Spread |
|---|---|---|
| | | |

---

## 5. Appendix — additional fixtures (optional)

Only if a second fixture was run. State clearly whether it is reproducible by third
parties, and **do not let any claim in the body depend on a non-reproducible one.**

⚠️ When comparing fixtures, remember that two fixtures usually differ in several
dimensions at once. **Do not attribute a difference to a single factor unless it was
isolated.**

---

## 6. Limitations specific to this platform

Do not repeat limitations already listed in METHOD or the README. Cover at least:

- Is the 1-GPU baseline measured or substituted?
- Which cells are missing, and why?
- Any determinism caveats (e.g. tensor parallelism changing output as N changes)
