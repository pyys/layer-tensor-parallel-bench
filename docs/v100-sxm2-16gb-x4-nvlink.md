# NVIDIA V100-SXM2 16GB x4 (NVLink, DGX-1) — layer split vs tensor parallelism

Measured 2026-10-19. Methodology **METHOD v1.4** ([`../METHOD.md`](../METHOD.md)).
Raw data [`../results/v100-sxm2-16gb-x4-nvlink.jsonl`](../results/v100-sxm2-16gb-x4-nvlink.jsonl).
Derived metrics live in
[`v100-sxm2-16gb-x4-nvlink-metrics.md`](v100-sxm2-16gb-x4-nvlink-metrics.md).

> Measurement conditions are in METHOD. This document holds only this platform's
> environment, results and interpretation.
> Conclusions that span platforms are in [`../FINDINGS.md`](../FINDINGS.md).

**The same card model as the earlier V100 report (V100-SXM2 16GB), measured in a factory
SXM2 system, a DGX-1, connected by NVLink.** The cards in the earlier run were SXM2
modules adapted to PCIe and had no NVLink. **This is the same card with the interconnect
changed from PCIe Gen3 x16 to NVLink** — although the whole system (power delivery,
cooling, host) changed with it (section 6).

**The DGX-1 is an eight-GPU system shared by several users.** This run used four of them,
GPUs 0~3, under the shared-system clause of METHOD v1.4 section 3 (devices selected by
UUID, telemetry limited to the devices under test).

**In short**, three conclusions from the earlier V100 run change here.

- **Four-card tensor scaling efficiency rises from 42% to 52%**
- **Tensor beats layer even on the cold prefill** (1.28x). On the earlier run layer was
  1.75x faster
- **`tensor-mtp-4` completed.** On the earlier V100 it died twice with Xid 79

---

## 1. Environment

### Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA Tesla V100-SXM2-16GB x4 (GV100, sm_70, Volta). GPUs 0~3 of an eight-GPU system |
| Form | **Factory SXM2 (DGX-1). NVLink** |
| VRAM | 16,384 MiB per card, 65,536 MiB total (the four used) |
| Memory bandwidth | 900 GB/s (HBM2) |
| Power limit | **300W** (not changed during measurement) |
| CPU / RAM | Intel Xeon E5-2698 v4 x2 (40 cores, 80 threads) / 220 GiB (as shown by `free -g`) |

**PCIe link** — all four cards used report `Gen3 x16 as the maximum`. In this run the
traffic between GPUs goes over NVLink, not PCIe.

```
index, name, pcie.link.gen.current, pcie.link.gen.max, pcie.link.width.current, pcie.link.width.max
0, Tesla V100-SXM2-16GB, 3, 3, 16, 16
1, Tesla V100-SXM2-16GB, 3, 3, 16, 16
2, Tesla V100-SXM2-16GB, 3, 3, 16, 16
3, Tesla V100-SXM2-16GB, 3, 3, 16, 16
```

**Topology** — **only the GPU 0~3 part** of `nvidia-smi topo -m`. The other four are other
users' devices and are not reproduced.

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV1     NV1     NV2     0-19,40-59      0
GPU1    NV1      X      NV2     NV1     0-19,40-59      0
GPU2    NV1     NV2      X      NV2     0-19,40-59      0
GPU3    NV2     NV1     NV2      X      0-19,40-59      0

  NV#  = Connection traversing a bonded set of # NVLinks
```

**All four are directly connected by NVLink, but the number of links differs by pair.**
One link is 25.781 GB/s (`nvidia-smi nvlink -s`). Each card has six links; the ones not in
the table above go to GPUs not used in this run.

| Pair | Links | | Pair | Links |
|---|---|---|---|---|
| 0–1 | **NV1** | | 1–2 | NV2 |
| 0–2 | NV1 | | 1–3 | NV1 |
| 0–3 | NV2 | | 2–3 | NV2 |

Following METHOD 1-6 (first N devices), **the two-card cells use an NV1 pair (0, 1)**,
the three-card cells 0, 1, 2, and the four-card cells 0~3. **A comparison of the same cell
on an NV2 pair is in the supplementary run** — decode differed by 0.1%
([`../supplements/gemma4-31b-v100-sxm2-16gb-x4-nvlink.md`](../supplements/gemma4-31b-v100-sxm2-16gb-x4-nvlink.md) section 5).

**P2P** — all four support P2P over NVLink (`nvidia-smi nvlink -c`, `P2P is supported: true`).

### Software

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 |
| Driver | 580.178.04 |
| CUDA Toolkit | 12.8 |
| llama.cpp | **465e49b9c (build 10830)** — **the same commit** as the other platforms |
| `CMAKE_CUDA_ARCHITECTURES` | **`70`** — **not the same binary** (built separately on this system) |
| **NCCL** | **2.26.2-1+cuda12.8** (pinned via apt) |
| Model | Qwen3.8-27B-Heretic Q3_K_M, 13,115 MiB (METHOD 1-1) |

### Model

The METHOD 1-1 file as is. **sha256 `727e7d8b3ab3af1a6eeb166f3d7aac93d51ac0cdc06200e8f1cbac9074591ca4`**
(13,752,763,616 bytes), matching the value in METHOD. The file was quantized on this
system, but the hash is identical, so it is byte-for-byte the same model as on the other
platforms.

> ⚠️ **Without NCCL installed this measurement is invalid.** Symptoms, before/after
> figures and how to check are in [METHOD 1-2](../METHOD.md). On this platform none of
> the sixteen server logs contained `NCCL not compiled in` or
> `internal AllReduce init failed`.

### Harness

[`../harness/server_bench.py`](../harness/server_bench.py) was run with **only paths and
device selection changed** — three things: the model and fixture paths, the default of
`BENCH_DEVICE_ORDER` set to the UUIDs of the GPUs used, and `-i <those UUIDs>` added to
the telemetry `nvidia-smi` (METHOD v1.4 section 3). Cell definitions, request parameters
and `-c 27000` are unchanged.

---

## 2. Results

- **Prefill (pp)** = how fast the prompt is read. `prompt_per_second`
- **Generation (tg)** = final output token rate. `predicted_per_second`
  (confirmed output, not drafts — `predicted_n = 400` on every turn)
- Cold is one run, warm is the mean of four turns
- **Percentages** are relative to **the slowest cell for that metric**, which is 100

| Metric | Baseline cell (100%) | Value |
|---|---|---|
| Cold pp | tensor / MTP on / 1 card | 664.9 t/s |
| Warm pp | tensor / MTP on / 1 card | 360.8 t/s |
| tg | layer / MTP off / 4 cards | 26.90 t/s |

### 2-1. Generation speed ranking (all 16 cells)

| Rank | Cell | tg | % |
|---|---|---|---|
| **1** | **tensor / MTP off / 4 cards** | **56.73** | **211%** |
| 2 | tensor / MTP on / 4 cards | 54.16 | 201% |
| 3 | tensor / MTP on / 3 cards | 50.27 | 187% |
| 4 | tensor / MTP off / 3 cards | 48.31 | 180% |
| 5 | tensor / MTP off / 2 cards | 42.40 | 158% |
| 6 | tensor / MTP on / 2 cards | 40.89 | 152% |
| 7 | layer / MTP on / 2 cards | 30.10 | 112% |
| 8 | layer / MTP on / 3 cards | 29.31 | 109% |
| 9 | layer / MTP on / 4 cards | 28.94 | 108% |
| 10 | layer / MTP on / 1 card | 28.90 | 107% |
| 11 | tensor / MTP on / 1 card | 28.18 | 105% |
| 12 | layer / MTP off / 2 cards | 27.97 | 104% |
| 13 | layer / MTP off / 3 cards | 27.39 | 102% |
| 14 | tensor / MTP off / 1 card | 27.17 | 101% |
| 15 | layer / MTP off / 1 card | 27.13 | 101% |
| 16 | layer / MTP off / 4 cards | 26.90 | **100%** |

**Three things to read off this.**

1. **The six tensor cells with two or more cards own the top of the table.** The slowest
   of them (40.89) is 36% faster than the fastest layer cell (30.10) — the same picture as
   the earlier V100.
2. **First place is still `tensor / MTP off / 4 cards`, but second is
   `tensor / MTP on / 4 cards`.** On the earlier V100 that cell could not be measured
   because of Xid 79 (section 5).
3. **The cold pp leader is also tensor** — `tensor / MTP off / 4 cards` at 1,968.2 (296%).
   On the earlier V100 the cold leader was `layer / MTP off / 4 cards`, which came last on
   generation. **Here one cell takes first place on cold, warm and generation** (3-3).

### 2-2. Layer split, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| 4 | 1,533.9 | 231% | 473.8 | 131% | 26.90 | **100%** | 16.5s |
| 3 | 1,412.9 | 213% | 488.5 | 135% | 27.39 | 102% | 16.2s |
| 2 | 1,062.7 | 160% | 485.9 | 135% | 27.97 | 104% | 15.8s |
| **1** | 720.7 | 108% | 472.5 | 131% | **27.13** | 101% | 16.3s |

**N=1 was measured** (METHOD 1-6), so `tg_1` in the derived metrics is not derived.

⚠️ **The warm spread of `layer-nomtp-4` is 5.01%** — 0.01 points over the METHOD
section 4 limit (5% or less; section 4 below). **It is reported as a borderline value, not
re-measured.** Its tg (26.90) is in the same range as the other layer MTP-off cells on
this platform (27.13~27.97) and does not affect any conclusion.

### 2-3. Layer split, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | 1,180.3 | 178% | 382.3 | 106% | 28.94 | 108% | **32.0%** | 15.8s |
| 3 | 1,102.8 | 166% | 384.5 | 107% | 29.31 | 109% | **32.0%** | 15.6s |
| 2 | 893.3 | 134% | 390.3 | 108% | **30.10** | **112%** | **32.0%** | 15.2s |
| 1 | 682.5 | 103% | 385.3 | 107% | 28.90 | 107% | **32.0%** | 15.8s |

**All four cells accept 32.0%, and drafts / accepted over the four warm turns are
`2440 / 780` — exactly as on the earlier V100.** METHOD 6-2's statement that layer split
is deterministic regardless of N **reproduces on a different system.**

### 2-4. Tensor parallelism, MTP off

| N | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|
| **4** | **1,968.2** | **296%** | **903.8** | **250%** | **56.73** | **211%** | **7.9s** |
| 3 | 1,458.4 | 219% | 774.6 | 215% | 48.31 | 180% | 9.3s |
| 2 | 1,188.2 | 179% | 693.5 | 192% | 42.40 | 158% | 10.5s |
| **1** | 705.6 | 106% | 445.4 | 123% | 27.17 | 101% | 16.4s |

**At one card layer 27.13 and tensor 27.17 differ by 0.15%.** The split mode does nothing
on a single card — a smaller gap than the 0.97% on the earlier V100.

### 2-5. Tensor parallelism, MTP on

| N | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|
| **4** | 1,625.7 | 245% | 596.3 | 165% | **54.16** | **201%** | 27.5% | 8.7s |
| 3 | 1,276.6 | 192% | 532.0 | 147% | 50.27 | 187% | 31.4% | 9.4s |
| 2 | 1,082.2 | 163% | 496.8 | 138% | 40.89 | 152% | 27.8% | 11.3s |
| 1 | 664.9 | **100%** | 360.8 | **100%** | 28.18 | 105% | 32.0% | 16.3s |

**The four-card cell completed** — the combination that died twice with Xid 79 on the
earlier V100 (section 5).

Acceptance differs by cell (27.5% / 31.4% / 27.8%). Tensor parallelism changes the
generated content itself as N changes (METHOD 6-2), so **do not read these differences as
an effect of the GPU count.** One card has no split and gives the same `2440 / 780` as
layer split.

### 2-6. Temperature and throttling

Aggregated from the METHOD 1-8 telemetry, **for the cards each cell used.**

| Cell | Peak | Mean | therm % | pcap % |
|---|---|---|---|---|
| layer-nomtp 4 / 3 / 2 / 1 | 52 / 59 / 63 / 67 | 44.8 / 50.9 / 55.9 / 62.9 | 0 / 0 / 0 / 0 | 3.7 / 4.6 / 6.7 / **45.2** |
| layer-mtp 4 / 3 / 2 / 1 | 60 / 60 / 63 / 66 | 48.2 / 52.1 / 55.5 / 61.4 | 0 / 0 / 0 / 0 | 2.7 / 6.5 / 5.3 / **33.0** |
| tensor-nomtp 4 / 3 / 2 / 1 | 64 / 66 / 71 / 66 | 53.6 / 59.7 / 63.3 / 63.6 | 0 / 0 / 0 / 0 | 3.1 / 2.5 / 5.2 / **50.0** |
| tensor-mtp 4 / 3 / 2 / 1 | 64 / 66 / 70 / 66 | 53.5 / 59.1 / 62.3 / 62.6 | 0 / 0 / 0 / 0 | 1.2 / 2.4 / 6.1 / **36.2** |

**Thermal throttling is 0% in every cell.** Peak 71 °C.

**Only the single-card cells show the power-cap bit, 33~50% of samples.** From two cards
up it is 1~7%. We **presume** a single card reaches the 300W cap because it computes
without pause (not verified). **The single-card tg (27.13 / 27.17) is within 3.1% of layer
split at two to four cards (26.90~27.97)**, so there is no sign that the power cap pulled
the single-card baseline down.

### Validation checklist (METHOD section 4)

| Item | Result |
|---|---|
| No `NCCL not compiled in` | ✅ all 16 cells |
| No `internal AllReduce init failed` | ✅ all 16 cells |
| `prompt_n` matches the targets | ✅ 23,037 / 23,250 / 23,883 / 24,650 / 25,436 (within ±1%) |
| Cold `cache_n` = 0 | ✅ |
| Warm `cache_n` ≥ 22,000 | ✅ 22,521 → 24,646 |
| `predicted_n` = 400 | ✅ every turn |
| Warm tg spread within 5% (MTP off cells) | ⚠️ **`layer-nomtp-4` 5.01%**, the other seven 0.5~1.7% |
| Thermal and power throttling aggregated | ✅ (2-6) |
| No kernel Xid during the run | **not checked** — see below |

**Xid was not checked.** The kernel log (`dmesg`) records the whole system, including
events on GPUs not used in this run (other users' devices). Under the shared-system
policy it was not read. **The sixteen server logs show no GPU errors and every cell
completed.**

---

## 3. Interpretation

The numbers and working for the derived metrics (session crossover, scaling efficiency,
MTP batch economics, roofline) are in
[`v100-sxm2-16gb-x4-nvlink-metrics.md`](v100-sxm2-16gb-x4-nvlink-metrics.md).

### 3-1. Layer split still does not use extra cards for decode — same as the earlier V100

```
1 card 27.13  ->  2 cards 27.97  ->  3 cards 27.39  ->  4 cards 26.90
earlier V100  27.71 / 27.97 / 27.78 / 27.29
```

**Changing the interconnect leaves layer-split decode at single-card speed.** Layer split
passes one activation across each card boundary, so the link has almost nothing to do.

Prefill follows the same pattern — cold pp 720.7 → 1,533.9 (**2.13x**), warm pp 472.5 /
485.9 / 488.5 / 473.8, independent of the card count.

### 3-2. Tensor parallelism scales better than on the earlier V100

| N | Earlier V100 tg | Earlier efficiency | **NVLink tg** | **NVLink efficiency** |
|---|---|---|---|---|
| 2 | 41.64 | 75% | 42.40 | **78%** |
| 3 | 42.79 | 51% | 48.31 | **59%** |
| 4 | 46.62 | 42% | **56.73** | **52%** |

**Two cards are nearly the same (+1.8%), and the gap widens with more cards** (three
+12.9%, four +21.7%). The two-card cell is an NV1 pair with a single link, and it still
matches or beats PCIe.

FINDINGS 5 says that raising PCIe bandwidth 16x did not improve scaling. This run shows
that **moving to NVLink did.** Together they suggest that what decides scaling may be
less the size of the PCIe bandwidth than **NVLink's direct path (GPU-to-GPU traffic that
does not cross a host bridge)** — **not settled** (section 6).

### 3-3. Tensor wins the cold prefill too — the opposite of every earlier platform

```
cold   tensor 1,968.2  vs  layer 1,533.9   =  tensor 1.28x
warm   tensor   903.8  vs  layer   473.8   =  tensor 1.91x
```

| 4 cards | Cold | Warm |
|---|---|---|
| P104 Gen1 x4 | layer ahead 1.91x | tensor ahead 1.51x |
| Earlier V100 (PCIe) | layer ahead 1.75x | tensor ahead 1.22x |
| **V100 NVLink** | **tensor ahead 1.28x** | **tensor ahead 1.91x** |

**On every platform so far layer split won the cold prefill. Here, for the first time,
tensor does.** Tensor parallelism issues an all-reduce per prefill batch; we read this as
that cost having become small enough on NVLink to outweigh the advantage of the layer
pipeline. The four-card tensor cold pp went from 887.4 on the earlier V100 to 1,968.2 here,
**2.22x** — on the same card model with the link changed.

As a result **the session crossover is negative** — tensor wins from the first (cold)
turn (metrics section 1).

### 3-4. MTP gains under layer split; under tensor the sign varies with the card count

```
layer   4 cards 26.90 -> 28.94   +7.6%   (earlier V100 +3.5%)
tensor  4 cards 56.73 -> 54.16   -4.5%   (earlier V100 failed)
        3 cards 48.31 -> 50.27   +4.1%   (earlier V100 -12.1%)
        2 cards 42.40 -> 40.89   -3.6%   (earlier V100 -5.9%)
```

**The layer-mode gain is twice the earlier V100's.** Acceptance is 32.0% on both, and the
cost multiple fell from 1.89 to 1.82.

**Under tensor the sign changes with the card count within ±5%.** Tensor-mode acceptance
differs by N (27.5 / 31.4 / 27.8%, METHOD 6-2), so **do not read the three-card gain as a
property of three cards** — the cost multiple is nearly the same at 1.86~1.90, and the
cell whose acceptance came out higher is the one that gained.

---

## 4. MTP spread

Spread of tg over the four warm turns, MTP off and on side by side.

| Cell | Four warm turns, tg | Spread |
|---|---|---|
| layer / MTP off / 4 cards | 27.12 / 27.70 / 26.38 / 26.41 | **5.0%** ⚠️ |
| layer / MTP off / 2 cards | 28.08 / 27.99 / 27.95 / 27.86 | 0.8% |
| layer / MTP on / 4 cards | 27.75 / 27.85 / 30.49 / 29.68 | **9.9%** |
| layer / MTP on / 2 cards | 28.68 / 28.40 / 32.54 / 30.80 | **14.6%** |
| tensor / MTP off / 4 cards | 56.52 / 56.84 / 56.80 / 56.75 | **0.6%** |
| tensor / MTP on / 4 cards | 51.80 / 53.96 / 55.93 / 54.93 | **8.0%** |
| tensor / MTP on / 3 cards | 49.21 / 49.34 / 51.94 / 50.60 | 5.5% |
| tensor / MTP on / 2 cards | 37.35 / 41.67 / 42.20 / 42.32 | **13.3%** |

**MTP-off tensor cells are the most stable (0.5~0.9%); MTP-on cells swing by 5~15%.** The
same pattern as the earlier V100.

⚠️ The 5.0% (5.01% exactly) of `layer / MTP off / 4 cards` is unusual for an MTP-off cell:
the third and fourth turns are low at 26.4. The other layer MTP-off cells in this run are
0.8~1.7%.

---

## 5. Failures and anomalies

**No cell failed.** All sixteen completed.

**That `tensor-mtp-4` completed is itself a result of this platform.** On the earlier V100
this combination died twice with Xid 79 (GPU fallen off the bus), and the remaining
hypothesis was **PCIe transaction frequency** (FINDINGS 7). Here traffic between GPUs runs
over NVLink, not PCIe. **Consistent with the hypothesis, but not a test of it** — the whole
system (power delivery, cooling) changed as well, and Xid could not be checked in the
kernel log (section 2 checklist). The server logs and results show nothing abnormal.

---

## 6. Limits specific to this platform

1. **The comparison with the earlier V100 is not single-variable.** The card model is the
   same, but adapted cards against factory SXM2 means different power delivery and
   cooling, and the host (EPYC 7232 / 32GB → Xeon E5-2698 v4 x2), the driver (580.173 →
   580.178) and the build (`61;70` → `70`, same commit) changed too. The differences in
   3-2 and 3-3 are not attributed to NVLink alone
2. **The number of NVLinks differs by pair.** The two-card cells use an NV1 pair; the
   three- and four-card cells mix NV1 and NV2. That the link count does not matter for
   two-card decode was checked only in a supplementary run
3. **Shared system.** Four of eight GPUs were used, and other users' work may have been
   running on the other four during the run. The host CPU, memory and PCIe root are shared
4. **Xid was not checked** (section 2), because the kernel log of a shared system was not
   read
5. **The spread of `layer-nomtp-4` is 0.01 points over the METHOD limit** (2-2)
6. **The single-card cells show the power-cap bit in 33~50% of samples** (2-6). No effect
   on the single-card baseline is visible, but it was not isolated

---

## 7. Raw data and cell validity

| | Contents |
|---|---|
| [`../results/v100-sxm2-16gb-x4-nvlink.jsonl`](../results/v100-sxm2-16gb-x4-nvlink.jsonl) | One JSON line per cell, 16 lines |

### Cell validity — all 16 cells completed

| Class | Cells | Basis |
|---|---|---|
| **ok (15)** | the 15 cells other than `layer-nomtp-4` | no NCCL messages, checklist passed (Xid not checked) |
| **borderline (1)** | `layer-nomtp-4` | warm spread 5.01% (limit 5%). Reported as is, not re-measured (2-2) |

Every line carries `validity`, `devices` (GPU numbers used), `concurrent_with` (`null`, no
concurrent runs) and `note`. The `note` of `layer-nomtp-4` records the spread.
