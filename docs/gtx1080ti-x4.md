# NVIDIA GTX 1080 Ti x4 on mixed PCIe Gen3 x4/x8 — layer split vs tensor parallelism

Measured 2026-11-12. Methodology **METHOD v1.5** ([`../METHOD.md`](../METHOD.md)) — v1.5 adds an
**asymmetric-device selection** clause to section 3 for this run; the measurement conditions
themselves are the same as v1.0~v1.4.
Raw data [`../results/gtx1080ti-x4.jsonl`](../results/gtx1080ti-x4.jsonl) — main runs `validity: ok` / `invalid`,
auxiliary runs `side`, the re-run `invalid`.
Derived metrics live in [`gtx1080ti-x4-metrics.md`](gtx1080ti-x4-metrics.md).

> Measurement conditions are in METHOD. This document holds only **this platform's
> environment, results and interpretation.**
> Conclusions that span platforms are in [`../FINDINGS.md`](../FINDINGS.md).

**Same host, same driver, same binary and same model file as the [P104-100 x4 Gen1 x1](p104-100-x4-gen1x1.md)
run — only the cards changed** (the [Gen1 x4 run](p104-100-x4.md) shares the commit and the model, but not the
host, driver or build). The P104 is fused to Gen1 x4; the 1080 Ti is the same Pascal generation (sm_61)
but **links at Gen3**. With the link 4~8x wider and card compute about 1.8x, this run checks whether
the earlier conclusions — "tensor parallelism wins on both prefill and decode", "layer split buys no
decode speed" — still hold, and **how tensor prefill changes with the number of cards.**

**What it confirms and what it overturns.** Both conclusions hold. What is new is that **tensor-parallel
cold prefill rises with the card count** (337 → 399 → 461 t/s from two to four cards), the opposite of
V100 on PCIe Gen3 x16, where it fell.

---

## 1. Environment

### Hardware

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 Ti x4 (GP102, sm_61, Pascal) |
| VRAM | 11,264 MiB per card / 44 GiB total |
| Memory bandwidth | 484 GB/s (specification) |
| Power limit | 250W (card default, not changed) |
| CPU / RAM | Intel Core i7-5820K, 6 cores 12 threads, 28 PCIe lanes / 16GB DDR4 (two channels) |
| Mainboard | Gigabyte GA-X99-UD4P |
| Cooling | **User fan-curve daemon** (40°C 30% → 100% from 80°C, via NVML). The P104 runs used the driver's default curve |

**PCIe link** — the cards are Gen3 x16 capable; the actual width is **set by the slot** (28 CPU lanes =
4 + 8 + 8 + 8). `pcie.link.gen.current` = 3 confirmed under load.

```
index, name, memory.total [MiB], power.limit [W], pcie.link.gen.max, pcie.link.width.max
0, NVIDIA GeForce GTX 1080 Ti, 11264 MiB, 250.00 W, 3, 16
1, NVIDIA GeForce GTX 1080 Ti, 11264 MiB, 250.00 W, 3, 16
2, NVIDIA GeForce GTX 1080 Ti, 11264 MiB, 250.00 W, 3, 16
3, NVIDIA GeForce GTX 1080 Ti, 11264 MiB, 250.00 W, 3, 16
```

| GPU | Slot | CPU root port | Actual link |
|---|---|---|---|
| 0 | PCIE_3 | 00:01.1 | **Gen3 x4** |
| 1 | PCIE_2 | 00:02.0 | Gen3 x8 |
| 2 | PCIE_1 | 00:03.0 (x16 port bifurcated x8+x8) | Gen3 x8 |
| 3 | PCIE_4 | 00:03.2 (the other x8 of the same port) | Gen3 x8 |

**Topology** — `nvidia-smi topo -m`

```
        GPU0  GPU1  GPU2  GPU3  CPU Affinity  NUMA Affinity
GPU0     X    PHB   PHB   PHB   0-11          0
GPU1    PHB    X    PHB   PHB   0-11          0
GPU2    PHB   PHB    X    PHB   0-11          0
GPU3    PHB   PHB   PHB    X    0-11          0

PHB = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
```

**P2P** — `nvidia-smi topo -p2p r` reports `OK` for every pair. No NVLink, no PCIe switch.

### Device selection (METHOD v1.5 section 3, asymmetric devices)

This system **mixes link widths** (one x4, three x8). Devices were chosen under the v1.5 clause; the
difference from the system as a whole is shown below.

| N | **Main-run devices** | Main-run links | Auxiliary-run devices (first N) | Auxiliary links |
|---|---|---|---|---|
| 4 | 0,1,2,3 | x4·x8·x8·x8 (= the whole system) | — | — |
| 3 | **1,2,3** | x8·x8·x8 | 0,1,2 | x4·x8·x8 |
| 2 | **1,3** | x8·x8 | 0,1 | x4·x8 |

- Two cards use GPUs 1 and 3: both x8, on different root ports, avoiding GPU 2 (top slot, the hottest)
- **Four cards cannot leave out the x4 device.** The four-card ring all-reduce is bound by the x4 link
- The auxiliary runs are kept, not discarded — [section 8](#8-raw-data-and-cell-validity)

### Software

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-147 |
| Driver / CUDA | 580.178.04 / CUDA 12.8 (V12.8.93) |
| llama.cpp | `465e49b9cea78a68b9c244ffb48d0ee24a82873d` (build 10830) — **the same binary as the P104 Gen1 x1 run** (`61` only). Same commit as the Gen1 x4 and V100 runs, different build (`61;70`) |
| `CMAKE_CUDA_ARCHITECTURES` | 61 |
| **NCCL** | 2.26.2 (`libnccl2 2.26.2-1+cuda12.8`), `GGML_CUDA_NCCL=ON` |
| Model | Qwen3.8-27B Heretic Q3_K_M (below) |

### Model

**The same file** as the other platforms. Quantization command and rationale are in
[METHOD 1-1](../METHOD.md). `qwen38-heretic-Q3KM-mtp.gguf`, 13,115 MiB,
sha256 `727e7d8b3ab3af1a6eeb166f3d7aac93d51ac0cdc06200e8f1cbac9074591ca4` (re-checked before the run).

> ⚠️ **Without NCCL installed this measurement is invalid.** Symptoms, before/after numbers and
> how to check are in [METHOD 1-2](../METHOD.md). This platform behaves as METHOD describes.

---

## 2. Results

- **Prefill (pp)** = `prompt_per_second`, **generation (tg)** = `predicted_per_second` (committed
  output, `predicted_n` = 400 in every turn)
- Cold is one run, warm the mean of four turns. **★ = invalid cell** (section 5) — values are shown but
  excluded from the ranking, the percentage baselines and the derived metrics
- **Percentages (%)** take the slowest valid cell of each metric as 100

| Metric | Baseline cell (100%) | Value |
|---|---|---|
| Cold pp | layer / MTP on / 2 cards | 279.5 t/s |
| Warm pp | layer / MTP on / 2 cards | 139.7 t/s |
| tg | layer / MTP on / 4 cards | 9.97 t/s |

### 2-1. Generation speed ranking (main runs, valid cells)

| Rank | Cell | tg | % |
|---|---|---|---|
| 1 | tensor / off / 4 | **28.94** | 290% |
| 2 | tensor / on / 4 | 26.92 | 270% |
| 3 | tensor / off / 3 | 25.59 | 257% |
| 4 | tensor / on / 3 | 22.09 | 222% |
| 5 | tensor / off / 2 | 20.83 | 209% |
| 6 | tensor / on / 2 | 17.30 | 174% |
| 7 | layer / off / 2 | 12.45 | 125% |
| 8 | layer / on / 2 | 11.05 | 111% |
| 9 | layer / on / 3 | 10.07 | 101% |
| 10 | layer / on / 4 | 9.97 | 100% |
| — | ★ layer / off / 3 | 12.20 | (invalid) |
| — | ★ layer / off / 4 | 11.92 | (invalid) |

**Three things to read**

- **The two split modes separate cleanly.** All six tensor cells rank above every layer cell — as on P104 and V100
- **Cold pp ranks differently.** At four cards layer split is faster on cold prefill (551.6 against 460.6);
  on warm prefill tensor is 2.3x faster
- **The top cell is tensor / MTP off / 4 cards** — the same as in all four earlier runs. MTP loses in every cell (section 4)

### 2-2. Layer split, MTP off

| N | Devices | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 ★ | 0-3 | 551.6 | — | 149.6 | — | 11.92 | — | 38.5 s |
| 3 ★ | 1,2,3 | 446.3 | — | 152.3 | — | 12.20 | — | 37.7 s |
| 2 | 1,3 | 349.1 | 125% | 150.5 | 108% | 12.45 | 125% | 37.0 s |
| 1 | — | — | | — | | — | | — |

N=1 is impossible: the model (13,115 MiB) does not fit an 11GB card. The derived `tg_1` is therefore a
**proxy value** (section 3).

### 2-3. Layer split, MTP on

| N | Devices | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0-3 | 360.4 | 129% | 141.0 | 101% | 9.97 | 100% | 29.1% | 45.3 s |
| 3 | 1,2,3 | 337.5 | 121% | 141.5 | 101% | 10.07 | 101% | 29.1% | 44.9 s |
| 2 | 1,3 | 279.5 | 100% | 139.7 | 100% | 11.05 | 111% | 29.1% | 41.5 s |

### 2-4. Tensor parallelism, MTP off

| N | Devices | Cold pp | % | Warm pp | % | **tg** | **%** | Warm turn |
|---|---|---|---|---|---|---|---|---|
| 4 | 0-3 | 460.6 | 165% | **343.4** | 246% | **28.94** | 290% | **16.0 s** |
| 3 | 1,2,3 | 398.8 | 143% | 286.2 | 205% | 25.59 | 257% | 18.2 s |
| 2 | 1,3 | 337.0 | 121% | 250.9 | 180% | 20.83 | 209% | 22.2 s |
| 1 | — | — | | — | | — | | — |

### 2-5. Tensor parallelism, MTP on

| N | Devices | Cold pp | % | Warm pp | % | **tg** | **%** | MTP acceptance | Warm turn |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0-3 | 430.4 | 154% | 286.3 | 205% | 26.92 | 270% | 30.5% | 17.5 s |
| 3 | 1,2,3 | 377.8 | 135% | 249.8 | 179% | 22.09 | 222% | 29.5% | 21.1 s |
| 2 | 1,3 | 320.0 | 114% | 222.2 | 159% | 17.30 | 174% | 26.3% | 26.5 s |

⚠️ Under tensor parallelism the generated text itself changes with N (METHOD 6-2), so **do not read the
acceptance differences as an effect of the GPU count.**

### 2-6. Temperature and throttling

From the METHOD 1-8 telemetry, **only the devices each cell used**. therm = `0x20|0x40`, pcap = `0x4` (SW power cap).

| Cell | Max | Mean | therm % | pcap % |
|---|---|---|---|---|
| tensor / off / 4 | 78°C | 67°C | 0.0 | 3.5 |
| tensor / on / 4 | 80°C | 70°C | 0.0 | 2.6 |
| tensor / off / 3 | 79°C | 70°C | 0.0 | 9.7 |
| tensor / on / 3 | 79°C | 73°C | 0.0 | 12.3 |
| tensor / off / 2 | 74°C | 70°C | 0.0 | **48.9** |
| tensor / on / 2 | 75°C | 72°C | 0.0 | **44.1** |
| layer / off / 2 | 71°C | 65°C | 0.0 | 24.8 |
| layer / on / 4 / 3 / 2 | 69 / 71 / 69°C | 58 / 59 / 64°C | 0.0 | 3.8 / 8.2 / 28.1 |
| ★ layer / off / 4 / 3 | 72 / 65°C | 58 / 55°C | 0.0 | 5.1 / 6.1 |

- **Thermal throttling is effectively absent** (0.2% once, in the auxiliary tensor / on / 3 run). This is the
  fan-curve daemon; cooling differs from the P104 runs
- **The power-cap flag is set 44~49% of the time in two-card cells** — with fewer cards each card does more work
  and reaches 250W during prefill. During decode the cards draw 80~90W, far from the cap

### Validation checklist (METHOD section 4)

| Item | Result |
|---|---|
| No `NCCL not compiled in` | ✅ every cell (main, auxiliary and re-run, 21 runs) |
| No `internal AllReduce init failed` | ✅ |
| `prompt_n` matches the target | ✅ 23,037 / 729 / 637 / 771 / 790 (`--check` max 25,436 + 400 = 25,836 / 27,000) |
| Cold `cache_n` = 0 | ✅ |
| Warm `cache_n` ≥ 22,000 | ✅ |
| `predicted_n` = 400 | ✅ |
| Warm `tg` spread within 5% (**MTP off cells only**) | ❌ **layer / off / 4 (6.7%, re-run 7.2%), layer / off / 3 (12.6%)** — invalid. Others 0.4~0.8% |
| Thermal and power throttle aggregated | ✅ 2-6 |
| No kernel Xid during the run | ✅ `dmesg` 0 Xid lines |

---

## 3. Interpretation

Figures and calculations are in [`gtx1080ti-x4-metrics.md`](gtx1080ti-x4-metrics.md).
Conclusions that span platforms are in [`../FINDINGS.md`](../FINDINGS.md).

### 3-1. Tensor parallelism wins on both prefill and decode — confirmed

Tensor over layer at four cards: warm pp 343.4 / 149.6 = **2.30x**, tg 28.94 / (best valid layer 12.45) = **2.32x**.
**There is no session break-even** — tensor four cards also has the shorter cold-turn wall time (63.8 s against
98.0 s for layer two cards), so it leads from the first turn. (On cold **pp** alone layer four cards, 551.6, is faster
than tensor four cards, 460.6, but the cold turn also includes 400 generated tokens.)

### 3-2. Layer split buys no decode speed — confirmed

Only the two-card cell is valid (12.45), but the invalid cells average 12.20 (three cards) and 11.92 (four), so
**decode does not get faster with more cards.** Warm pp is flat at 149.6~152.3. Same as P104 and V100.

### 3-3. Tensor prefill rises with the card count — the opposite of V100 on PCIe

| Tensor / off cold pp | 2 cards | 3 cards | 4 cards |
|---|---|---|---|
| P104, Gen1 x4 | 160.2 | 151.5 | 171.4 |
| V100, PCIe Gen3 x16 | 1,140.5 | 961.3 | 887.4 |
| V100, NVLink | 1,188.2 | 1,458.4 | 1,968.2 |
| **1080 Ti, Gen3 x8 (four cards include the x4)** | **337.0** | **398.8** | **460.6** |

Adding cards reduces each card's compute and adds all-reduce traffic. The 1080 Ti **computes far more slowly than
the V100 and has a far wider link than the P104**, so the compute reduction wins. That V100 goes the other way on the
same Gen3 PCIe is, we think, because its faster cards make communication a larger share (hypothesis; not separated
by measurement).

### 3-4. Tensor scaling efficiency

`tg_1` = **12.45 (layer / off / 2 used as a proxy)**. Layer three and four cards are invalid, so the single valid
two-card cell was used instead of the layer average METHOD section 5 prescribes.

| N | Multiple | Efficiency |
|---|---|---|
| 2 | 1.67x | 83.6% |
| 3 | 2.05x | 68.5% |
| 4 | 2.32x | 58.1% |

Four cards include the x4 device (the auxiliary runs suggest the x4 device costs three cards 5% of tg — section 8).
The four-card efficiency is almost the same as P104 at Gen1 x4 (57%). That a much wider link leaves efficiency
unchanged is consistent with **FINDINGS 5: what governs scaling is not the size of the PCIe bandwidth.**

### 3-5. Decode is not bandwidth bound — confirmed

| Cell | Calculation | Share of specification (484 GB/s) |
|---|---|---|
| tensor 4 cards | 13,115 MiB / 4 × 28.94 = 99.5 GB/s | **21%** |
| layer 2 cards (each card active half the time) | 6.88 GB × 12.45 × 2 = 171 GB/s | 35% |

### 3-6. Against P104 (four cards)

**The Gen1 x1 run shares the host, driver and binary**, so against it only the cards (and the link) changed. The Gen1 x4
run also differs in host (EPYC 7232 / 32GB), driver and build, so it is not a single-variable comparison.

| Four cards | P104 Gen1 x1 (same host) | P104 Gen1 x4 (other host) | 1080 Ti | Multiple (vs x1 / vs x4) |
|---|---|---|---|---|
| Layer / off cold · warm · tg | 302.6 / 68.6 / 7.40 | 327.3 / 85.3 / 7.82 | 551.6 / 149.6 / 11.92★ | 1.82 / 2.18 / 1.61 · 1.69 / 1.75 / 1.52 |
| Tensor / off cold · warm · tg | 60.7 / 47.2 / 11.16 | 171.4 / 128.8 / 17.89 | 460.6 / 343.4 / 28.94 | **7.59 / 7.28** / 2.59 · **2.69 / 2.67** / 1.62 |

Prefill grows most on the tensor side, where the link widened; decode grows by about the card compute ratio
(about 1.8x) or less. Most of the 7x tensor multiple against Gen1 x1 is the link.

---

## 4. MTP variance and payoff

| Cell | Warm four-turn tg | Spread |
|---|---|---|
| tensor / off / 4 | mean 28.94 | 0.4% |
| tensor / on / 4 | mean 26.92 | **10.9%** |
| tensor / on / 3 | mean 22.09 | 7.4% |
| tensor / on / 2 | mean 17.30 | 8.1% |
| layer / on / 4 / 3 / 2 | 9.97 / 10.07 / 11.05 | 8.2 / 7.7 / 7.3% |

**MTP loses in every cell.**

| Cell | Cost multiple | Output per step | Payoff | Break-even acceptance | Measured acceptance |
|---|---|---|---|---|---|
| layer / 4 | 2.230 | 1.865 | −16.4% | 41.5% | 29.1% |
| layer / 3 | 2.256 | 1.865 | −17.3% | 42.3% | 29.1% |
| layer / 2 | 2.103 | 1.865 | −11.3% | 37.2% | 29.1% |
| tensor / 4 | 2.051 | 1.905 | −7.1% | 35.4% | 30.5% |
| tensor / 3 | 2.177 | 1.878 | −13.7% | 39.6% | 29.5% |
| tensor / 2 | 2.148 | 1.782 | −17.1% | 38.7% | 26.3% |

The cost multiples of 2.05~2.26 sit between P104 (Gen1 x4: 2.25 / 2.42) and V100 (1.82~1.90).
⚠️ `ignore_eos` makes acceptance lower than in real use (METHOD 1-5).

---

## 5. Failures and anomalies

### Invalid: layer / MTP off / three and four cards — warm tg spread above 5%

| Run | Warm four-turn tg | Spread |
|---|---|---|
| layer / off / 4 (main) | 11.58 / 11.70 / 12.04 / 12.37 | 6.7% |
| layer / off / 4 (same cell re-run) | 11.48 / 11.62 / 11.96 / 12.30 | 7.2% |
| layer / off / 3 (main, 1,2,3) | 12.52 / 12.60 / 12.49 / 11.19 | **12.6%** |
| layer / off / 3 (auxiliary, 0,1,2) | 12.74 / 12.55 / 12.58 / 12.61 | 1.5% — passes |
| layer / off / 2 (main and auxiliary) | — | 0.6% / 0.8% — pass |

- No crash, NCCL warning, Xid or thermal throttling. The condition checks (`prompt_n`, `cache_n`, `predicted_n`) pass
- **Suspected cause (not established):** under layer split only one card computes at a time during decode. The telemetry
  shows the idle cards' SM clocks moving between 1,480 MHz and the 1,800 MHz range. More cards mean longer idle periods
  and, we think, more wobble. That the same cell passed (1.5%) on one device set and failed (12.6%) on another suggests
  **a run-to-run effect, not the device combination**
- **We did not dig further.** Locking clocks (`nvidia-smi -lgc`) would test it, but that is not an allowed variation
  in METHOD and would make it a separate measurement
- **The practical consequence is small** — on this platform layer split is 2.3x slower than tensor anyway

---

## 7. Limits specific to this platform

- **The single-card baseline is a proxy** (layer / off / 2 = 12.45). With layer three and four cards invalid, one cell
  was used instead of the average METHOD section 5 prescribes
- **No N=1 cell** — the model does not fit in 11GB
- **Measured under METHOD v1.5** (device-selection clause). The measurement conditions equal v1.0~v1.4, so the results
  can share a table with earlier runs
- **Only the four-card cells include the x4 link** — card-count comparisons (two and three against four) mix in a link
  difference. The auxiliary runs put its size at 6~7% of tensor warm pp and 1~5% of tg
- **Cooling differs from the P104 runs** (fan-curve daemon). With no thermal throttling here, the multiples against
  P104 may include a cooling difference
- Host, driver and binary are **the same as the P104 Gen1 x1 run** — but the link changed with the cards (Gen1 x1 →
  Gen3 x4/x8), so card and link effects are mixed

---

## 8. Raw data and cell validity

| | Contents |
|---|---|
| [`../results/gtx1080ti-x4.jsonl`](../results/gtx1080ti-x4.jsonl) | One JSON line per run, 21 rows: main runs 12 (`ok` 10, `invalid` 2), the layer / off / 4 re-run (`invalid`), auxiliary runs 8 (`side` — two cards on 0,1 / three cards on 0,1,2). Each row carries `devices` and a `note` with the PCIe links |

### Cell validity

| Class | Cells | Basis |
|---|---|---|
| **Valid** | 10 main cells | Every checklist item passes |
| **Invalid** | layer / off / 4, layer / off / 3 (main) | Warm tg spread 6.7% (re-run 7.2%), 12.6% |
| **Auxiliary** | two cards on 0,1 (4 cells), three cards on 0,1,2 (4 cells) | Only the device set differs. All pass the checklist |

**Invalid cells and the re-run are kept, not discarded.**

### Auxiliary runs (first N devices, including the x4)

| Cell | Devices | Cold pp | Warm pp | tg | tg against main run |
|---|---|---|---|---|---|
| tensor / off / 2 | 0,1 | 313.7 | 235.8 | 20.62 | −1.0% |
| tensor / on / 2 | 0,1 | 298.8 | 208.8 | 17.21 | −0.5% |
| layer / off / 2 | 0,1 | 355.1 | 150.0 | 12.97 | +4.2% |
| layer / on / 2 | 0,1 | 283.8 | 140.7 | 11.17 | +1.1% |
| tensor / off / 3 | 0,1,2 | 368.3 | 267.1 | 24.25 | −5.2% |
| tensor / on / 3 | 0,1,2 | 347.8 | 232.4 | 21.99 | −0.5% |
| layer / off / 3 | 0,1,2 | 433.7 | 151.3 | 12.62 | (main run invalid) |
| layer / on / 3 | 0,1,2 | 335.6 | 140.9 | 10.20 | +1.3% |
