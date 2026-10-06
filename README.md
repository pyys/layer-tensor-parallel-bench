# layer-tensor-parallel-bench

**llama.cpp layer split vs tensor parallelism, measured across GPUs and interconnects.**

llama.cpp's [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
describes what each split mode is for but publishes no measured numbers. This repository
records numbers from one frozen methodology, applied to two kinds of GPU with four cards
each — P104-100 x4 (PCIe Gen1 x4) and V100 16GB x4 (PCIe Gen3 x16) — and to the same
P104 card model again at PCIe Gen1 x1, the narrowest link PCIe defines.

The workload follows the pattern the author actually runs — one cold prefill of ~23k
tokens, then warm turns hitting the prompt cache. **Split modes rank differently on cold
prefill and on warm turns**, so both have to be measured.

**Summary of results**

- **Layer split buys no decode speed.** Now confirmed against a measured single-card
  baseline: 27.71 t/s on one card, 27.29 on four.
- **Two cards with tensor parallelism beat four with layer split.** On P104 that
  crossover paid for itself by the third message; on V100 it pays from the first; at
  Gen1 x1 it takes 34 messages.
- **Speculative decoding flips sign with the card and the split mode.** At nearly
  identical acceptance rates it loses 16% on P104 and gains 3% on V100; on the same
  card, switching layer to tensor turns the gain back into a loss.
- **Interconnect bandwidth and tensor-parallel scaling.** A 16x faster link (V100)
  did not scale better, but cutting the same P104 card's link from x4 to x1 did scale
  worse — two to four cards went from 1.37x to 1.16x.
- **Split modes respond oppositely to the link.** At a quarter of the bandwidth, layer
  split loses 4~6% of decode and tensor parallelism 24~48% — yet tensor parallelism
  still decodes 1.51x faster.

English is authoritative; Korean lives alongside each file as `*.ko.md`.
**이 문서의 한국어판: [README.ko.md](README.ko.md)**
Code and raw data: MIT. Documents: CC BY 4.0.

| | |
|---|---|
| [FINDINGS.md](FINDINGS.md) | What we found, and what we did not |
| [METHOD.md](METHOD.md) | Frozen measurement contract, reproduction, pitfalls |
| [docs/](docs/) | Per-platform reports and derived metrics |
| [results/](results/) | Raw `timings` output, one JSON line per cell |

**Start with [FINDINGS.md](FINDINGS.md).** It carries the conclusions, the numbers
behind them and the limits on each one. Everything else in the repository exists to
support it — METHOD is the contract the numbers were produced under, `docs/` holds one
report per platform, and `results/` is the data itself.
