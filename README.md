# layer-tensor-parallel-bench

**llama.cpp layer split vs tensor parallelism, measured across GPUs and interconnects.**

llama.cpp's [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
describes what each split mode is for but publishes no measured numbers. This repository
records numbers from one frozen methodology, applied to two kinds of GPU with four cards
each: P104-100 x4 (PCIe Gen1 x4) and V100 16GB x4 (PCIe Gen3 x16).

The workload follows the pattern the author actually runs — one cold prefill of ~23k
tokens, then warm turns hitting the prompt cache. **Split modes rank differently on cold
prefill and on warm turns**, so both have to be measured.

**Summary of results**

- **Layer split buys no decode speed.** Now confirmed against a measured single-card
  baseline: 27.71 t/s on one card, 27.29 on four.
- **Two cards with tensor parallelism beat four with layer split.** On P104 that
  crossover paid for itself by the third message; on V100 it pays from the first.
- **Speculative decoding flips sign with the card and the split mode.** At nearly
  identical acceptance rates it loses 16% on P104 and gains 3% on V100; on the same
  card, switching layer to tensor turns the gain back into a loss.
- **Interconnect bandwidth is not what limits tensor-parallel scaling.** Four cards
  return 2.29x the single-card rate on P104 but only 1.68x on V100 — the platform with
  the 16x faster link scales worse.

English is authoritative; Korean lives alongside each file as `*.ko.md`.
Code and raw data: MIT. Documents: CC BY 4.0.

| | |
|---|---|
| [FINDINGS.md](FINDINGS.md) | What we found, and what we did not |
| [METHOD.md](METHOD.md) | Frozen measurement contract, reproduction, pitfalls |
| [docs/](docs/) | Per-platform reports and derived metrics |
| [results/](results/) | Raw `timings` output, one JSON line per cell |
