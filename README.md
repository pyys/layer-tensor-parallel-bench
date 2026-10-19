# layer-tensor-parallel-bench

**llama.cpp layer split vs tensor parallelism, measured across GPUs and interconnects.**

llama.cpp's [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
describes what each split mode is for but publishes no measured numbers. This repository
records numbers from one frozen methodology, applied to two kinds of GPU, four cards
each, on two interconnects each — P104-100 x4 at PCIe Gen1 x4 and at Gen1 x1, the narrowest
link PCIe defines, and V100 16GB x4 at PCIe Gen3 x16 and on NVLink (DGX-1).

The workload follows the author's real usage — one cold prefill of ~23k
tokens, then warm turns hitting the prompt cache. **Split modes rank differently on cold
prefill and on warm turns**, so both are measured.

**Summary of results**

- **Layer split buys no decode speed.** Measured single-card baseline: 27.71 t/s on
  one card, 27.29 on four.
- **Two cards with tensor parallelism beat four with layer split** — from the third
  message on P104, the first on V100, the 34th at Gen1 x1, and from the cold prefill on
  NVLink.
- **Speculative decoding flips sign with the card and the split mode** — at similar
  acceptance it loses 16% on P104, gains 3% on V100, and loses again under tensor.
- **Interconnect and tensor-parallel scaling.** A 16x faster PCIe link did not scale
  better, cutting P104 to x1 scaled worse (two to four cards: 1.37x to 1.16x), and moving
  the same V100 card to NVLink scaled better (four-card efficiency 42% to 52%).
- **Split modes respond oppositely to the link.** At a quarter of the bandwidth layer
  split loses 4~6% of decode and tensor 24~48%, yet tensor still decodes 1.51x faster.
- *(Supplementary)* Gemma 4 31B on the same NVLink system left tensor scaling unchanged
  but **reversed MTP** (Qwen against Gemma: four-card layer +7.6% / −41.2%, tensor
  −4.5% / +26.9%). [S1](FINDINGS.md#supplementary-findings--outside-the-frozen-method)

English is authoritative; Korean lives alongside each file as `*.ko.md`.
**이 문서의 한국어판: [README.ko.md](README.ko.md)**
Code and raw data: MIT. Documents: CC BY 4.0.

| | |
|---|---|
| [FINDINGS.md](FINDINGS.md) | What we found, and what we did not |
| [METHOD.md](METHOD.md) | Frozen measurement contract, reproduction, pitfalls |
| [docs/](docs/) | Per-platform reports and derived metrics |
| [supplements/](supplements/README.md) | Supplementary runs that change a fixed item of the method (kept apart from the main findings) |
| [results/](results/) | Raw `timings` output, one JSON line per cell |

**Start with [FINDINGS.md](FINDINGS.md).** It carries the conclusions, the numbers
behind them and the limits on each one. Everything else in the repository exists to
support it — METHOD is the contract the numbers were produced under, `docs/` holds one
report per platform, and `results/` is the data itself.
