# layer-tensor-parallel-bench

**llama.cpp layer split vs tensor parallelism, measured across GPUs and interconnects.**

llama.cpp's [multi-GPU guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)
describes what each split mode is for but publishes **no measured numbers.** This repository
measures them with **one frozen methodology** on three kinds of GPU, four cards each. The main
axes of comparison are the GPU and the PCIe bandwidth; some runs also compare NVLink.

The workload follows the author's real usage — one cold prefill of ~23k tokens, then warm
turns hitting the prompt cache. **Split modes rank differently on cold prefill and on warm
turns**, so both are measured.

**Summary of results**

1. **Layer split buys no decode speed.** Two cards with tensor parallelism decode faster
   than four with layer split, and **on the same card** a wider link (Gen1 x1 → x4,
   PCIe → NVLink) widens the gap.
2. **Speculative decoding (MTP) flips sign with the card and the split mode.** At similar
   acceptance it loses 16% on P104 and gains 3% on V100; switching the same card to tensor
   makes it a loss again. On 1080 Ti it loses in every cell (−7~−17%).
3. **Interconnect bandwidth and tensor-parallel scaling.** V100 on a 16x wider PCIe link
   scaled worse than P104 (four-card efficiency 57% → 42%), cutting P104's link from x4 to
   x1 also scaled worse, and NVLink scaled better.
4. **Split modes respond oppositely to the link.** At a quarter of the bandwidth (P104,
   Gen1 x4 → x1) decode drops 4~6% under layer split and 24~48% under tensor.
5. **More cards can slow tensor prefill or speed it up.** From two to four cards, tensor
   cold prefill falls on PCIe V100 (1,141 → 887 t/s) and rises on 1080 Ti (337 → 461) and
   on NVLink.

*(Supplementary)* On the same system, the MTP payoff reversed with the model —
[Supplementary finding S1](FINDINGS.md#supplementary-findings--outside-the-frozen-method)

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
