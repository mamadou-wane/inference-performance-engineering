# Qwen3 fixtures

Files from immutable revisions of the official Qwen repositories. None contains tensor data or tokenizer files.

| File | Source | Kind | SHA-256 |
| --- | --- | --- | --- |
| qwen3-1.7b-config.json | [Qwen/Qwen3-1.7B, 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/config.json), `config.json` | Unmodified copy | 1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197 |
| qwen3-0.6b-config.json | [Qwen/Qwen3-0.6B, c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json), `config.json` | Unmodified copy | 660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd |
| qwen3-1.7b-model.safetensors.index.json | Qwen/Qwen3-1.7B, 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, `model.safetensors.index.json` | Unmodified copy | 0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d |
| qwen3-1.7b-model-00001-of-00002.header | Qwen/Qwen3-1.7B, 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, `model-00001-of-00002.safetensors` | Modified: truncated to bytes 0 to 35,655 | 1ee0e4e7bf2b79af3ee159ee7894dc37663ca8118eeb88ddd9a3ff2c1c4d91e7 |
| qwen3-1.7b-model-00002-of-00002.header | Qwen/Qwen3-1.7B, 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, `model-00002-of-00002.safetensors` | Modified: truncated to bytes 0 to 127 | de3c1e531e8d90b962f43c40af894e503d94d171899631d258620241fff6c898 |
| qwen3-0.6b-model.header | Qwen/Qwen3-0.6B, c1899de289a04d12100db370d81485cdf75e47ca, `model.safetensors` | Modified: truncated to bytes 0 to 35,559 | 1119c4d33e17f4cd2d3fbebbc4176d40555e61361cb44076319167621f4034d4 |

## Structural fixtures

The configuration and index files are unmodified copies. Each `.header` file is modified: truncated to the 8-byte header length and the JSON header of the named file, with all tensor data removed. The extracts were cut from downloads that matched the Hub's published SHA-256 for the full files; the Hub publishes no hash for an extract, so the SHA-256 above is the reference.

Tests rebuild each extract as a sparse file of the original size, with zeros where the tensor data was. The extracts and the index check structure only: names, dtypes, shapes, offsets, layout, the index's agreement with the shards, and the configuration mapping. Tensor values need the downloaded checkpoints.

The configuration fixtures are checked against [Transformers 5.17.0 at commit 856157a2f3e9594954310df18fdccc31ffddebe9](https://github.com/huggingface/transformers/blob/856157a2f3e9594954310df18fdccc31ffddebe9/src/transformers/models/qwen3/configuration_qwen3.py), with literal reference values for expected dimensions. The 0.6B configuration has a hidden size of 1024 with 16 query heads of dimension 128, so attention projection width differs from hidden size.

## License

These files are distributed under the Apache License, Version 2.0, the license of the source repositories. [LICENSE-Qwen3-Apache-2.0](LICENSE-Qwen3-Apache-2.0) is an unmodified copy of the `LICENSE` file at both revisions (SHA-256 832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e), including its notice "Copyright 2024 Alibaba Cloud". Neither repository has a NOTICE file. The rest of this repository is MIT-licensed; see the top-level LICENSE.

Attribution: Qwen3-1.7B and Qwen3-0.6B by the Qwen team, Alibaba Cloud. License files at the source revisions: [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE).
