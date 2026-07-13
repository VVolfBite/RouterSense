# M-HOTPATH-GPU1 Report

## Goal
Validate that full 16-layer 4GPU forward only pays RouterSense heavy hook cost on selected layers 0 and 1, then collect small-scale cost attribution for native, B-core async, and U-zero async.

## Environment
- GPUs: 4 x NVIDIA GeForce RTX 4090 D
- Torch/CUDA: 2.8.0+cu128 / 12.8
- Commit at run start: 9215363a80b7894d84d30a1014f1238562e5ac29
- Config: configs/official/gpu_hotpath_iteration.yaml

## Workload
- Prompts: 8
- Tokenized shape: 8 x 16
- Padded tokens: 56
- Valid non-padding tokens: 72

## Count Smoke
- B-core selected P0/P1 all-rank: 16 / 16
- U-zero selected P0/P1 all-rank: 16 / 16
- Prediction-source P0 all-rank: 0
- None-heavy all-rank: 0
- U-zero raw-U build all-rank: 16, per selected layer per rank <= 2
- Preflight requested/effective: compact / compact

## Perf Smoke Medians
- native total forward: 139060.546875 us
- B-core total forward: 223674.30114746094 us
- U-zero total forward: 211923.03466796875 us

## Cost Attribution
- Native communication timing is unavailable, not zero.
- B-core communication span median: 87619.009 us; active transport median: 7992.548999999999 us; control median: 9099.378000000002 us.
- U-zero communication span median: 89184.663 us; active transport median: 5815.274 us; control median: 6062.950000000001 us.
- First hotspot for both RouterSense strategies is communication span / rank-level transport span, not raw-U build.

## Not Run
No full pytest, full first bring-up, full C2, predicted raw/safe, seven-strategy A2, target lifecycle matrix, Nsight, or PyTorch profiler trace.

## Conclusion
HOTPATH_SCOPE_GPU_VALIDATED. This is scope and attribution validation only; it is not a paper performance claim.
