# Benchmark: timm's ResNet18 on torch-mlx vs. real PyTorch (CPU/MPS)

This fork adds two files, `benchmark_torch_mlx.py` and `tv_stub.py`, and changes
nothing else in the original project. It runs this repo's own, completely unmodified
`timm.create_model('resnet18', ...)` against three backends:

- Real PyTorch, CPU
- Real PyTorch, MPS (Apple Silicon GPU via Metal)
- [torch-mlx](https://github.com/bahaehmimdi/torch-mlx) — a from-scratch PyTorch-API-
  compatible layer backed by Apple's `mlx.core` array framework instead of PyTorch's own
  ATen/Metal backend. No source changes to this repo's own model code are needed, only
  which `torch` is on `sys.path`.

`tv_stub.py` replaces `torchvision` (an optional timm dependency, imported for its FX
feature-extraction helper and a couple of pure-python fallback classes like
`FrozenBatchNorm2d`) with a lightweight stand-in — torchvision's own native C++
extension genuinely cannot run under torch-mlx (the same class of issue affecting every
other repo in this project that depends on it), and none of the stubbed functionality
(feature extraction, dataset loaders, PIL image transforms) is needed just to construct
and run a model on tensors already in memory.

Getting timm importable and runnable under torch-mlx also required fixing **6 real,
general gaps** in torch-mlx itself (see
[the commit](https://github.com/bahaehmimdi/torch-mlx/commit/6f93bf3)): `torch.fx.wrap`
(an FX-tracing leaf-function decorator that is a genuine no-op at eager runtime),
`torch.jit.Final`/`torch.jit.annotations` (both just `typing` re-exports in real
PyTorch), `nn.init._calculate_fan_in_and_fan_out`, `torch.utils.data.Sampler` needing
to be `Generic` (for `Sampler[SomeType]` subscripting), and `nn.Conv2d`/`_BatchNorm`/
`Linear` missing the `device=`/`dtype=` "factory kwargs" present on every real `nn`
layer constructor.

## Results

Apple M1 Pro, ResNet18 (`num_classes=10`), 64x64 input, batch 4, forward + backward +
one SGD step, 10-step average after 3 warmup steps:

| Backend | Time/step |
|---|---:|
| PyTorch CPU | 28.67ms |
| PyTorch MPS | 17.56ms |
| torch-mlx (eager) | 19.28ms |
| **torch-mlx (compiled)** | **12.60ms** |

torch-mlx compiled is **~1.4x faster than MPS** and **~2.3x faster than CPU** — a
smaller margin than this project's other benchmarks, matching the pattern already
recorded for ResNet-family conv nets elsewhere in torch-mlx (see
[FIELD_GUIDE.md](https://github.com/bahaehmimdi/torch-mlx/blob/master/FIELD_GUIDE.md)):
plain conv+BatchNorm stacks are close to MPS's own well-tuned kernels, so the win is
real but modest rather than dramatic.

## Running it yourself

```bash
git clone --recursive https://github.com/bahaehmimdi/torch-mlx
pip install "torch==2.5.1" mlx numpy pillow pyyaml huggingface_hub safetensors

python benchmark_torch_mlx.py --backend cpu
python benchmark_torch_mlx.py --backend mps
python benchmark_torch_mlx.py --backend torch-mlx --torch-mlx-path /path/to/torch-mlx
python benchmark_torch_mlx.py --backend torch-mlx --torch-mlx-path /path/to/torch-mlx --compile

# any other timm model works the same way, e.g.:
python benchmark_torch_mlx.py --backend torch-mlx --torch-mlx-path /path/to/torch-mlx --compile --model resnet50
```
