import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["cpu", "mps", "torch-mlx"], required=True)
ap.add_argument("--torch-mlx-path", default="")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--model", default="resnet18")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--num-classes", type=int, default=10)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--warmup", type=int, default=3)
args = ap.parse_args()

if args.backend == "torch-mlx" and args.torch_mlx_path:
    sys.path.insert(0, args.torch_mlx_path)

# torchvision (an optional timm dependency, used only for its FX
# feature-extraction helper and a couple of pure-python fallback
# classes) is stubbed rather than installed: its native C++ extension
# genuinely cannot run under torch-mlx, the same class of issue as
# every other repo in this project that depends on it. See tv_stub.py.
import tv_stub
tv_stub.install()

# Import this repo's own, unmodified `timm` directly.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import torch
import timm

torch.manual_seed(0)
BATCH, SIZE = args.batch_size, args.size
LR = args.lr

model = timm.create_model(args.model, pretrained=False, num_classes=args.num_classes)
x = torch.randn(BATCH, 3, SIZE, SIZE)


def loss_fn(out):
    return out.pow(2).mean()


if args.backend == "torch-mlx" and args.compile:
    import mlx.core as mx
    from torch._tensor import Tensor
    from torch.func import functional_call

    raw_params = {n: p.data for n, p in model.named_parameters()}
    raw_x = x.data

    def raw_step(params, raw_x):
        p = {k: Tensor(v, requires_grad=True) for k, v in params.items()}
        out = functional_call(model, p, (Tensor(raw_x),))
        loss = loss_fn(out)
        loss.backward()
        new_params = {}
        for k, t_ in p.items():
            g = t_.grad
            new_params[k] = (t_.data - LR * g.data) if g is not None else t_.data
        return new_params, loss.data

    step_fn = mx.compile(raw_step)
    for _ in range(args.warmup):
        raw_params, loss_val = step_fn(raw_params, raw_x)
        mx.eval(*raw_params.values(), loss_val)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        raw_params, loss_val = step_fn(raw_params, raw_x)
        mx.eval(*raw_params.values(), loss_val)
    dt = (time.perf_counter() - t0) * 1000 / args.steps
    print(f"torch-mlx (compiled): loss={float(loss_val):.6f} time={dt:.2f}ms/step")
else:
    device = args.backend if args.backend != "torch-mlx" else None
    if device is not None:
        model = model.to(device)
        x = x.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif args.backend == "torch-mlx":
            import mlx.core as mx
            mx.eval(*[p.data for p in model.parameters()])

    for _ in range(args.warmup):
        out = model(x)
        loss = loss_fn(out)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        out = model(x)
        loss = loss_fn(out)
        loss.backward()
        opt.step()
        opt.zero_grad()
        sync()
    dt = (time.perf_counter() - t0) * 1000 / args.steps
    label = args.backend if args.backend != "torch-mlx" else "torch-mlx (eager)"
    print(f"{label}: loss={float(loss):.6f} time={dt:.2f}ms/step")
