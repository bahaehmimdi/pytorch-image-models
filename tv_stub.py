import sys, types
import torch


def install():
    fake_tv = types.ModuleType("torchvision")
    fake_tv.models = types.ModuleType("torchvision.models")
    fake_tv.models.feature_extraction = types.ModuleType("torchvision.models.feature_extraction")
    fake_tv.models.feature_extraction.create_feature_extractor = lambda *a, **k: None
    fake_tv.models.feature_extraction.get_graph_node_names = lambda *a, **k: ([], [])

    fake_tv.ops = types.ModuleType("torchvision.ops")
    fake_tv.ops.misc = types.ModuleType("torchvision.ops.misc")

    class FrozenBatchNorm2d(torch.nn.Module):
        def __init__(self, num_features, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.register_buffer("weight", torch.ones(num_features))
            self.register_buffer("bias", torch.zeros(num_features))
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))

        def forward(self, x):
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            scale = w * (rv + self.eps).rsqrt()
            bias = b - rm * scale
            return x * scale + bias

    fake_tv.ops.misc.FrozenBatchNorm2d = FrozenBatchNorm2d

    fake_tv.datasets = types.ModuleType("torchvision.datasets")
    for name in ("CIFAR100", "CIFAR10", "MNIST", "KMNIST", "FashionMNIST", "ImageFolder", "ImageNet", "QMNIST", "Places365", "INaturalist"):
        setattr(fake_tv.datasets, name, type(name, (), {}))

    fake_tv.transforms = types.ModuleType("torchvision.transforms")

    import enum

    class InterpolationMode(enum.Enum):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    fake_tv.transforms.InterpolationMode = InterpolationMode

    def _dummy_transform_class(name):
        return type(name, (), {
            "__init__": lambda self, *a, **k: None,
            "__call__": lambda self, x, *a, **k: x,
        })

    def _transforms_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        cls = _dummy_transform_class(name)
        setattr(fake_tv.transforms, name, cls)
        return cls

    fake_tv.transforms.__getattr__ = _transforms_getattr
    fake_tv.transforms.functional = types.ModuleType("torchvision.transforms.functional")
    fake_tv.transforms.functional.InterpolationMode = InterpolationMode
    for _name in ("pil_to_tensor", "to_tensor", "get_image_size", "resized_crop",
                  "get_dimensions", "pad", "crop", "resize"):
        setattr(fake_tv.transforms.functional, _name, lambda *a, **k: None)

    sys.modules["torchvision"] = fake_tv
    sys.modules["torchvision.models"] = fake_tv.models
    sys.modules["torchvision.models.feature_extraction"] = fake_tv.models.feature_extraction
    sys.modules["torchvision.ops"] = fake_tv.ops
    sys.modules["torchvision.ops.misc"] = fake_tv.ops.misc
    sys.modules["torchvision.datasets"] = fake_tv.datasets
    sys.modules["torchvision.transforms"] = fake_tv.transforms
    sys.modules["torchvision.transforms.functional"] = fake_tv.transforms.functional
