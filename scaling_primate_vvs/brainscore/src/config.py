import json

from brainscore_vision import load_benchmark

try:
    with open('../results/commitments.json', 'r') as f:
        MODEL_COMMITMENTS = json.load(f)
except FileNotFoundError:
    try:
        with open('results/commitments.json', 'r') as f:
            MODEL_COMMITMENTS = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Could not find `commitments.json` in `results/` directory")
except Exception as e:
    MODEL_COMMITMENTS = {}
    print(f"Error loading `commitments.json`: {e}")

BENCHMARKS_IDENTIFIERS = {
    "V1": "FreemanZiemba2013public.V1-pls",
    "V2": "FreemanZiemba2013public.V2-pls",
    "V4": "MajajHong2015public.V4-pls",
    "IT": "MajajHong2015public.IT-pls",
    "Behaviour": "Rajalingham2018public-i2n",
}

BENCHMARKS = {region: load_benchmark(benchmark) for region, benchmark in BENCHMARKS_IDENTIFIERS.items()}

STANDARD_REGION_BENCHMARKS = {
    'V1': BENCHMARKS['V1'],
    'V2': BENCHMARKS['V2'],
    'V4': BENCHMARKS['V4'],
    'IT': BENCHMARKS['IT'],
}

MODEL_IDENTIFIERS = {
    "cornet_s": "CORnet_S",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "resnet101": "ResNet101",
    "resnet152": "ResNet152",
    "efficientnet_b0": "EfficientNet_B0",
    "efficientnet_b1": "EfficientNet_B1",
    "efficientnet_b2": "EfficientNet_B2",
    "efficientnet_b3": "EfficientNet_B3",
    "efficientnet_b4": "EfficientNet_B4",
    "convnext_tiny": "ConvNeXt_Tiny",
    "convnext_small": "ConvNeXt_Small",
    "convnext_base": "ConvNeXt_Base",
    "convnext_large": "ConvNeXt_Large",
    "deit_tiny": "DeiT_Tiny",
    "deit_small": "DeiT_Small",
    "deit_base": "DeiT_Base",
    "deit_large": "DeiT_Large",
    "deit3_tiny_patch16_224": "DeiT_Tiny",
    "deit3_small_patch16_224": "DeiT_Small",
    "deit3_base_patch16_224": "DeiT_Base",
    "deit3_large_patch16_224": "DeiT_Large",
    "vit_b_16": "ViT_B_16",
    "vit_l_16": "ViT_L_16",
    "alexnet": "AlexNet",
}

ABLATION = [
    "resnet18-001",
    "resnet18-05",
    "resnet18-a3",
    "resnet18-adam-00001",
    "resnet18-adam-0001",
    "resnet18-adam-001",
    "resnet18-adam-01",
    "resnet18-adamw-00001",
    "resnet18-adamw-0001",
    "resnet18-adamw-001",
    "resnet18-adamw-01",
    "resnet18-constlr-001",
    "resnet18-constlr-01",
    "resnet18-constlr-05",
    "resnet18-heavy-aug",
    "resnet18-minlr-001",
    "resnet18-nowarmup",
    "resnet18-steplr-10",
    "resnet18-steplr-20",
    "resnet18-steplr-30",
    "resnet18-warmup10",
    "resnet18-warmup2",
    "resnet152-heavy-aug",
    "resnet152-heavy-aug-200ep",
    "resnet18-001-200ep",
    "resnet18-01-200ep",
    "resnet18-heavy-aug-200ep",
    # "resnet18_imagenet_full_seed-1",
    # "resnet18_imagenet_full_seed-2",
    # "resnet18-warmup10_imagenet_full_seed-1",
    "resnet18-warmup20"
]