import argparse

def str2bool(v):
    """
    Converts a string representation of a boolean value to its corresponding boolean value.
    
    Args:
        v (str): The string representation of the boolean value.
        
    Returns:
        bool: The corresponding boolean value.
    """
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_argparser():
    parser = argparse.ArgumentParser(description="BrainScore Benchmarking")
    parser.add_argument(
        "--save-dir",
        default="./layer_selection",
        type=str,
        metavar="PATH",
        help="path to save output (default: ./layer_selection/)",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        type=str,
        metavar="PATH",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="cornet_s",
        # choices=["cornet_s", "resnet18"],
        help="model architecture: " + " (default: cornet_s)",
    )
    parser.add_argument(
        "--num-classes",
        dest="num_classes",
        type=int,
        default=1000,
        help="Number of classes of the model",
    )
    parser.add_argument(
        "--training-dataset",
        dest="training_dataset",
        type=str,
        choices=["imagenet", "ecoset", "unknown", "webvision", "imagenet21kP", "webvisionP", "laion", "imagenet21kP-class-1000", "imagenet21kP-class-5000", "places365", "inaturalist"],
        default="imagenet",
        help="Training dataset",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained torchvision weights",
    )
    parser.add_argument(
        "--pretrained-weights",
        dest="pretrained_weights",
        type=str,
        default="DEFAULT",
        help="which pre-trained torchvision weights to use",
    )
    parser.add_argument(
        '--load-model-ema',
        dest='load_model_ema',
        type=str2bool,
        default=False,
        help='Load model ema weight to model.ema (default: False)'
    )
    # parser.add_argument(
    #     '--use-model-ema',
    #     dest='use_model_ema',
    #     type=str2bool,
    #     default=False,
    #     help='Use model ema for benchmarking (default: False)'
    # )
    
    parser.add_argument(
        "--use-timm",
        dest="use_timm",
        action="store_true",
        help="use a timm model"   
    )
    parser.add_argument(
        "--use-open-clip",
        dest="use_open_clip",
        action="store_true",
        help="use an open clip model"   
    )
    parser.add_argument(
        "--layer-type",
        dest="layer_type",
        default="all",
        help="Layer type to select",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        help="Run name",
    )
    parser.add_argument(
        "--resize-size",
        dest="resize_size",
        default=256,
        type=int,
        metavar="N",
        help="resize size (default: 256)",
    )
    parser.add_argument(
        "--crop-size",
        dest="crop_size",
        default=224,
        type=int,
        metavar="N",
        help="crop size (default: 224)",
    )
    parser.add_argument(
        "--interpolation",
        default='bilinear',
        type=str,
        metavar="N",
        help="interpolation mode for resizing (default: bilinear)",
    )
    parser.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default='',
        help="Suffix for the output file name",
    )
    parser.add_argument(
        "--ckpt-src",
        dest="ckpt_src",
        type=str,
        default='full',
        choices=["full", "pretrained"],
        help="Whether use layer selected using full training or a pretrained model",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        metavar="N",
        help="random seed (default: 0)",
    )
    parser.add_argument(
        "--disable-pca",
        dest="disable_pca",
        action="store_true",
        help="disable PCA, which is used to reduce layer activations",
    )
    parser.add_argument(
        "--enable-random-projection",
        dest="enable_random_projection",
        action="store_true",
        help="enable random projections, which is used to reduce layer activations",
    )
    parser.add_argument(
        "--n-compression-components",
        dest="n_compression_components",
        type=int,
        default=1000,
        help="Number of components to use for PCA or random projection",
    )
    parser.add_argument(
        "--append-layer-type",
        dest="append_layer_type",
        action="store_true",
        help="append layer type to output file name",
    )
    parser.add_argument(
        "--output-method",
        dest="output_method",
        type=str,
        default='combined',
        choices=['combined', 'separate'],
        help="Output method to use (default: combined)"
    )
    parser.add_argument(
        "--layer-config",
        dest="layer_config",
        type=str,
        default=None,
        help="Layer configuration for ResNetFlex model (default: None)",
    )
    parser.add_argument(
        "--ssl-method",
        dest="ssl_method",
        type=str,
        default=None,
        help='SSL method to use (default: None)'
    )
    parser.add_argument(
        "--model-commitment",
        dest="model_commitment",
        type=str,
        default=None,
        help='Model commitment to use (default: None)'
    )
    parser.add_argument(
        "--benchmark-layer",
        dest="benchmark_layer",
        type=str,
        default=None,
        help='Layer to benchmark'
    )
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help="Skip existing files"
    )
    parser.add_argument(
        '--adv-config',
        dest='adv_config',
        type=str,
        default=None,
        help='Path to adversarial config file (default: None)'
    )
    
    return parser