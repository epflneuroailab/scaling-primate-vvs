import torch.nn as nn
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

def postprocess_score(x):
    x = {
        'score': float(x),
        'error': float(x.attrs['error']),
        'raw': float(x.attrs['raw']),
        'ceiling': float(x.attrs['ceiling'])
    }
    
    return x


class PostprocessWrapper(nn.Module):
    """
    A wrapper class that performs post-processing on features of specific layers of a given model.

    Args:
        model (nn.Module): The original model.
        layers (list): List of layer names to perform post-processing on.
        postprocess_func (callable): The post-processing function to apply on the layer outputs.
        skip_layers (list): List of layer names to skip post-processing on.

    Attributes:
        layers (list): List of layer names to perform post-processing on.
        layers_ (list): List of modified layer names (with '.' replaced by '_').
        postprocess_func (callable): The post-processing function to apply on the layer outputs.
        feature_extractor (nn.Module): The feature extractor module.
    """

    def __init__(self, model, layers, postprocess_func, skip_layers=[], separator='/', layer_names_modified=False):
        super().__init__()


        if not layer_names_modified:
            layer_map = {l: l.replace('.', separator) for l in layers}
        else:
            layer_map = {l.replace(separator, '.'):l for l in layers}
        self.separator = separator
        self.layers = list(layer_map.keys())
        self.layers_ = list(layer_map.values())
        self.skip_layers = skip_layers
        self.postprocess_func = postprocess_func
        
        train_nodes, eval_nodes = get_graph_node_names(model)
        assert all([l in eval_nodes for l in self.layers])
        self.feature_extractor = create_feature_extractor(model, return_nodes=layer_map)
        for l in self.layers_:
            setattr(self, l, nn.Identity())