import logging
import numpy as np
import os
from result_caching import store_dict
from typing import Dict, Literal, Union
from sklearn import random_projection
from copy import deepcopy

from brainscore_vision.model_helpers.activations.core import flatten, change_dict
from brainscore_vision.model_helpers.utils import fullname


class LayerRandomProjection:
    def __init__(
        self, 
        activations_extractor, 
        n_components, 
        projection_type:Literal['gaussian', 'sparse']='sparse',
        epsilon=1e-1,
        density:float | Literal['auto'] = "auto",
        random_state: Union[int, np.random.RandomState] = 0
    ):
        
        self._logger = logging.getLogger(fullname(self))
        self._extractor = activations_extractor
        self._n_components = n_components
        self._projection_type = projection_type
        self._epsilon = epsilon
        self._density = density
        self._random_state = random_state
        self._layer_transformers = {}
        print(f"Random State of LayerRandomProjection: {random_state}")
        

        if self._projection_type == 'gaussian':
            self._transformer = random_projection.GaussianRandomProjection(
                n_components=self._n_components,
                random_state=self._random_state
            )
        elif self._projection_type == 'sparse':
            self._transformer = random_projection.SparseRandomProjection(
                n_components=self._n_components,
                density=self._density,
                dense_output=True,
                eps=self._epsilon,
                random_state=self._random_state
            )
        else:
            raise ValueError('Invalid projection type')

    def __call__(self, batch_activations):
        self._ensure_initialized(batch_activations)

        def apply_random_projection(layer, activations):
            activations = flatten(activations)
            transformer = self._layer_transformers[layer]
            if transformer is None:
                return activations
            return transformer.transform(activations)

        return change_dict(batch_activations, apply_random_projection, keep_name=True,
                           multithread=os.getenv('MT_MULTITHREAD', '1') == '1')

    def _ensure_initialized(self, batch_activations: Dict[str, np.ndarray]):
        layers = list(batch_activations.keys())
        missing_layers = [layer for layer in layers if layer not in self._layer_transformers]
        if len(missing_layers) == 0:
            return
        missing_layer_activations = {layer: batch_activations[layer] for layer in missing_layers}
        layer_transformers = self._transformers(
            identifier=self._extractor.identifier,
            layers=list(missing_layer_activations.keys()),
            missing_layer_activations=missing_layer_activations,
            n_components=self._n_components,
            random_state=self._random_state
        )
        self._layer_transformers = {**self._layer_transformers, **layer_transformers}

    @store_dict(dict_key='layers', identifier_ignore=['layers', 'missing_layer_activations'])
    def _transformers(self, identifier, layers, missing_layer_activations, n_components, random_state):
        # if n_components is None:
        #     n_components = self._n_components
        def init_and_progress(layer, activations):
            activations = flatten(activations)
            if activations.shape[1] <= n_components:
                log = f"Not computing a random projection for {layer} " \
                        f"activations {activations.shape} as shape is small enough already"
                self._logger.debug(log)
                print(log)
                transformer = None
            else:
                transformer = deepcopy(self._transformer)
                transformer.fit(activations)
                # print(np.where(transformer.components_.toarray())[1])
            return transformer

        layer_transformers = change_dict(missing_layer_activations, init_and_progress, keep_name=True,
                                 multithread=os.getenv('MT_MULTITHREAD', '1') == '1')
        return layer_transformers

    @classmethod
    def hook(cls, activations_extractor, n_components, projection_type:Literal['gaussian', 'sparse']='gaussian', random_state:Union[int, np.random.RandomState]=0):
        hook = LayerRandomProjection(activations_extractor=activations_extractor, n_components=n_components, projection_type=projection_type, random_state=random_state)
        assert not cls.is_hooked(activations_extractor), "Random projection already hooked"
        handle = activations_extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle

    @classmethod
    def is_hooked(cls, activations_extractor):
        return any(isinstance(hook, cls) for hook in
                   activations_extractor._extractor._batch_activations_hooks.values())