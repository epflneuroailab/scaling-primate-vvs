from typing import List, Dict

import numpy as np
import pandas as pd

def filter_arch_family_by_samples(df: pd.DataFrame, arch_family: List[str], samples_per_class: List[int]) -> pd.DataFrame:
    """
    Filters the DataFrame based on architecture family and samples per class.
    For a given list of architecture families, the function filters the DataFrame 
    to include only the rows that belong to the specified architecture families with 
    the specified number of samples per class, or any other model architectures.
    
    For example, we take all rows that do not belong to ViT and ConvNeXt
    such as ResNet and EfficientNet, while only keeping rows that belong to ViT and ConvNeXt
    with training set sizes of 300 or all samples per class.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be filtered.
    arch_family (List[str]): A list of architecture families to be considered.
    samples_per_class (List[int]): A list of sample counts per class to be considered.
    
    Returns:
    pd.DataFrame: A filtered DataFrame where rows belong to the specified architecture families 
                  and have the specified number of samples per class.
    """
    df = df[
        ~df.arch_family.isin(arch_family)
        | df.arch_family.isin(arch_family) & df.samples_per_class.isin(samples_per_class)
    ].copy()
    return df


def combine_arch_family(df: pd.DataFrame, arch_family_map:Dict[str, str]) -> pd.DataFrame:
    """
    Combines architecture families in the given DataFrame based on the provided mapping.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing architecture family samples.
    arch_family_map (Dict[str, str]): A dictionary mapping specific architecture family names to their general names.
                                        If not provided, a default mapping will be used.
    Returns:
    pd.DataFrame: The DataFrame with updated architecture family names.
    """
    
    if not arch_family_map or not isinstance(arch_family_map, dict):
        arch_family_map = {
            'ResNetFlex': 'ResNet', 
            'ViTFlex': 'ViT', 
            'ConvNeXtFlex': 'ConvNeXt'
        }

    df.arch_family = df.arch_family.apply(lambda x: arch_family_map[x] if x in arch_family_map else x)
    
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    
    df = df.copy()
    
    # Combine architecture families
    combine_arch_families = filters.get('combine_arch_families', False)
    if combine_arch_families:
        df = combine_arch_family(df, combine_arch_families)
    
    # Apply boolean filters
    boolean_filters = filters.get('boolean_filters', {})
    equals_true = boolean_filters.get('equals_true', [])
    equals_false = boolean_filters.get('equals_false', [])
    for fltr in equals_true:
        df = df[df[fltr]]
    for fltr in equals_false:
        df = df[~df[fltr]]
                
    # Apply set filters
    set_filters = filters.get('set_filters', {})
    for k, v in set_filters.items():
        if isinstance(v, list):
            df = df[df[k].isin(v)]
        else:
            df = df[df[k] == v]
            
    # Apply range filters
    range_filters = filters.get('range_filters', {})
    for k, v in range_filters.items():
        df = df[(df[k] >= v[0]) & (df[k] <= v[1])]

    # Apply composite filters
    composite_filters = filters.get('composite_filters', {})
    for fltr_name, cmpst_fltr in composite_filters.items():
        mask = np.ones(len(df), dtype=bool)
        for k, v in cmpst_fltr.items():
            if isinstance(v, list):
                mask &= df[k].isin(v)
            else:
                mask &= df[k] == v
        df = df[mask]
        
    # Apply group by
    group_by = filters.get('group_by', {})
    for gb_name, gb_props in group_by.items():
        keys = gb_props['keys']
        reduce = gb_props['reduce']
        df = df.groupby(keys).agg({**reduce}).reset_index()
        
    # Apply custom filters
    arch_families_n_samples = filters.get('arch_families_samples', {})
    if arch_families_n_samples:
        df = filter_arch_family_by_samples(df, **arch_families_n_samples)
        
        
    # Apply modifiers
    modifiers = filters.get('modifiers', {})
    flops_multiplier = modifiers.get('flops_multiplier', 1.0)
    df['total_flops'] = flops_multiplier * df['total_flops']


    
    df = df.reset_index(drop=True)
    return df

