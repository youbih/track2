"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import gzip
import logging
import os
import random as rnd
import tarfile
import zipfile
import random
from typing import List
from tqdm import tqdm

import decord
from decord import VideoReader
import webdataset as wds
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.datasets.base_dataset import ConcatDataset

decord.bridge.set_bridge("torch")
MAX_INT = registry.get("MAX_INT")

class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: List[wds.DataPipeline]) -> None:
        super().__init__()
        self.datasets = datasets
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                logging.info("One of the datapipeline doesn't define ratio and set to 1 automatically.")

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()
    
    return apply_to_sample(_move_to_cuda, sample)

# prepare_sample: mvoe all tensor into cuda
def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    return samples


def deduplicate_cross_dataset(datasets):
    """
    Ensure no sample appears in both a val split and any other dataset's train split.
    Also remove duplicates when a sample appears in multiple val splits.

    For samples that appear in multiple val splits, keep them in the alphabetically
    first dataset and remove from others.

    Args:
        datasets: dict of {dataset_name: {split_name: dataset}}

    Returns:
        Modified datasets dict.
    """
    if len(datasets) <= 1:
        return datasets

    sorted_names = sorted(datasets.keys())

    # Step 1: Collect all val sample names per dataset (before any dedup)
    val_names_per_dataset = {}
    for dataset_name in sorted_names:
        val_names = set()
        splits = datasets[dataset_name]
        for split_name, dataset_split in splits.items():
            if split_name.endswith("val") and hasattr(dataset_split, "annotation"):
                for ann in dataset_split.annotation:
                    name = ann.get("name", "")
                    if name:
                        val_names.add(name)
        val_names_per_dataset[dataset_name] = val_names

    # Step 2: Build ownership map - each shared sample is owned by alphabetically first dataset
    # This ensures shared samples appear in only ONE val set
    sample_to_owner = {}
    for dataset_name in sorted_names:
        for name in val_names_per_dataset[dataset_name]:
            if name not in sample_to_owner:
                sample_to_owner[name] = dataset_name

    # Step 3: For each val split, keep only samples owned by this dataset
    for dataset_name in sorted_names:
        splits = datasets[dataset_name]
        for split_name, dataset_split in list(splits.items()):
            if not split_name.endswith("val") or not hasattr(dataset_split, "annotation"):
                continue
            original_len = len(dataset_split.annotation)
            dataset_split.annotation = [
                ann for ann in dataset_split.annotation
                if sample_to_owner.get(ann.get("name", ""), dataset_name) == dataset_name
            ]
            new_len = len(dataset_split.annotation)
            if original_len != new_len:
                print(f"[Deduplicate] Removed {original_len - new_len} samples from {dataset_name}/{split_name} "
                      f"(owned by earlier dataset)")

    # Step 4: Collect updated val names after cross-dataset dedup
    updated_val_names_per_dataset = {}
    for dataset_name in sorted_names:
        val_names = set()
        splits = datasets[dataset_name]
        for split_name, dataset_split in splits.items():
            if split_name.endswith("val") and hasattr(dataset_split, "annotation"):
                for ann in dataset_split.annotation:
                    name = ann.get("name", "")
                    if name:
                        val_names.add(name)
        updated_val_names_per_dataset[dataset_name] = val_names

    # Step 5: Remove OTHER datasets' val samples from each dataset's train
    for dataset_name in sorted_names:
        splits = datasets[dataset_name]
        if "train" not in splits:
            continue
        train_dataset = splits["train"]
        if not hasattr(train_dataset, "annotation"):
            continue

        # Collect val names from ALL OTHER datasets (after dedup)
        other_val_names = set()
        for other_name in sorted_names:
            if other_name != dataset_name:
                other_val_names.update(updated_val_names_per_dataset[other_name])

        original_len = len(train_dataset.annotation)
        train_dataset.annotation = [
            ann for ann in train_dataset.annotation
            if ann.get("name", "") not in other_val_names
        ]
        new_len = len(train_dataset.annotation)
        if original_len != new_len:
            print(f"[Deduplicate] Removed {original_len - new_len} samples from {dataset_name}/train "
                  f"(appear in other datasets' val)")

    return datasets


def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # Deduplicate before reorganization
    datasets = deduplicate_cross_dataset(datasets)

    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # concatenate map-style datasets and iterable-style datasets separately
            if len(iterable_datasets) > 1:
                chained_datasets = (
                    ChainDataset(iterable_datasets)
                )
            elif len(iterable_datasets) == 1:
                chained_datasets = iterable_datasets[0]
            else:
                chained_datasets = None

            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets

