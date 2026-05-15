import os
import copy
import logging
import random
import warnings

from my_affectgpt.common.registry import registry
from my_affectgpt.datasets.datasets.base_dataset import BaseDataset
from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from my_affectgpt.datasets.datasets.mer2026ov_dataset import MER2026OV_Dataset
from my_affectgpt.datasets.datasets.human_dataset import Human_Dataset
from my_affectgpt.datasets.datasets.mercaptionplus_dataset import MERCaptionPlus_Dataset


# get name -> dataset_cls
def get_name2cls(dataset):
    if dataset == 'Human': return Human_Dataset()
    if dataset == 'MERCaptionPlus': return MERCaptionPlus_Dataset()
    if dataset == 'MER2026OV': return MER2026OV_Dataset()
    print ('dataset cls not provided!')
    return None


def _split_annotations(base_dataset, val_ratio, split_seed):
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1."

    total_size = len(base_dataset.annotation)
    assert total_size > 1, "Need at least 2 samples to create a train/val split."

    indices = list(range(total_size))
    rng = random.Random(split_seed)
    permuted = indices[:]
    rng.shuffle(permuted)

    val_size = max(1, int(total_size * val_ratio))
    val_index_set = set(permuted[:val_size])
    train_indices = [idx for idx in indices if idx not in val_index_set]
    val_indices = [idx for idx in indices if idx in val_index_set]

    train_dataset = copy.deepcopy(base_dataset)
    val_dataset = copy.deepcopy(base_dataset)
    train_dataset.annotation = [train_dataset.annotation[idx] for idx in train_indices]
    val_dataset.annotation = [val_dataset.annotation[idx] for idx in val_indices]
    return train_dataset, val_dataset


def _sample_annotations(dataset, ratio, seed, log_prefix):
    if ratio in [None, ""] or float(ratio) >= 1:
        return dataset

    ratio = float(ratio)
    assert 0 < ratio <= 1, "ratio must be between 0 and 1."

    total_size = len(dataset.annotation)
    sampled_size = max(1, int(total_size * ratio))
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(total_size), sampled_size))
    dataset.annotation = [dataset.annotation[idx] for idx in sampled_indices]

    logging.info(
        "%s sampled with ratio=%s, kept=%s/%s.",
        log_prefix,
        ratio,
        len(dataset.annotation),
        total_size,
    )
    return dataset


def _build_train_monitor_dataset(train_dataset, dataset_cfg, split_seed, dataset_name):
    monitor_ratio = float(dataset_cfg.get("train_monitor_ratio", 0.0))
    if monitor_ratio <= 0:
        return None

    monitor_dataset = copy.deepcopy(train_dataset)
    monitor_dataset = _sample_annotations(
        monitor_dataset,
        ratio=monitor_ratio,
        seed=split_seed + 1,
        log_prefix=f"{dataset_name} train_monitor",
    )
    return monitor_dataset


@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        logging.info("Building datasets MERCaptionPlus_Dataset")
        self.build_processors()
        self.dataset_cfg.apply_ratio_after_split = True

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        base_dataset = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )

        split_role = str(self.dataset_cfg.get("split_role", "train")).lower()
        if split_role == "test":
            datasets["test"] = base_dataset
            return datasets

        split_seed = int(self.dataset_cfg.get("split_seed", 42))
        enable_val_split = self.dataset_cfg.get("enable_val_split", False)

        if enable_val_split:
            val_ratio = float(self.dataset_cfg.get("val_ratio", 0.2))
            train_dataset, val_dataset = _split_annotations(
                base_dataset, val_ratio=val_ratio, split_seed=split_seed
            )
            logging.info(
                "MERCaptionPlus split with seed=%s, train=%s, val=%s.",
                split_seed,
                len(train_dataset.annotation),
                len(val_dataset.annotation),
            )
            datasets["val"] = val_dataset
        else:
            train_dataset = base_dataset

        train_dataset = _sample_annotations(
            train_dataset,
            ratio=self.dataset_cfg.get("ratio", 1.0),
            seed=split_seed,
            log_prefix="MERCaptionPlus train",
        )
        datasets["train"] = train_dataset

        train_monitor_dataset = _build_train_monitor_dataset(
            train_dataset, self.dataset_cfg, split_seed, "MERCaptionPlus"
        )
        if train_monitor_dataset is not None:
            datasets["train_monitor"] = train_monitor_dataset

        return datasets
    

@registry.register_builder("human")
class Human_Builder(BaseDatasetBuilder):
    train_dataset_cls = Human_Dataset

    def build_datasets(self):
        logging.info("Building datasets Human_Dataset")
        self.build_processors()
        self.dataset_cfg.apply_ratio_after_split = True

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        base_dataset = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )

        split_role = str(self.dataset_cfg.get("split_role", "train")).lower()
        if split_role == "test":
            datasets["test"] = base_dataset
            return datasets

        split_seed = int(self.dataset_cfg.get("split_seed", 42))

        enable_val_split = self.dataset_cfg.get("enable_val_split", False)
        if enable_val_split:
            val_ratio = float(self.dataset_cfg.get("val_ratio", 0.2))
            train_dataset, val_dataset = _split_annotations(
                base_dataset, val_ratio=val_ratio, split_seed=split_seed
            )

            logging.info(
                "Human dataset split with seed=%s, train=%s, val=%s.",
                split_seed,
                len(train_dataset.annotation),
                len(val_dataset.annotation),
            )
            datasets["val"] = val_dataset
        else:
            train_dataset = base_dataset

        train_dataset = _sample_annotations(
            train_dataset,
            ratio=self.dataset_cfg.get("ratio", 1.0),
            seed=split_seed,
            log_prefix="Human train",
        )
        datasets['train'] = train_dataset

        train_monitor_dataset = _build_train_monitor_dataset(
            train_dataset, self.dataset_cfg, split_seed, "Human"
        )
        if train_monitor_dataset is not None:
            datasets["train_monitor"] = train_monitor_dataset

        return datasets


@registry.register_builder("mer2026ov")
class MER2026OV_Builder(BaseDatasetBuilder):
    train_dataset_cls = MER2026OV_Dataset

    def build_datasets(self):
        logging.info("Building datasets MER2026OV_Dataset")
        self.build_processors()

        datasets = dict()
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            txt_processor=self.txt_processors["train"],
            img_processor=self.img_processors["train"],
            dataset_cfg=self.dataset_cfg,
            model_cfg=self.model_cfg,
            )
        return datasets
