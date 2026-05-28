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


def _build_standard_split_datasets(builder, dataset_name, default_seed=42,
                                    auto_seed=False, supports_test_split=False):
    """Shared dataset construction logic for builders that support
    train/val split, ratio sampling, and train_monitor.

    Args:
        builder: The BaseDatasetBuilder subclass instance.
        dataset_name: Human-readable dataset name for logging.
        default_seed: Default random seed for splitting/sampling.
        auto_seed: If True, generates a random seed when split_seed == -1.
        supports_test_split: If True, also creates a test split from training data.
    Returns:
        dict with 'train' and optionally 'val', 'test', 'train_monitor'.
    """
    logging.info("Building datasets %s", builder.train_dataset_cls.__name__)
    builder.build_processors()
    builder.dataset_cfg.apply_ratio_after_split = True

    datasets = dict()
    base_dataset = builder.train_dataset_cls(
        vis_processor=builder.vis_processors["train"],
        txt_processor=builder.txt_processors["train"],
        img_processor=builder.img_processors["train"],
        dataset_cfg=builder.dataset_cfg,
        model_cfg=builder.model_cfg,
    )

    split_role = str(builder.dataset_cfg.get("split_role", "train")).lower()
    if split_role == "test":
        datasets["test"] = base_dataset
        return datasets

    split_seed = int(builder.dataset_cfg.get("split_seed", default_seed))
    if auto_seed and split_seed == -1:
        import time
        split_seed = int(time.time()) % 100000
        logging.info("%s split using auto-generated seed=%s", dataset_name, split_seed)

    enable_val_split = builder.dataset_cfg.get("enable_val_split", False)
    train_dataset = base_dataset

    if enable_val_split:
        val_ratio = float(builder.dataset_cfg.get("val_ratio", 0.2))
        train_dataset, val_dataset = _split_annotations(
            base_dataset, val_ratio=val_ratio, split_seed=split_seed
        )
        datasets["val"] = val_dataset
        # Create prefixed val split for separate evaluation (e.g., machine_val, human_val)
        split_prefix = builder.dataset_cfg.get("split_prefix", "")
        if split_prefix:
            datasets[f"{split_prefix}val"] = val_dataset

    if supports_test_split:
        test_ratio = float(builder.dataset_cfg.get("test_ratio", 0.0))
        if test_ratio > 0 and len(train_dataset.annotation) > 1:
            train_dataset, test_dataset = _split_annotations(
                train_dataset, val_ratio=test_ratio, split_seed=split_seed + 100
            )
            datasets["test"] = test_dataset

    train_dataset = _sample_annotations(
        train_dataset,
        ratio=builder.dataset_cfg.get("ratio", 1.0),
        seed=split_seed,
        log_prefix=f"{dataset_name} train",
    )
    datasets["train"] = train_dataset

    train_monitor_dataset = _build_train_monitor_dataset(
        train_dataset, builder.dataset_cfg, split_seed, dataset_name
    )
    if train_monitor_dataset is not None:
        datasets["train_monitor"] = train_monitor_dataset

    # Unified logging
    log_parts = [f"{dataset_name} dataset split with seed={split_seed}",
                 f"train={len(datasets['train'].annotation)}"]
    if "val" in datasets:
        log_parts.append(f"val={len(datasets['val'].annotation)}")
    if "test" in datasets:
        log_parts.append(f"test={len(datasets['test'].annotation)}")
    if enable_val_split or (supports_test_split and "test" in datasets):
        logging.info(", ".join(log_parts) + ".")

    return datasets


@registry.register_builder("mercaptionplus")
class MERCaptionPlus_Builder(BaseDatasetBuilder):
    train_dataset_cls = MERCaptionPlus_Dataset

    def build_datasets(self):
        return _build_standard_split_datasets(
            self, "MERCaptionPlus", default_seed=42,
            auto_seed=False, supports_test_split=False
        )


@registry.register_builder("human")
class Human_Builder(BaseDatasetBuilder):
    train_dataset_cls = Human_Dataset

    def build_datasets(self):
        return _build_standard_split_datasets(
            self, "Human", default_seed=-1,
            auto_seed=True, supports_test_split=True
        )


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
