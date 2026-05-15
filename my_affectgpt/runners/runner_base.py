"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import glob
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from my_affectgpt.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from my_affectgpt.common.registry import registry
from my_affectgpt.common.utils import is_url
from my_affectgpt.datasets.data_utils import reorg_datasets_by_split, ChainDataset
from my_affectgpt.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id
        self.run_started_at = datetime.datetime.now()

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        self.history = {}

        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        这块是让模型能够支持多机多卡操作
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                # if self._wrapped_model is None:
                #     self._wrapped_model = DDP(
                #         self._model, device_ids=[self.config.run_cfg.gpu]
                #     )
                if self._wrapped_model is None:
                    # 启用find_unused_parameters=True，可以解决模型中有些参数没有梯度的问题，但是会增加显存的使用
                    # 最好写模型的时候就保证所有参数都有用到 zhuofan
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu],
                        find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        ## self.optimizer 只初始化 self._optimizer 一次
        ## 后面要用的话，就直接调用 self._optimizer 了
        if self._optimizer is None:

            ## trainable params 中有些层的 weight decay 设置为0
            ## weight decay 表示正则项系数，防止模型过拟合
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {"params": p_wd,     "weight_decay": float(self.config.run_cfg.weight_decay)},
                {"params": p_non_wd, "weight_decay": 0},
            ]

            ## 依据梯度信息设置 _optimizer
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    # 混合精度训练就需要设置scaler
    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                if torch.__version__.startswith('2.4.0') or torch.__version__.startswith('2.6.0'):
                    self._scaler = torch.amp.GradScaler('cuda')
                elif torch.__version__.startswith('2.1.0'):
                    self._scaler = torch.cuda.amp.GradScaler()
                else:
                    raise RuntimeError(f'Unsupported torch version: {torch.__version__}')
        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            max_epoch = self.max_epoch
            min_lr = self.min_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(self.datasets[split_name], list):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets  = [self.datasets[split]       for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def val_freq(self):
        return max(1, int(self.config.run_cfg.get("val_freq", 1)))

    @property
    def early_stop_patience(self):
        return int(self.config.run_cfg.get("early_stop_patience", 0))

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def monitor_splits(self):
        return self.config.run_cfg.get("monitor_splits", [])

    @property
    def metric_splits(self):
        split_order = []
        for split_name in list(self.monitor_splits) + list(self.valid_splits) + list(self.test_splits):
            if split_name not in split_order:
                split_order.append(split_name)
        return split_order

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    # self.dataloaders是一个函数，是需要调用的
    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]
        return train_dataloader

    def setup_output_dir(self):
        
        output_dir = Path(self.config.run_cfg.output_dir) / self.job_id
        result_dir = output_dir / "result"
        curve_dir = output_dir / "curves"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        curve_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir
        self.curve_dir = curve_dir

    def should_run_periodic_eval(self, cur_epoch):
        return (cur_epoch % self.val_freq == 0) or (cur_epoch == self.max_epoch)

    def get_metric_sample_limit(self, split_name):
        default_limit = self.config.run_cfg.get("metric_max_samples", None)
        split_limits = self.config.run_cfg.get("metric_max_samples_by_split", {})
        if split_name in split_limits:
            return int(split_limits[split_name])
        if default_limit in [None, "", 0]:
            return None
        return int(default_limit)

    def update_history(self, split_name, epoch, stats):
        if split_name not in self.history:
            self.history[split_name] = []

        normalized = {"epoch": int(epoch)}
        for key, value in stats.items():
            if isinstance(value, str):
                try:
                    normalized[key] = float(value)
                except ValueError:
                    normalized[key] = value
            else:
                normalized[key] = value
        self.history[split_name].append(normalized)

    def save_history(self):
        history_path = self.curve_dir / "metrics_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def build_plot_metadata(self):
        dataset_summary = []
        for dataset_name, dataset_cfg in self.config.datasets_cfg.items():
            ratio = dataset_cfg.get("ratio", 1.0)
            val_ratio = dataset_cfg.get("val_ratio", None)
            split_role = dataset_cfg.get("split_role", "train")
            train_monitor_ratio = dataset_cfg.get("train_monitor_ratio", 0.0)
            dataset_summary.append(
                f"{dataset_name}: split_role={split_role}, ratio={ratio}, "
                f"val_ratio={val_ratio}, train_monitor_ratio={train_monitor_ratio}"
            )

        return "\n".join(
            [
                f"run={self.job_id}",
                f"started_at={self.run_started_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"max_epoch={self.max_epoch}, iters_per_epoch={self.lr_scheduler.iters_per_epoch}, val_freq={self.val_freq}",
                f"batch_train={self.config.run_cfg.batch_size_train}, batch_eval={self.config.run_cfg.batch_size_eval}",
                f"early_stop_patience={self.early_stop_patience}, lr={self.init_lr}, min_lr={self.min_lr}",
                *dataset_summary,
            ]
        )

    def save_training_plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:
            logging.warning("Skipping training plot because matplotlib is unavailable: %s", error)
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        if "train" in self.history:
            epochs = [item["epoch"] for item in self.history["train"]]
            losses = [item.get("loss") for item in self.history["train"]]
            axes[0].plot(epochs, losses, marker="o", label="train_loss")

        for split_name, label in [("train_monitor", "train_monitor"), ("val", "val"), ("test", "test")]:
            if split_name in self.history:
                epochs = [item["epoch"] for item in self.history[split_name]]
                losses = [item.get("loss") for item in self.history[split_name]]
                axes[0].plot(epochs, losses, marker="o", label=f"{label}_loss")

                f1_values = [item.get("f1") for item in self.history[split_name] if "f1" in item]
                f1_epochs = [item["epoch"] for item in self.history[split_name] if "f1" in item]
                if len(f1_values) > 0:
                    axes[1].plot(f1_epochs, f1_values, marker="o", label=f"{label}_micro_f1")

                wheel_f1_values = [item.get("wheel_f1") for item in self.history[split_name] if "wheel_f1" in item]
                wheel_f1_epochs = [item["epoch"] for item in self.history[split_name] if "wheel_f1" in item]
                if len(wheel_f1_values) > 0:
                    axes[1].plot(wheel_f1_epochs, wheel_f1_values, marker="x", linestyle="--", label=f"{label}_wheel_f1")

        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("F1")
        axes[1].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.3)
        axes[1].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(f"Training Curves: {self.job_id}", fontsize=14)
        fig.text(0.01, 0.01, self.build_plot_metadata(), fontsize=9, family="monospace", va="bottom")
        fig.tight_layout(rect=[0, 0.1, 1, 0.97])
        fig.savefig(self.curve_dir / "training_curves.png", dpi=200)
        plt.close(fig)

    ############################
    ## main training process
    ############################
    def train(self):
        start_time = time.time()
        best_epoch = 0
        best_agg_metric = float("-inf")
        no_improve_validations = 0
        self.log_config() # config -> self.output_dir/log.txt

        # resume from checkpoint if specified => 'not evaluate_only' = 'train'
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)
        
        ## save initial model for debug
        self._save_checkpoint(cur_epoch=0, train_stats=None, is_best=False)

        for cur_epoch in range(self.start_epoch+1, self.max_epoch+1):

            # training phase
            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)
                self.update_history("train", cur_epoch, train_stats)
                self.save_history()

            # evaluation phase
            eval_splits = []
            if self.should_run_periodic_eval(cur_epoch):
                for split_name in list(self.monitor_splits) + list(self.valid_splits):
                    if split_name not in eval_splits:
                        eval_splits.append(split_name)

            if len(eval_splits) > 0:
                improved_on_val = False
                for split_name in eval_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            if split_name in self.valid_splits:
                                assert (
                                    "agg_metrics" in val_log
                                ), "No agg_metrics found in validation log."

                                agg_metrics = val_log["agg_metrics"]
                                if agg_metrics > best_agg_metric:
                                    best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                    improved_on_val = True
                                    self._save_checkpoint(cur_epoch, is_best=True)

                            if split_name in self.valid_splits:
                                val_log.update({"best_epoch": best_epoch})

                            self.log_stats(val_log, split_name)
                            self.update_history(split_name, cur_epoch, val_log)

                if improved_on_val:
                    no_improve_validations = 0
                elif len(self.valid_splits) > 0:
                    no_improve_validations += 1

                self.save_history()
                self.save_training_plot()

                if self.early_stop_patience > 0 and no_improve_validations >= self.early_stop_patience:
                    logging.info(
                        "Early stopping triggered after %s validation rounds without improvement.",
                        no_improve_validations,
                    )
                    break

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(cur_epoch, train_stats, is_best=False)

            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 and best_epoch > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)
        self.save_history()
        self.save_training_plot()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    # evaluate for test_splits
    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )
                if test_logs[split_name] is not None:
                    self.log_stats(test_logs[split_name], split_name)
                    self.update_history(split_name, cur_epoch if isinstance(cur_epoch, int) else self.max_epoch, test_logs[split_name])

            return test_logs

    def train_epoch(self, epoch):
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model, # support distributed training
            data_loader=self.train_loader, # load train dataloader
            optimizer=self.optimizer, # set weight decay to reduce over-fitting
            scaler=self.scaler, # for amp training
            lr_scheduler=self.lr_scheduler, # setup learning rate
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq, # print log for 50 iters
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(
            model,
            data_loader,
            cuda_enabled=self.cuda_enabled,
            split_name=split_name,
            runner=self,
        )

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        ## 这是 create_loaders 里面的一个子函数
        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(dataset, wds.DataPipeline):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(), # rank是当前进程编号
                    )
                    if not self.use_dist_eval_sampler: # 默认是 True
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz, # 每个进程的 batch_size
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler, # 每个进程的采样器
                    shuffle=sampler is None and is_train, # 默认是 False，因为已经在 sampler 内部采样了
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        # 相当于数据分组读取了
        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            # Validation/test splits are reorganized as single-item lists.
            # They should use a plain loader instead of MultiIterLoader.
            if (isinstance(dataset, list) or isinstance(dataset, tuple)) and (not is_train) and len(dataset) == 1:
                dataset = dataset[0]
                if isinstance(collate_fn, list) or isinstance(collate_fn, tuple):
                    collate_fn = collate_fn[0]

            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, train_stats=None, is_best=False):
        """
        Save the checkpoint at the current epoch. (only for trainable params)
        """
        ## case1: 原始保存方式
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = { # {param: whether trainable}
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict() # {param: weight}
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict, # only save trainable params
            "optimizer": self.optimizer.state_dict(), # first time will generate 'self._optimizer'
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }

        if is_best:
            save_to = os.path.join(self.output_dir, "checkpoint_best.pth")
            # delete old non-best checkpoints to save disk space
            for old_ckpt in glob.glob(os.path.join(self.output_dir, "checkpoint_[0-9]*.pth")):
                os.remove(old_ckpt)
                logging.info("Removed old checkpoint: {}".format(old_ckpt))
        elif not train_stats: # 模型的 zero-shot performance
            save_to = os.path.join(self.output_dir, "checkpoint_%06d_loss_%s.pth" %(cur_epoch, '0.000'))
        else:
            save_to = os.path.join(self.output_dir, "checkpoint_%06d_loss_%s.pth" %(cur_epoch, train_stats['loss']))
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

        # ## case2: transformers 保存方式 [直接保存模型和tokenizer] => 测试是否采用这种可以解决batch-calling的问题？
        # model_no_ddp = self.unwrap_dist_model(self.model)
        # if is_best:
        #     save_to = os.path.join(self.output_dir, "checkpoint_best")
        # elif not train_stats: # 模型的 zero-shot performance
        #     save_to = os.path.join(self.output_dir, "checkpoint_%06d_loss_%s" %(cur_epoch, '0.000'))
        # else:
        #     save_to = os.path.join(self.output_dir, "checkpoint_%06d_loss_%s" %(cur_epoch, train_stats['loss']))
        # logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        # # 保存模型和 tokenizer
        # model_no_ddp.save_pretrained(save_to)
        # model_no_ddp.llama_tokenizer.save_pretrained(save_to)


    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device, weights_only=True)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device, weights_only=True)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    # stats -> self.output_dir/log.txt
    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass
    
    # self.config -> self.output_dir/log.txt
    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
