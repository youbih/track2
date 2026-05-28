import os
import json
import glob
import logging
from typing import Dict
from omegaconf import OmegaConf
from my_affectgpt.common.registry import registry

class Config:
    def __init__(self, args):
        self.config = {}
        self.args = args
        registry.register("configuration", self)
        
        # overwrite [model_config, runner_config, dataset_config] with user_config
        options = self.args.options
        cfg_path = self.args.cfg_path
        user_config = self._build_opt_list(options)
        model_config = self.build_model_config(cfg_path, **user_config)
        runner_config = self.build_runner_config(cfg_path, **user_config)
        dataset_config = self.build_dataset_config(cfg_path, **user_config)
        inference_config = self.build_inference_config(cfg_path, **user_config) # 新加一个 inference config 文件
        self.config = OmegaConf.merge(runner_config, model_config, dataset_config, inference_config)

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)
    
    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        
        has_equal = opts[0].find("=") != -1
        if has_equal:
            return opts
        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]
    
    @staticmethod
    def _build_subconfig(cfg_path, key, **kwargs):
        """Generic helper to extract and merge a top-level config section.

        Args:
            cfg_path: Path to the YAML config file.
            key: Top-level key to extract (e.g., 'run', 'model', 'inference').
            **kwargs: Optional user overrides from command line.
        Returns:
            OmegaConf object containing the merged config section.
        """
        config = OmegaConf.load(cfg_path)
        section = config.get(key, None)
        assert section is not None, f"Missing {key} configuration in {cfg_path}."

        section_config = OmegaConf.create()
        if key in kwargs:
            section_config = OmegaConf.merge(
                section_config,
                {key: section},
                {key: kwargs[key]}
            )
        else:
            section_config = OmegaConf.merge(
                section_config,
                {key: section}
            )
        return section_config

    @staticmethod
    def build_runner_config(cfg_path, **kwargs):
        config = OmegaConf.load(cfg_path)
        # Only set default output_dir when not explicitly specified in YAML
        if not config.run.get("output_dir", None):
            output_dir = os.path.basename(cfg_path).rsplit('.', 1)[0]
            config.run.output_dir = os.path.join('output', output_dir)
        return Config._build_subconfig(cfg_path, "run", **kwargs)

    @staticmethod
    def build_model_config(cfg_path, **kwargs):
        return Config._build_subconfig(cfg_path, "model", **kwargs)

    @staticmethod
    def build_inference_config(cfg_path, **kwargs):
        return Config._build_subconfig(cfg_path, "inference", **kwargs)

    @staticmethod
    def build_dataset_config(cfg_path, **kwargs):
        config = OmegaConf.load(cfg_path)
        datasets = config.get("datasets", None)
        assert datasets is not None, "Missing datasets configuration file."

        dataset_config = OmegaConf.create()
        for dataset_name in datasets:
            temp_config = OmegaConf.create()
            if "datasets" in kwargs and dataset_name in kwargs["datasets"]:
                temp_config = OmegaConf.merge(
                    temp_config,
                    {dataset_name: config["datasets"][dataset_name]},
                    {dataset_name: kwargs["datasets"][dataset_name]}
                )
            else:
                temp_config = OmegaConf.merge(
                    temp_config,
                    {dataset_name: config["datasets"][dataset_name]}
                )
            dataset_config = OmegaConf.merge(
                dataset_config,
                {"datasets": temp_config},
            )
        return dataset_config

    def get_config(self):
        return self.config
    
    @property
    def run_cfg(self):
        return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    @property
    def model_cfg(self):
        return self.config.model
    
    @property
    def inference_cfg(self):
        return self.config.inference

    # print config infos
    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    # write into logging
    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)
