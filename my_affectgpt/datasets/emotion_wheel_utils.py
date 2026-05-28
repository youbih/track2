import numpy as np
import torch
import config


class EmotionWheelMapper:
    """Maps emotion labels to emotion wheel binary targets."""

    def __init__(self, mapping_path=None):
        if mapping_path is None:
            mapping_path = config.OUTSIDE_WHEEL_MAPPING
        self._load_mapping(mapping_path)

    def _load_mapping(self, mapping_path):
        data = np.load(mapping_path, allow_pickle=True)
        self._format_mapping = data["format_mapping"].tolist()
        self._raw_mapping = data["raw_mapping"].tolist()
        self._wheel_map_whole = data["wheel_map_whole"].tolist()

        self._level1_to_wheels = {}
        for wname in self._wheel_map_whole:
            for level1 in self._wheel_map_whole[wname]["level1"]:
                if level1 not in self._level1_to_wheels:
                    self._level1_to_wheels[level1] = set()
                self._level1_to_wheels[level1].add(wname)

    def compute_targets(self, ovlabel_list):
        """Compute 5D binary wheel targets from ovlabel strings.

        Args:
            ovlabel_list: list of ovlabel strings (batch_size,)
        Returns:
            targets: [batch_size, 5] binary tensor
        """
        wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4", "wheel5"]
        batch_targets = []
        for ovlabel in ovlabel_list:
            labels = [l.strip().lower() for l in ovlabel.split(",") if l.strip()]
            present_wheels = set()
            for label in labels:
                if label not in self._format_mapping:
                    continue
                level2 = self._format_mapping[label][0]
                if level2 in self._raw_mapping:
                    level1 = self._raw_mapping[level2][0]
                    if level1 in self._level1_to_wheels:
                        present_wheels.update(self._level1_to_wheels[level1])
            target = [1.0 if w in present_wheels else 0.0 for w in wheel_names]
            batch_targets.append(target)
        return torch.tensor(batch_targets, dtype=torch.float32)
