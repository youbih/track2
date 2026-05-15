"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from my_affectgpt.common.registry import registry
from my_affectgpt.common.logger import MetricLogger
from my_affectgpt.datasets.data_utils import prepare_sample
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.evaluation.wheel import (
    calculate_exact_match_accuracy,
    calculate_micro_metrics,
    wheel_metric_calculation,
)
from my_affectgpt.tasks.base_task import BaseTask


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask): # 所有内容继承自 video_text_pretrain task
    def __init__(self):
        super().__init__()
        self._metric_chat = None
        self._metric_chat_model_id = None

    def _get_metric_chat(self, model, runner):
        model_id = id(model)
        if self._metric_chat is None or self._metric_chat_model_id != model_id:
            self._metric_chat = Chat(model, runner.config.model_cfg, device=str(runner.device))
            self._metric_chat_model_id = model_id
        return self._metric_chat

    def _extract_prediction_label(self, response):
        text = response.strip().replace("\n", " ")
        patterns = [
            r"emotional state is\s*(.+)",
            r"most likely label is\s*(.+)",
            r"sentiment state is\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match is not None:
                text = match.group(1)
                break

        text = text.split("###")[0].strip()
        text = text.rstrip(" .。")
        return text

    def _build_single_sample_data(self, samples, index):
        sample_data = {}
        key_mapping = {
            "audio": "audios",
            "raw_audio": "raw_audios",
            "frame": "frames",
            "raw_frame": "raw_frames",
            "face": "faces",
            "raw_face": "raw_faces",
            "image": "images",
            "raw_image": "raw_images",
        }
        for sample_key, batch_key in key_mapping.items():
            sample_data[sample_key] = samples[batch_key][index] if batch_key in samples else None
        return sample_data

    def _generate_predictions(self, samples, runner, chat, remaining_samples):
        if remaining_samples == 0:
            return []

        predictions = []
        batch_size = len(samples["name"])
        face_or_frame = samples["face_or_frame"]
        max_new_tokens = int(runner.config.run_cfg.get("metric_max_new_tokens", 128))

        for index in range(batch_size):
            if remaining_samples is not None and len(predictions) >= remaining_samples:
                break

            sample_data = self._build_single_sample_data(samples, index)
            audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
            frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
            face_hiddens, face_llms = chat.postprocess_face(sample_data)
            _, image_llms = chat.postprocess_image(sample_data)

            multi_llms = None
            if face_or_frame.startswith("multiface"):
                _, multi_llms = chat.postprocess_multi(face_hiddens, audio_hiddens)
            elif face_or_frame.startswith("multiframe"):
                _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)

            img_list = {
                "audio": audio_llms,
                "frame": frame_llms,
                "face": face_llms,
                "image": image_llms,
                "multi": multi_llms,
            }

            response = chat.answer_sample(
                prompt=samples["prompt"][index],
                img_list=img_list,
                num_beams=1,
                do_sample=False,
                top_p=1.0,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                max_length=int(runner.config.model_cfg.max_length),
            )
            predictions.append(
                {
                    "name": samples["name"][index],
                    "gt": samples["ovlabel"][index],
                    "pred": self._extract_prediction_label(response),
                    "raw_pred": response,
                }
            )

        return predictions

    def evaluation(self, model, data_loader, cuda_enabled=True, split_name=None, runner=None):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        print_freq = 10
        results = {"losses": [], "predictions": []}

        collect_predictions = runner is not None and split_name in runner.metric_splits
        chat = self._get_metric_chat(model, runner) if collect_predictions else None
        remaining_samples = runner.get_metric_sample_limit(split_name) if runner is not None else None

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            loss = self.valid_step(model=model, samples=samples)
            results["losses"].append(float(loss))

            if collect_predictions:
                current_limit = None
                if remaining_samples is not None:
                    current_limit = max(remaining_samples - len(results["predictions"]), 0)
                batch_predictions = self._generate_predictions(samples, runner, chat, current_limit)
                results["predictions"].extend(batch_predictions)

        return results

    def valid_step(self, model, samples):
        loss = model(samples)["loss"]
        return float(loss.item())

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        losses = val_result.get("losses", []) if isinstance(val_result, dict) else []
        predictions = val_result.get("predictions", []) if isinstance(val_result, dict) else []

        if len(losses) == 0:
            return {"agg_metrics": float("-inf"), "loss": float("inf")}

        avg_loss = sum(losses) / len(losses)
        # Runner expects a larger agg_metrics to indicate a better checkpoint.
        metrics = {
            "agg_metrics": -avg_loss,
            "loss": avg_loss,
        }

        if len(predictions) > 0:
            name2gt = {item["name"]: item["gt"] for item in predictions}
            name2pred = {item["name"]: item["pred"] for item in predictions}

            exact_match_acc = calculate_exact_match_accuracy(
                name2gt=name2gt,
                name2pred=name2pred,
                metric="raw",
            )
            micro_metrics = calculate_micro_metrics(
                name2gt=name2gt,
                name2pred=name2pred,
                metric="raw",
            )
            wheel_f1, wheel_precision, wheel_recall = wheel_metric_calculation(
                name2gt=name2gt,
                name2pred=name2pred,
                inter_print=False,
            )
            metrics.update(
                {
                    # Keep f1 as the primary metric alias for existing plotting/checkpoint logic.
                    "f1": micro_metrics["f1"],
                    "exact_match_acc": exact_match_acc,
                    "micro_precision": micro_metrics["precision"],
                    "micro_recall": micro_metrics["recall"],
                    "micro_f1": micro_metrics["f1"],
                    "wheel_precision": wheel_precision,
                    "wheel_recall": wheel_recall,
                    "wheel_f1": wheel_f1,
                    "metric_samples": len(predictions),
                    "evaluated_samples": micro_metrics["evaluated_samples"],
                }
            )
            metrics["agg_metrics"] = micro_metrics["f1"]

        return metrics
