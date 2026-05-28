"""
Single-GPU test evaluation script.
Reuses training pipeline to ensure correct train/val/test split.
"""
import os
import sys
import argparse
import numpy as np

import torch

from my_affectgpt.common.config import Config
from my_affectgpt.common.registry import registry
from my_affectgpt.common.logger import setup_logger
from my_affectgpt.conversation.conversation_video import Chat
from my_affectgpt.datasets.data_utils import prepare_sample
from my_affectgpt.evaluation.wheel import (
    calculate_exact_match_accuracy,
    calculate_micro_metrics,
    wheel_metric_calculation,
)

import my_affectgpt.tasks as tasks
from my_affectgpt.tasks import *
from my_affectgpt.models import *
from my_affectgpt.runners import *
from my_affectgpt.processors import *
from my_affectgpt.datasets.builders import *

import re


def extract_prediction_label(response):
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


def build_single_sample_data(samples, index):
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


def main():
    parser = argparse.ArgumentParser(description="Single-GPU Test Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to best checkpoint.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="overwrite params in yaml.",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = f"cuda:0"

    cfg = Config(args)
    setup_logger()
    cfg.pretty_print()

    # Build datasets using the same pipeline as training
    task = tasks.setup_task(cfg)
    all_datasets = task.build_datasets(cfg)

    # Find test split from any dataset builder
    test_dataset = None
    for ds_name, ds_splits in all_datasets.items():
        if "test" in ds_splits:
            test_dataset = ds_splits["test"]
            print(f"Found test split in '{ds_name}' dataset")
            break

    if test_dataset is None:
        print("ERROR: No test split found in any dataset!")
        return
    print(f"Test dataset size: {len(test_dataset.annotation)}")

    # Build model
    model = task.build_model(cfg)

    # Load best checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
    except RuntimeError:
        print("Key mismatch, loading with strict=False")
        model.load_state_dict(checkpoint["model"], strict=False)
    del checkpoint

    model = model.to(device).eval()
    print("Model loaded and set to eval mode.")

    # Create chat for generation
    chat = Chat(model, cfg.model_cfg, device=device)

    # Create dataloader for test split
    from torch.utils.data import DataLoader
    from my_affectgpt.datasets.datasets.dataloader_utils import PrefetchLoader

    collate_fn = getattr(test_dataset, "collater", None)
    if collate_fn is None:
        from my_affectgpt.datasets.datasets.base_dataset import collate_fn as default_collate
        collate_fn = default_collate

    data_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    data_loader = PrefetchLoader(data_loader)

    # Run evaluation
    face_or_frame = cfg.datasets_cfg.human.face_or_frame
    max_new_tokens = int(cfg.run_cfg.get("metric_max_new_tokens", 128))
    max_length = int(cfg.model_cfg.max_length)

    predictions = []
    total = len(data_loader)

    print(f"Starting test evaluation on {total} samples...")

    with torch.no_grad():
        for i, samples in enumerate(data_loader):
            samples = prepare_sample(samples, cuda_enabled=True)

            batch_size = len(samples["name"])
            for index in range(batch_size):
                sample_data = build_single_sample_data(samples, index)
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
                    max_length=max_length,
                )

                pred = extract_prediction_label(response)
                predictions.append({
                    "name": samples["name"][index],
                    "gt": samples["ovlabel"][index],
                    "pred": pred,
                    "raw_pred": response,
                })

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[{i+1}/{total}] pred: {predictions[-1]['pred']} | gt: {predictions[-1]['gt']}")

    # Calculate metrics
    print("\n" + "=" * 60)
    print("TEST EVALUATION RESULTS")
    print("=" * 60)

    name2gt = {item["name"]: item["gt"] for item in predictions}
    name2pred = {item["name"]: item["pred"] for item in predictions}

    exact_match_acc = calculate_exact_match_accuracy(name2gt=name2gt, name2pred=name2pred, metric="raw")
    micro_metrics = calculate_micro_metrics(name2gt=name2gt, name2pred=name2pred, metric="raw")
    wheel_f1, wheel_precision, wheel_recall = wheel_metric_calculation(
        name2gt=name2gt, name2pred=name2pred, inter_print=True
    )

    print(f"Samples evaluated: {micro_metrics['evaluated_samples']}")
    print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
    print(f"Micro Precision: {micro_metrics['precision']:.4f}")
    print(f"Micro Recall: {micro_metrics['recall']:.4f}")
    print(f"Micro F1: {micro_metrics['f1']:.4f}")
    print(f"Wheel Precision: {wheel_precision:.4f}")
    print(f"Wheel Recall: {wheel_recall:.4f}")
    print(f"Wheel F1: {wheel_f1:.4f}")

    # Save predictions
    output_dir = os.path.dirname(args.ckpt_path)
    save_path = os.path.join(output_dir, "test_predictions.npz")
    np.savez_compressed(save_path, predictions=predictions)
    print(f"\nPredictions saved to {save_path}")


if __name__ == "__main__":
    main()
