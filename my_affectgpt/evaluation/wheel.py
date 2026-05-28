import os
import glob
import random
import numpy as np

from toolkit.utils.read_files import *
from toolkit.utils.chatgpt import *
from toolkit.utils.functions import *
import config


#########################################################################
######## 采用得到的mapping，去度量 gt and pred openset 之间的重叠度 ########
## => 所有评价指标，都是把不在词表中的元素直接剔除掉 [统一化处理，方便后续比较]
##########################################################################
def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + '/*.npy'):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


def _load_label_mappings():
    mapping_path = config.OUTSIDE_WHEEL_MAPPING
    format_mapping = np.load(mapping_path, allow_pickle=True)['format_mapping'].tolist()  # level3 -> level2
    raw_mapping = np.load(mapping_path, allow_pickle=True)['raw_mapping'].tolist()  # level2 -> level1
    wheel_map_whole = np.load(mapping_path, allow_pickle=True)['wheel_map_whole'].tolist()  # level1 -> level0
    return format_mapping, raw_mapping, wheel_map_whole


def calculate_exact_match_accuracy(name2gt, name2pred, process_names=None, metric='raw'):
    format_mapping, raw_mapping, wheel_map_whole = _load_label_mappings()
    if metric.startswith('case3'):
        _, wheelname, levelname = metric.split('_')
        wheel_map = wheel_map_whole[wheelname][levelname]
    else:
        wheel_map = None

    if process_names is None:
        process_names = [name for name in name2gt]

    matched = 0
    total = 0
    for name in process_names:
        if name not in name2pred:
            continue
        gt_labels = string_to_list(name2gt[name])
        gt_labels = [item.lower().strip() for item in gt_labels if item is not None and item.strip() != ""]
        if metric != 'raw':
            mapped = []
            for label in gt_labels:
                if label not in format_mapping:
                    continue
                if metric.startswith('case1'):
                    candidates = format_mapping[label]
                    mapped.append(random.choice(candidates))
                elif metric.startswith('case2'):
                    candidates = format_mapping[label]
                    level2 = random.choice(candidates)
                    if level2 in raw_mapping:
                        mapped.append(random.choice(raw_mapping[level2]))
                elif metric.startswith('case3'):
                    for fmt in format_mapping[label]:
                        for raw in raw_mapping[fmt]:
                            if raw in wheel_map:
                                mapped.append(wheel_map[raw])
                                break
            gt_labels = [l for l in mapped if l]
        gt_labels = set(sorted(set(gt_labels)))

        pred_labels = string_to_list(name2pred[name])
        pred_labels = [item.lower().strip() for item in pred_labels if item is not None and item.strip() != ""]
        if metric != 'raw':
            mapped = []
            for label in pred_labels:
                if label not in format_mapping:
                    continue
                if metric.startswith('case1'):
                    candidates = format_mapping[label]
                    mapped.append(random.choice(candidates))
                elif metric.startswith('case2'):
                    candidates = format_mapping[label]
                    level2 = random.choice(candidates)
                    if level2 in raw_mapping:
                        mapped.append(random.choice(raw_mapping[level2]))
                elif metric.startswith('case3'):
                    for fmt in format_mapping[label]:
                        for raw in raw_mapping[fmt]:
                            if raw in wheel_map:
                                mapped.append(wheel_map[raw])
                                break
            pred_labels = [l for l in mapped if l]
        pred_labels = set(sorted(set(pred_labels)))

        matched += int(gt_labels == pred_labels)
        total += 1

    if total == 0:
        return 0.0
    return matched / total


def calculate_micro_metrics(name2gt, name2pred, process_names=None, metric='raw'):
    format_mapping, raw_mapping, wheel_map_whole = _load_label_mappings()
    if metric.startswith('case3'):
        _, wheelname, levelname = metric.split('_')
        wheel_map = wheel_map_whole[wheelname][levelname]
    else:
        wheel_map = None

    if process_names is None:
        process_names = [name for name in name2gt]

    true_positive = 0
    false_positive = 0
    false_negative = 0
    total = 0

    for name in process_names:
        if name not in name2pred:
            continue

        gt_labels = string_to_list(name2gt[name])
        gt_labels = [item.lower().strip() for item in gt_labels if item is not None and item.strip() != ""]
        if metric != 'raw':
            mapped = []
            for label in gt_labels:
                if label not in format_mapping:
                    continue
                if metric.startswith('case1'):
                    candidates = format_mapping[label]
                    mapped.append(random.choice(candidates))
                elif metric.startswith('case2'):
                    candidates = format_mapping[label]
                    level2 = random.choice(candidates)
                    if level2 in raw_mapping:
                        mapped.append(random.choice(raw_mapping[level2]))
                elif metric.startswith('case3'):
                    for fmt in format_mapping[label]:
                        for raw in raw_mapping[fmt]:
                            if raw in wheel_map:
                                mapped.append(wheel_map[raw])
                                break
            gt_labels = [l for l in mapped if l]
        gt_labels = set(sorted(set(gt_labels)))

        pred_labels = string_to_list(name2pred[name])
        pred_labels = [item.lower().strip() for item in pred_labels if item is not None and item.strip() != ""]
        if metric != 'raw':
            mapped = []
            for label in pred_labels:
                if label not in format_mapping:
                    continue
                if metric.startswith('case1'):
                    candidates = format_mapping[label]
                    mapped.append(random.choice(candidates))
                elif metric.startswith('case2'):
                    candidates = format_mapping[label]
                    level2 = random.choice(candidates)
                    if level2 in raw_mapping:
                        mapped.append(random.choice(raw_mapping[level2]))
                elif metric.startswith('case3'):
                    for fmt in format_mapping[label]:
                        for raw in raw_mapping[fmt]:
                            if raw in wheel_map:
                                mapped.append(wheel_map[raw])
                                break
            pred_labels = [l for l in mapped if l]
        pred_labels = set(sorted(set(pred_labels)))

        true_positive += len(gt_labels & pred_labels)
        false_positive += len(pred_labels - gt_labels)
        false_negative += len(gt_labels - pred_labels)
        total += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluated_samples": total,
    }


def wheel_metric_calculation(gt_root=None, gt_csv=None, name2gt=None,
                             openset_root=None, openset_npz=None, name2pred=None,
                             process_names=None, inter_print=True, level='level1'):

    if level == 'level1':
        candidate_metrics = [
                            'case3_wheel1_level1',
                            'case3_wheel2_level1',
                            'case3_wheel3_level1',
                            'case3_wheel4_level1',
                            'case3_wheel5_level1',
                            ]
    elif level == 'level2':
        candidate_metrics = [
                            'case3_wheel1_level2',
                            'case3_wheel2_level2',
                            'case3_wheel3_level2',
                            'case3_wheel4_level2',
                            'case3_wheel5_level2',
                            ]

    format_mapping, raw_mapping, wheel_map_whole = _load_label_mappings()

    # read name2gt
    if name2gt is None:
        if gt_root is not None:
            name2gt = func_get_name2reason(gt_root)
        elif gt_csv is not None:
            name2gt = {}
            names = func_read_key_from_csv(gt_csv, 'name')
            gts   = func_read_key_from_csv(gt_csv, 'openset')
            for (name, gt) in zip(names, gts):
                name2gt[name] = gt

    # read name2pred
    if name2pred is None:
        if openset_root is not None:
            name2pred = func_get_name2reason(openset_root)
        elif openset_npz is not None:
            names = np.load(openset_npz)['filenames']
            items = np.load(openset_npz)['fileitems']
            name2pred = {}
            for (name, item) in zip(names, items):
                name2pred[name] = item

    if process_names is None:
        process_names = [name for name in name2gt]

    whole_scores = []
    for metric in candidate_metrics:
        _, wheelname, levelname = metric.split('_')
        wheel_map = wheel_map_whole[wheelname][levelname]

        accuracy, recall = [], []
        for name in process_names:
            gt = string_to_list(name2gt[name])
            gt = [item.lower().strip() for item in gt]
            mapped_gt = []
            for label in gt:
                if label not in format_mapping:
                    continue
                level1_whole = []
                for fmt in format_mapping[label]:
                    for raw in raw_mapping[fmt]:
                        level1_whole.append(raw)
                random.shuffle(level1_whole)
                found = ""
                for level1 in level1_whole:
                    if level1 in wheel_map:
                        found = wheel_map[level1]
                        break
                if found:
                    mapped_gt.append(found)
            gt = set(mapped_gt)

            pred = string_to_list(name2pred[name])
            pred = [item.lower().strip() for item in pred]
            mapped_pred = []
            for label in pred:
                if label not in format_mapping:
                    continue
                level1_whole = []
                for fmt in format_mapping[label]:
                    for raw in raw_mapping[fmt]:
                        level1_whole.append(raw)
                random.shuffle(level1_whole)
                found = ""
                for level1 in level1_whole:
                    if level1 in wheel_map:
                        found = wheel_map[level1]
                        break
                if found:
                    mapped_pred.append(found)
            pred = set(mapped_pred)

            if len(gt) == 0:
                continue
            if len(pred) == 0:
                accuracy.append(0)
                recall.append(0)
            else:
                accuracy.append(len(gt & pred) / len(pred))
                recall.append(len(gt & pred) / len(gt))

        if inter_print:
            print('process number (after filter): ', len(accuracy))

        if len(accuracy) != 0:
            avg_accuracy = np.mean(accuracy)
        else:
            avg_accuracy = 0

        if len(recall) != 0:
            avg_recall = np.mean(recall)
        else:
            avg_recall = 0

        if inter_print:
            print(f'avg acc: {avg_accuracy} avg recall: {avg_recall}')

        if avg_accuracy + avg_recall == 0:
            fscore = 0
        else:
            fscore = 2 * (avg_accuracy * avg_recall) / (avg_accuracy + avg_recall)
        whole_scores.append([fscore, avg_accuracy, avg_recall])

    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores
