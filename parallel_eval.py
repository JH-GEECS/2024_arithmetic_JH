import torch
import sys
import os

import numpy as np
from src.eval import eval_single_dataset_with_prediction, eval_single_dataset
# from src.main import save_scale_factors
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.modeling import ImageEncoder, ImageClassifier
from src.task_vectors import TaskVector
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import open_clip

import csv
import ray
from easydict import EasyDict
from datetime import datetime

from typing import List


class Args:
    def __init__(self):
        self.model = 'ViT-L-14'
        self.tasks = ["Cars", "DTD", "EuroSAT", "GTSRB",
                      "MNIST", "RESISC45", "SUN397", "SVHN"]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task_scale_factors = None
        self.save = '/data1/common_datasets/shared_weight/task_vector/ViT-L-14/'
        self.data_location = '/data1/common_datasets/vision_cls/'
        self.eval_datasets = None
        self.train_dataset = None
        self.exp_name = None
        self.results_db = None
        self.batch_size = 256
        self.lr = 0.001
        self.wd = 0.1
        self.ls = 0.0
        self.warmup_length = 500
        self.epochs = 10
        self.load = None
        self.cache_dir = None
        self.openclip_cachedir = '/data2/david3684/.cache/open_clip'
        self.initial_rank_ratio = 1.0
        self.low_rank_mode = 'SoRA'
        self.pretrained_model = 'openai'
        self.scale_shared_weight = False
        self.num_test_samples = 2048
        self.no_shared_weight = False


def transform_key(old_key):
    if old_key.startswith('shared.attn.layer') or old_key.startswith('clip_vit'):
        parts = old_key.split('.')
        layer_idx = parts[3]
        # print(layer_idx)
        sub_key = parts[4]
        if sub_key in ['q', 'k', 'v']:
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.{sub_key}_weight'
        elif sub_key == 'out_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight'
        elif sub_key == 'c_fc' or sub_key == 'c_proj':
            return f'model.visual.transformer.resblocks.{layer_idx}.mlp.{sub_key}.weight'
    return old_key


def format_shared_weight(shared_weight_state_dict, open_clip_state_dict_template):
    qkv_store = {}
    for old_key, value in shared_weight_state_dict.items():
        if 'diff' in old_key or 'scale_dict' in old_key:
            continue

        new_key = transform_key(old_key)
        layer_idx = new_key.split('.')[4]

        if layer_idx not in qkv_store:
            qkv_store[layer_idx] = {'q': None, 'k': None, 'v': None}

        weight_type = new_key.split('.')[-1]
        # in_proj.weight (q, k, v)
        if weight_type in ['q_weight', 'k_weight', 'v_weight']:
            qkv_store[layer_idx][weight_type[0]] = value
        else:  # out_proj.weight, c_fc.weight, c_proj.weight
            assert new_key in open_clip_state_dict_template
            open_clip_state_dict_template[new_key] = value

    for layer_idx, qkv in qkv_store.items():
        if all(v.bool().all().item() for v in qkv.values()):
            in_proj_weight = torch.cat([qkv['q'], qkv['k'], qkv['v']], dim=0)
            # concat qkv into 3072*1024 tensor
            new_key = f'model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight'
            assert new_key in open_clip_state_dict_template
            open_clip_state_dict_template[new_key] = in_proj_weight
        else:
            print(
                f"Missing q, k, or v for layer {layer_idx}. q: {qkv['q']}, k: {qkv['k']}, v: {qkv['v']}")

    return open_clip_state_dict_template

# 나머지 처리 필요


def eval_single_task_wrapper(is_single, encoder_state_dict, task, args, scale_coef):

    encoder = ImageEncoder(args, keep_lang=False)
    encoder.load_state_dict(encoder_state_dict)
    result = eval_single_dataset(encoder, task, args)

    if is_single:
        print(f"Running single task evaluation for {task}: {result}")
    else:
        print(f"Running multitask task evaluation for {task}: {result}")

    ret = {
        "is_single": is_single,
        "task": task,
        "top1": result["top1"],
        "scaling_coef": 1.0,
        "initial_rank_ratio_list": args.initial_rank_ratio,
    }
    if not is_single:
        ret["scaling_coef"] = scale_coef

    return ret


def result_writer(results: List):
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file_path = f"./eval_results_{current_time}.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = ["is_single", "task", "top1",
                      "initial_rank_ratio_list", "scaling_coef"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Results are written to {csv_file_path}")


if __name__ == '__main__':
    args = Args()  # overall task setting

    model_list = {}
    for idx, name in enumerate(args.tasks):
        print(f'Loading {name}')
        model = torch.load(
            f'/data1/common_datasets/shared_weight/task_vector/ViT-L-14/{name}/finetuned.pt').to(args.device)
        model_list[name] = model

    shared_weight_state_dict = torch.load(
        '/data1/common_datasets/shared_weight/20241025/vanilla_T8/rankmin_config_20241025_uni_T8_vanilla.bin')
    zero_shot_encoder = ImageEncoder(args, keep_lang=False)
    formatted_shared_weight = format_shared_weight(
        shared_weight_state_dict, zero_shot_encoder.state_dict())
    print('Shared weight formatted')
    zero_shot_encoder = zero_shot_encoder.to(args.device)

    # ray runner
    ray.init()
    eval_ray_runner = ray.remote(
        eval_single_task_wrapper).options(num_gpus=0.5, num_cpus=4)
    ray_pack = []

    experiment_vector = EasyDict()
    experiment_vector.initial_rank_ratio_list = [
        1.0, 0.64, 0.5, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.001, 0.0]

    experiment_vector.scaling_coef_list = [0.2, 0.3, 0.4, 0.5, 0.6, 1.0]

    for initial_rank_ratio in experiment_vector.initial_rank_ratio_list:
        print(f'Initial rank ratio: {initial_rank_ratio}')
        args.initial_rank_ratio = initial_rank_ratio
        low_rank_task_vectors = {}
        # Build low rank task vectors
        for task in args.tasks:
            model = model_list[task]
            finetuned_state_dict = model.state_dict()
            low_rank_task_vectors[task] = TaskVector(
                args, zero_shot_encoder.state_dict(), finetuned_state_dict, task).to(args.device)

        low_rank_task_vector_sum = sum(low_rank_task_vectors.values())

        for task in args.tasks:
            # print(f'Evaluate {task}')
            # Evaluate sinlge task model
            low_rank_single_task_encoder = low_rank_task_vectors[task].apply_to(
                deepcopy(zero_shot_encoder), scaling_coef=1.0)
            # single_task_reuslt = eval_single_dataset(low_rank_single_task_encoder, task, deepcopy(args))
            ray_pack.append(eval_ray_runner.remote(
                True, low_rank_single_task_encoder.state_dict(), task, deepcopy(args), each_scale_factor))
            print(f'Single task model for {task} fetched')
            for each_scale_factor in experiment_vector.scaling_coef_list:
                # Evaluate multi task model
                # print('Multi task model')
                low_rank_multi_task_encoder = low_rank_task_vector_sum.apply_to(
                    deepcopy(zero_shot_encoder), scaling_coef=each_scale_factor)
                # multi_task_result = eval_single_dataset(low_rank_multi_task_encoder, task, deepcopy(args))
                ray_pack.append(eval_ray_runner.remote(
                    False, low_rank_multi_task_encoder.state_dict(), task, deepcopy(args), each_scale_factor))
                print(f'Multi task model for {task} fetched')

    results = ray.get(ray_pack)
    result_writer(results)

    ray.shutdown()
