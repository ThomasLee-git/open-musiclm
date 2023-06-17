import os
import sys

import torch
import torchaudio
from einops import rearrange
from pathlib import Path
import argparse
import math
import time

import torch.distributed as torch_dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, random_split, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from open_musiclm.config import load_model_config, create_musiclm_from_config


def get_prompts(path: str):
    if not path:
        return None
    with open(path, mode="r") as rf:
        prompts = rf.read().splitlines()
    return prompts


def extend_list(orig_list: list, repeat: int) -> list:
    tmp_list = list()
    for x in orig_list:
        tmp_list.extend([x for _ in range(repeat)])
    return tmp_list


def divide_list(orig_list: list, num_parts: int) -> list:
    num_elements_per_part = int(math.ceil(len(orig_list) / num_parts))
    sliced_list = list()
    for i in range(num_parts):
        sliced_list.append(
            orig_list[i * num_elements_per_part : (i + 1) * num_elements_per_part]
        )
    return sliced_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run inference using ddp")
    parser.add_argument("--num_samples", default=4, type=int)
    # parser.add_argument('--num_top_matches', default=1, type=int)
    parser.add_argument(
        "--model_config",
        default="./configs/model/musiclm_small.json",
        help="path to model config",
    )
    parser.add_argument(
        "--semantic_path", required=True, help="path to semantic stage checkpoint"
    )
    parser.add_argument(
        "--coarse_path", required=True, help="path to coarse stage checkpoint"
    )
    parser.add_argument(
        "--fine_path", required=True, help="path to fine stage checkpoint"
    )
    parser.add_argument("--rvq_path", default="./checkpoints/clap.rvq.350.pt")
    parser.add_argument(
        "--kmeans_path", default="./results/hubert_kmeans/kmeans.joblib"
    )
    parser.add_argument("--results_folder", default="./results", type=str)
    parser.add_argument(
        "--return_coarse_wave", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--duration",
        default=4,
        type=float,
        help="duration of audio to generate in seconds",
    )
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)

    # set up env
    assert torch_dist.is_nccl_available(), f"nccl is not supported"
    backend = "nccl"
    device_str = "cuda"
    torch_dist.init_process_group(backend=backend)
    world_size = torch_dist.get_world_size()
    rank = torch_dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(device_str, local_rank)
    # NOTE: manually set device for broadcasting
    torch.cuda.set_device(local_rank)
    print(f"{rank=} {local_rank=} {device=}")

    semantic_path = args.semantic_path
    coarse_path = args.coarse_path
    fine_path = args.fine_path
    return_coarse_wave = args.return_coarse_wave
    duration = args.duration
    kmeans_path = args.kmeans_path
    rvq_path = args.rvq_path
    seed = args.seed
    if seed < 0:
        seed = time.time() * 1e6
        print(f"{rank=} {local_rank=} reset {seed=}")
    torch.manual_seed(seed)
    results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    musiclm = create_musiclm_from_config(
        model_config=model_config,
        semantic_path=semantic_path,
        coarse_path=coarse_path,
        fine_path=fine_path,
        rvq_path=rvq_path,
        kmeans_path=kmeans_path,
        device=device,
    )
    generated_wavs = musiclm.forward3(
        num_samples=args.num_samples,
        output_seconds=duration,
        semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds,
        fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds,
        semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
        acoustic_steps_per_second=model_config.encodec_cfg.output_hz,
        return_coarse_generated_wave=return_coarse_wave,
    )
    generated_wavs = generated_wavs.detach().cpu()
    for i, wave in enumerate(generated_wavs):
        save_path = results_folder.joinpath(f"rank{rank}_unconditinal_gen{i}.mp3")
        torchaudio.save(save_path, wave, musiclm.neural_codec.sample_rate)
        print(f"done saving {save_path.as_posix()}")
