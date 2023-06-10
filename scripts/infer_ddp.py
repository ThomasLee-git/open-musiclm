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
    parser.add_argument(
        "--text_prompt_path", type=str, default=None, help="path to prompts"
    )
    parser.add_argument(
        "--audio_prompt_path",
        type=str,
        default=None,
        help="path to audio for continuation generation",
    )
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
    results_folder = Path(args.results_folder)

    # scatter prompts
    text_prompt_list = get_prompts(args.text_prompt_path)
    audio_prompt_list = get_prompts(args.audio_prompt_path)
    assert (
        text_prompt_list is not None or audio_prompt_list is not None
    ), f"at least either text or audio should be present"
    if text_prompt_list and audio_prompt_list:
        assert len(text_prompt_list) == len(
            audio_prompt_list
        ), f"mismatched prompt size {len(text_prompt_list)} vs. {len(audio_prompt_list)}"
    rank_text_prompt_list = [[]]
    rank_audio_prompt_list = [[]]
    if text_prompt_list:
        text_prompt_list = extend_list(text_prompt_list, args.num_samples)
        text_prompt_list = divide_list(text_prompt_list, world_size)
        torch_dist.scatter_object_list(rank_text_prompt_list, text_prompt_list, src=0)
    rank_text_prompt_list = rank_text_prompt_list[0]
    if audio_prompt_list:
        audio_prompt_list = extend_list(audio_prompt_list, args.num_samples)
        audio_prompt_list = divide_list(audio_prompt_list, world_size)
        torch_dist.scatter_object_list(rank_audio_prompt_list, audio_prompt_list, src=0)
    rank_audio_prompt_list = rank_audio_prompt_list[0]
    print(f"{rank=} {local_rank=} {rank_text_prompt_list=} {rank_audio_prompt_list=}")
    if len(rank_text_prompt_list) < 1 and len(rank_audio_prompt_list) < 1:
        print("empty prompt, exit")
        exit(0)
    # read audio
    rank_audios = None
    audio_sr = None
    if len(rank_audio_prompt_list) > 0:
        audio_list = list()
        for audio_path in rank_audio_prompt_list:
            tmp_audio, tmp_sr = torchaudio.load(audio_path)
            # slice for condition and concatenation
            target_len = (
                int(tmp_sr * model_config.global_cfg.semantic_audio_length_seconds) + 1
            )
            tmp_audio = tmp_audio[:, :target_len]
            if audio_sr is None:
                audio_sr = tmp_sr
            else:
                assert audio_sr == tmp_sr, f"mismatched sample rate"
            audio_list.append(tmp_audio.unsqueeze(0))
        rank_audios = torch.cat(audio_list, dim=0).to(device)

    results_folder.mkdir(parents=True, exist_ok=True)
    rank_results_folder = results_folder.joinpath(f"rank_{rank}")
    rank_results_folder.mkdir(parents=True, exist_ok=True)

    musiclm = create_musiclm_from_config(
        model_config=model_config,
        semantic_path=semantic_path,
        coarse_path=coarse_path,
        fine_path=fine_path,
        rvq_path=rvq_path,
        kmeans_path=kmeans_path,
        device=device,
    )

    torch.manual_seed(seed)
    generated_wavs, similarities = musiclm.generate_top_match2(
        text_list=rank_text_prompt_list,
        audios=rank_audios,
        audio_sample_rate=audio_sr,
        output_seconds=duration,
        semantic_window_seconds=model_config.global_cfg.semantic_audio_length_seconds,
        coarse_window_seconds=model_config.global_cfg.coarse_audio_length_seconds,
        fine_window_seconds=model_config.global_cfg.fine_audio_length_seconds,
        semantic_steps_per_second=model_config.hubert_kmeans_cfg.output_hz,
        acoustic_steps_per_second=model_config.encodec_cfg.output_hz,
        return_coarse_generated_wave=return_coarse_wave,
    )
    generated_wavs = generated_wavs.detach().cpu()
    for i, (wave, sim) in enumerate(zip(generated_wavs, similarities)):
        print(f"{rank=} {local_rank=} prompt: {rank_text_prompt_list[i]} idx_{i}")
        save_path = rank_results_folder.joinpath(
            f"{rank_text_prompt_list[i][:25]}_gen{i}_sim{sim}.mp3"
        )
        torchaudio.save(save_path, wave, musiclm.neural_codec.sample_rate)
