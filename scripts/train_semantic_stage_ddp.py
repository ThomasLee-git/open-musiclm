import argparse
import os
import sys
import datetime
import time

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from open_musiclm.config import (create_clap_quantized_from_config,
                                 create_hubert_kmeans_from_config,
                                 create_semantic_transformer_from_config,
                                 load_model_config, load_training_config)
from open_musiclm.open_musiclm import SemanticStage
from scripts.train_utils import validate_train_args, get_latest_checkpoints
from open_musiclm.config import MusicLMModelConfig, MusicLMTrainingConfig
from open_musiclm.optimizer import get_optimizer, get_linear_scheduler
from open_musiclm.mp_data import get_distributed_shared_filelist
from open_musiclm.data import SoundDataset

def train_ddp(args, model_config:MusicLMModelConfig, training_config:MusicLMTrainingConfig):
    """multi-node multi-gpu"""
    import torch.distributed as torch_dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DistributedSampler, random_split,DataLoader

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

    # init conditioners
    tmp_device = 'cpu'
    trainer_cfg = training_config.semantic_trainer_cfg
    use_preprocessed_data = trainer_cfg.use_preprocessed_data
    if use_preprocessed_data:
        clap = None
        wav2vec = None
        print(f'training from preprocessed data {trainer_cfg.folder}')
    else:
        print('loading clap...')
        clap = create_clap_quantized_from_config(model_config, args.rvq_path, tmp_device).to(device)
        if args.use_batch_kmeans:
            print("loading wav2vec with batch_kmeans...")
        else:
            print("loading wav2vec with original kmeans...")
        wav2vec = create_hubert_kmeans_from_config(
            model_config,
            args.kmeans_path,
            tmp_device,
            use_batch_kmeans=args.use_batch_kmeans,
        ).to(device)
    # init network
    print('loading semantic stage...')
    transformer = create_semantic_transformer_from_config(model_config, args.fine_tune_from, None).to(device)
    # optimizer and scheduler
    optimizer = get_optimizer(transformer.parameters(), lr=trainer_cfg.lr, wd=trainer_cfg.wd)
    scheduler = None
    if trainer_cfg.lr_warmup > 0:
        scheduler = get_linear_scheduler(optimizer, total_iters=trainer_cfg.lr_warmup)
    # load checkpoints
    global_steps = 0
    if (args.continue_from_dir):
        checkpoint_paths, global_steps = get_latest_checkpoints(args.continue_from_dir, args.continue_from_step)
        for path, model in zip(checkpoint_paths, (transformer, optimizer, scheduler)):
            if path and model:
                tmp_state_dict = torch.load(path, map_location=device)
                model.load_state_dict(tmp_state_dict)
    # ddp
    transformer = DistributedDataParallel(transformer)

    # dataset config
    stage = "semantic"
    ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic')
    ds_target_sample_hz = np.array((clap.sample_rate, wav2vec.target_sample_hz))
    ds_normalize = np.array((False, True))
    ds_seq_len_multiple_of = wav2vec.seq_len_multiple_of
    ds_data_max_length_seconds = np.array((model_config.global_cfg.semantic_audio_length_seconds, model_config.global_cfg.semantic_audio_length_seconds))

    # dataset
    np_list, np_addr_list = get_distributed_shared_filelist(
        world_size,
        rank,
        local_rank,
        trainer_cfg.filelist_path,
        trainer_cfg.blacklist_path,
    )
    assert (
        np_list is not None
    ), f"{rank=} {local_rank=} np_list is None"
    assert (
        np_addr_list is not None
    ), f"{rank=} {local_rank=} np_addr_list is None"
    dataset = SoundDataset(
        np_list,
        np_addr_list,
        trainer_cfg.folder,
        max_length_seconds=ds_data_max_length_seconds,
        normalize=ds_normalize,
        target_sample_hz=ds_target_sample_hz,
        seq_len_multiple_of=ds_seq_len_multiple_of,
        ignore_files=None,
        ignore_load_errors=True
    )
    if trainer_cfg.valid_frac > 0:
        train_size = int((1 - trainer_cfg.valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
        print(
            f'training with dataset of {len(dataset)} samples and validating with randomly splitted {len(valid_dataset)} samples')

    else:
        valid_dataset = dataset
        print(f'training with shared training and valid dataset of {len(dataset)} samples')
    dataset_sampler = DistributedSampler(dataset)
    valid_dataset_sampler = DistributedSampler(valid_dataset)
    num_workers = 16
    dataloader = DataLoader(
        dataset,
        sampler=dataset_sampler,
        batch_size=trainer_cfg.batch_size,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=valid_dataset_sampler,
        batch_size=trainer_cfg.batch_size,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    # wrapper
    model_wrapper = SemanticStage(semantic_transformer=transformer, wav2vec=wav2vec, clap=clap,cross_entropy_loss_weights=trainer_cfg.cross_entropy_loss_weights)
    # train loop
    acc_train_loss = 0.
    dataset_epoch_idx = 0
    valid_dataset_epoch_idx = 0
    while global_steps < trainer_cfg.num_train_steps:
        dataset_sampler.set_epoch(dataset_epoch_idx)
        torch_dist.barrier()
        if rank == 0:
            epoch_time = time.time()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader, start=1):
            train_loss = None
            grad_norms = None
            valid_loss = None
            valid_accuracy = None
            transformer.train()
            batch_data, batch_names = batch[:-1], batch[-1]
            print(
                f"{rank=} {local_rank=} {global_steps=} {batch_idx=} {batch_names=}"
            )
            batch_data = [d.to(device) for d in batch_data]
            data_kwargs = dict(zip(ds_fields, batch_data))
            if batch_idx % trainer_cfg.grad_accum_every:
                # accumulate gradients
                with transformer.no_sync():
                    tmp_loss, _, _ = model_wrapper(**data_kwargs, return_loss=True)
                    acc_train_loss += tmp_loss.item()
                    tmp_loss.backward()
            else:
                tmp_loss, _, _ = model_wrapper(all_token_ids=(clap_tokens, semantic_tokens), return_loss=True)
                acc_train_loss += tmp_loss.item()
                if trainer_cfg.max_grad_norm is not None:
                    grad_norms = torch.nn.utils.clip_grad_norm_(transformer.parameters(), trainer_cfg.max_grad_norm).item()
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
                train_loss = acc_train_loss / trainer_cfg.grad_accum_every
                acc_train_loss = 0.
                print(
                    f"{datetime.datetime.now()}: {global_steps=} {batch_idx=} {train_loss=} {grad_norms=}"
                )

            # sample results every so often
            if not (global_steps % trainer_cfg.save_results_every):
                transformer.eval()
                valid_dataset_sampler.set_epoch(valid_dataset_epoch_idx)
                for batch in valid_dataloader:
                    batch_data, batch_names = batch[:-1], batch[-1]
                    batch_data = [d.to(device) for d in batch_data]
                    data_kwargs = dict(zip(ds_fields, batch_data))
                    with torch.no_grad():
                        valid_loss, all_logits, all_labels = model_wrapper(
                            **data_kwargs, return_loss=True
                        )
                        # reduce
                        torch_dist.all_gather_into_tensor(pred_tokens,
                            all_logits[-1].argmax(1).contiguous()
                        )
                        torch_dist.all_gather_into_tensor(gt_tokens, all_labels[-1].contiguous())
                        pred_tokens = pred_tokens.detach().cpu().long()
                        gt_tokens = gt_tokens.detach().cpu().long()
                        valid_accuracy = (pred_tokens == gt_tokens).float().mean().item()
                        torch_dist.all_reduce(valid_loss)
                        valid_loss /= world_size
                        print(
                            f"{datetime.datetime.now()}: {rank=} {local_rank=} {global_steps=} {valid_loss=} {valid_accuracy=}")
                    break
                # update idx
                valid_dataset_epoch_idx += 1
            # TODO: log
            # self.accelerator.log(
            #     {
            #         "train_loss": train_loss,
            #         "grad_norms": grad_norms,
            #         "valid_loss": valid_loss,
            #         "valid_accuracy": valid_accuracy,
            #     },
            #     step=global_steps,
            # )

            # TODO: save model every so often
            # if rank == 0 and not (global_steps % trainer_cfg.save_model_every):
            #     model_path = str(
            #         self.results_folder / f"{self.stage}.transformer.{steps}.pt"
            #     )
            #     optim_path = str(
            #         self.results_folder / f"{self.stage}.optimizer.{steps}.pt"
            #     )
            #     scheduler_path = str(
            #         self.results_folder / f"{self.stage}.scheduler.{steps}.pt"
            #     )
            #     self.save(model_path, optim_path, scheduler_path)
            #     # save audio conditioner (clap) rvq checkpoint
            #     if exists(self.audio_conditioner) and self.audio_conditioner.learn_rvq:
            #         rvq_state_dict = self.audio_conditioner.rq.state_dict()
            #         rvq_path = str(
            #             self.results_folder / f"{self.stage}.conditioner_rvq.{steps}.pt"
            #         )
            #         torch.save(rvq_state_dict, rvq_path)
            #     self.print(
            #         f"{datetime.datetime.now()}: {steps=} done saving model to {model_path=} {optim_path=} {scheduler_path=}"
            #     )
            # update steps
            step_list = [None]
            if rank == 0:
                global_steps += 1
                step_list[0] = global_steps
            torch_dist.broadcast_object_list(step_list, src=0)
            global_steps = step_list[0]
            print(
                f"{datetime.datetime.now()}: {rank=} {local_rank=} updated {global_steps=}"
            )

        # update epoch
        if rank == 0:
            epoch_time = time.time() - epoch_time
        dataset_epoch_idx += 1
        print(f"{datetime.datetime.now()} done {dataset_epoch_idx=} {epoch_time=}")
    print(f"{datetime.datetime.now()} done training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train semantic stage")
    parser.add_argument("--results_folder", default=None, type=str)
    parser.add_argument("--project_dir", default=None, type=str)
    parser.add_argument(
        "--use_batch_kmeans", default=True, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--continue_from_dir", default=None, type=str)
    parser.add_argument("--continue_from_step", default=None, type=int)
    parser.add_argument("--model_config", default=None, type=str)
    parser.add_argument("--training_config", default=None, type=str)
    parser.add_argument("--rvq_path", default=None, type=str)
    parser.add_argument("--kmeans_path", default=None, type=str)
    parser.add_argument("--fine_tune_from", default=None, type=str)

    args = parser.parse_args()

    validate_train_args(args)

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    train_ddp(args, model_config, training_config)

    