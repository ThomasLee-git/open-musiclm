import itertools
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import rmtree
import os

# ThomasLee
import datetime
import accelerate
from .mp_data import get_distributed_shared_filelist

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import (Accelerator, DistributedDataParallelKwargs,
                        DistributedType)
from beartype.door import is_bearable
from beartype.typing import Dict, List, Literal, Optional, Union
from beartype.vale import Is
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from typing_extensions import Annotated

from .clap_quantized import ClapQuantized
from .data import (PreprocessedDataset, SoundDataset, get_dataloader,
                   get_preprocessed_dataloader)
from .hf_hubert_kmeans import HfHubertWithKmeans, learn_kmeans, HfHubertWithBatchKmeans
from .model_types import NeuralCodec, Wav2Vec
from .open_musiclm import (CoarseStage, FineStage, SemanticStage,
                           TokenConditionedTransformer)
from .optimizer import get_linear_scheduler, get_optimizer
from .utils import (all_rows_have_eos_id, append_eos_id,
                    batch_unique_consecutive, beartype_jit, ceil_div,
                    copy_file_to_folder, default, eval_decorator, exists,
                    generate_mask_with_prob, get_embeds, gumbel_sample,
                    mask_out_after_eos_id, round_down_nearest_multiple, top_k)

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    input_audio=Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def sanitize_hparams(hps):
    for key, value in hps.items():
        if not (
            isinstance(value, int) or
            isinstance(value, float) or
            isinstance(value, str) or
            isinstance(value, bool) or
            isinstance(value, torch.Tensor)
        ):
            hps[key] = str(value)
    return hps

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))


def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)


def noop(*args, **kwargs):
    pass


@beartype_jit
class SingleStageTrainer(nn.Module):
    """
    General trainer for any stage of MusicLM.
        semantic: requires audio_conditioner and wav2vec
        coarse: requires audio_conditioner, wav2vec, and neural_codec
        fine: requires audio_conditioner and neural_codec
    """

    def __init__(
        self,
        transformer: TokenConditionedTransformer,
        stage: Literal['semantic', 'coarse', 'fine'],
        *,
        num_train_steps,
        batch_size,
        model_config,
        training_config,
        dataset: Optional[Dataset] = None,
        wav2vec: Optional[Wav2Vec] = None,
        neural_codec: Optional[NeuralCodec] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        data_max_length_seconds = 1,
        ignore_files: Optional[List[str]]=None,
        cross_entropy_loss_weights: Optional[List[float]]=None,
        ignore_load_errors=True,
        folder=None,
        filelist_path: str = None,
        blacklist_path: str = None,
        use_preprocessed_data=False,
        lr=3e-4,
        lr_warmup=0,
        grad_accum_every=1,
        wd=0.,
        max_grad_norm=0.5,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_predicted_tokens=True,
        save_reconstructed_wave=True,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        kwargs_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            **accelerate_kwargs, kwargs_handlers=[kwargs_handler]
        )
        # ThomasLee
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index
        # move
        if exists(transformer):
            transformer.to(self.accelerator.device)
        if exists(wav2vec):
            wav2vec.to(self.accelerator.device)
        if exists(audio_conditioner):
            audio_conditioner.to(self.accelerator.device)
        if exists(neural_codec):
            neural_codec.to(self.accelerator.device)

        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None

        self.use_preprocessed_data = use_preprocessed_data
        self.model_config = model_config
        self.training_config = training_config

        self.transformer = transformer

        self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner
        self.neural_codec = neural_codec

        self.stage = stage

        if stage == 'semantic':
            assert self.use_preprocessed_data or (exists(audio_conditioner) and exists(wav2vec))
            self.train_wrapper = SemanticStage(
                semantic_transformer=transformer,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'semantic_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic')
                target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz)
                normalize = (False, True)
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'coarse':
            assert self.use_preprocessed_data or (exists(wav2vec) and exists(audio_conditioner) and exists(neural_codec))
            self.train_wrapper = CoarseStage(
                coarse_transformer=transformer,
                neural_codec=neural_codec,
                wav2vec=wav2vec,
                clap=audio_conditioner,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'semantic_token_ids', 'coarse_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_semantic', 'raw_wave_for_acoustic')
                target_sample_hz = (audio_conditioner.sample_rate, wav2vec.target_sample_hz, neural_codec.sample_rate)
                normalize = (False, True, False)
                seq_len_multiple_of = wav2vec.seq_len_multiple_of
        elif stage == 'fine':
            assert self.use_preprocessed_data or (exists(audio_conditioner) and exists(neural_codec))
            self.train_wrapper = FineStage(
                fine_transformer=transformer,
                clap=audio_conditioner,
                neural_codec=neural_codec,
                cross_entropy_loss_weights=default(cross_entropy_loss_weights, [0., 0., 1.])
            )
            if self.use_preprocessed_data:
                self.ds_fields = ('clap_token_ids', 'coarse_token_ids', 'fine_token_ids')
            else:
                self.ds_fields = ('raw_wave_for_clap', 'raw_wave_for_acoustic')
                target_sample_hz = (audio_conditioner.sample_rate, neural_codec.sample_rate)
                normalize = (False, False)
                seq_len_multiple_of = None
        else:
            raise ValueError(f'invalid stage: {stage}')

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr=lr, wd=wd)

        if lr_warmup > 0:
            self.scheduler = get_linear_scheduler(
                self.optim,
                total_iters=lr_warmup,
            )
        else:
            self.scheduler = None

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        if self.use_preprocessed_data:
            self.ds = PreprocessedDataset(
                folder,
                stage=self.stage,
                semantic_window_seconds=int(self.model_config.global_cfg.semantic_audio_length_seconds),
                coarse_window_seconds=int(self.model_config.global_cfg.coarse_audio_length_seconds),
                fine_window_seconds=int(self.model_config.global_cfg.fine_audio_length_seconds),
                semantic_steps_per_second=self.model_config.hubert_kmeans_cfg.output_hz,
                acoustic_steps_per_second=self.model_config.encodec_cfg.output_hz,
            )
        else:
            self.ds = dataset
            if not exists(self.ds):
                assert exists(
                    folder
                ), "folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training"
                np_list, np_addr_list = get_distributed_shared_filelist(
                    self.accelerator.num_processes,
                    self.rank,
                    self.local_rank,
                    filelist_path,
                    blacklist_path,
                )
                assert (
                    np_list is not None
                ), f"rank={self.rank} local_rank={self.local_rank} np_list is None"
                assert (
                    np_addr_list is not None
                ), f"rank={self.rank} local_rank={self.local_rank} np_addr_list is None"
                self.ds = SoundDataset(
                    np_list,
                    np_addr_list,
                    folder,
                    # filelist_path=filelist_path,
                    # blacklist_path=blacklist_path,
                    max_length_seconds=data_max_length_seconds,
                    normalize=normalize,
                    target_sample_hz=target_sample_hz,
                    seq_len_multiple_of=seq_len_multiple_of,
                    ignore_files=default(ignore_files, None),
                    ignore_load_errors=ignore_load_errors
                )

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader
        dataloader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "num_workers": 8,
            # "multiprocessing_context": "forkserver",
        }
        if self.use_preprocessed_data:
            self.dl = get_preprocessed_dataloader(self.ds, **dataloader_kwargs)
            self.valid_dl = get_preprocessed_dataloader(
                self.valid_ds, **dataloader_kwargs
            )
        else:
            self.dl = get_dataloader(self.ds, **dataloader_kwargs)
            self.valid_dl = get_dataloader(self.valid_ds, **dataloader_kwargs)

        # prepare with accelerator
        # ThomasLee: according to accelerate documentation, modules SHOULD be passed together
        if exists(self.scheduler):
            (
                self.train_wrapper,
                self.optim,
                self.scheduler,
                self.dl,
                self.valid_dl
            ) = self.accelerator.prepare(
                self.train_wrapper,
                self.optim,
                self.scheduler,
                self.dl,
                self.valid_dl
            )
        else:
            (
                self.train_wrapper,
                self.optim,
                self.dl,
                self.valid_dl
            ) = self.accelerator.prepare(
                self.train_wrapper,
                self.optim,
                self.dl,
                self.valid_dl
            )
        # if exists(self.scheduler):
        #     self.scheduler = self.accelerator.prepare(self.scheduler)

        # dataloader iterators

        # self.dl_iter = iter(self.dl)
        # self.valid_dl_iter = iter(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.save_predicted_tokens = save_predicted_tokens
        self.save_reconstructed_wave = save_reconstructed_wave

        self.results_folder = Path(results_folder)

        # if self.is_main and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
        #     rmtree(str(self.results_folder))

        #     self.results_folder.mkdir(parents=True, exist_ok=True)

        if exists(save_reconstructed_wave):
            self.waves_folder = self.results_folder / 'reconstructed_waves'
            self.waves_folder.mkdir(parents=True, exist_ok=True)
        if exists(save_predicted_tokens):
            self.tokens_folder = self.results_folder / 'tokens'
            self.tokens_folder.mkdir(parents=True, exist_ok=True)

        hps = asdict(self.model_config.global_cfg)
        if stage == 'semantic':
            hps.update(asdict(self.model_config.semantic_cfg))
            hps.update(asdict(self.training_config.semantic_trainer_cfg))
        elif stage == 'coarse':
            hps.update(asdict(self.model_config.coarse_cfg))
            hps.update(asdict(self.training_config.coarse_trainer_cfg))
        elif stage == 'fine':
            hps.update(asdict(self.model_config.fine_cfg))
            hps.update(asdict(self.training_config.fine_trainer_cfg))

        hps = sanitize_hparams(hps)

        if 'tensorboard' in self.log_with:
            self.accelerator.init_trackers(f"{stage}_stage_{int(time.time() * 1000)}", config=hps)
        else:
            self.accelerator.init_trackers(f"{stage}_stage", config=hps)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

        print(f"dataloader len {len(self.dl)}")
        self.to(self.accelerator.device)
        self.accelerator.wait_for_everyone()

    def save(self, model_path, optim_path, scheduler_path=None):
        model_state_dict = self.accelerator.get_state_dict(self.transformer)
        torch.save(model_state_dict, model_path)

        optim_state_dict = self.optim.state_dict()
        torch.save(optim_state_dict, optim_path)

        if exists(self.scheduler):
            assert exists(scheduler_path)
            scheduler_state_dict = self.scheduler.state_dict()
            torch.save(scheduler_state_dict, scheduler_path)

    def load(self, model_path, optim_path, scheduler_path=None, steps=0):
        model_path = Path(model_path)
        optim_path = Path(optim_path)
        assert model_path.exists() and optim_path.exists()

        model_state_dict = torch.load(model_path, map_location=self.device)
        optim_state_dict = torch.load(optim_path, map_location=self.device)
        transformer = self.accelerator.unwrap_model(self.transformer)
        transformer.load_state_dict(model_state_dict)
        self.optim.load_state_dict(optim_state_dict)

        if exists(self.scheduler):
            assert exists(scheduler_path), 'the config specifies lr warmup is used, but no scheduler checkpoint is given. try setting lr_warmup to 0.'
            scheduler_path = Path(scheduler_path)
            assert scheduler_path.exists()
            scheduler_state_dict = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state_dict)

        if steps > 0:
            assert int(self.steps.item()) == 0, 'steps should be 0 when loading a checkpoint for the first time'
            self.steps += steps

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        # train
        acc_train_loss = 0.
        for batch_idx, batch in enumerate(self.dl, start=1):
            train_loss = None
            grad_norms = None
            valid_loss = None
            valid_accuracy = None
            self.transformer.train()
            steps = int(self.steps.item())
            with self.accelerator.accumulate(self.train_wrapper):
                batch_data, batch_names = batch[:-1], batch[-1]
                print(
                    f"rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} {batch_names=}"
                )
                data_kwargs = dict(zip(self.ds_fields, batch_data))
                loss, _, _ = self.train_wrapper(**data_kwargs, return_loss=True)
                acc_train_loss += loss.item()
                # update
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    if exists(self.max_grad_norm):
                        grad_norms = self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), self.max_grad_norm
                        ).item()
                    acc_train_loss_t = torch.tensor(acc_train_loss).to(
                        self.accelerator.device
                    )
                    gathered_acc_train_loss_t = self.accelerator.gather(
                        acc_train_loss_t
                    )
                    acc_train_loss = gathered_acc_train_loss_t.mean().item()
                    self.print(
                        f"{datetime.datetime.now()}: {steps=} {batch_idx=} {acc_train_loss=} {grad_norms=}"
                    )
                    train_loss = acc_train_loss
                    acc_train_loss = 0.0
                self.optim.step()
                if exists(self.scheduler):
                    self.scheduler.step()
                self.optim.zero_grad()

            # sample results every so often
            if not (steps % self.save_results_every):
                self.transformer.eval()
                data_kwargs = None
                for batch in self.valid_dl:
                    batch_data, batch_names = batch[:-1], batch[-1]
                    data_kwargs = dict(zip(self.ds_fields, batch_data))
                    break
                with torch.no_grad():
                    valid_loss, all_logits, all_labels = self.train_wrapper(
                        **data_kwargs, return_loss=True
                    )
                    # reduce
                    pred_tokens = self.accelerator.gather(
                        all_logits[-1].argmax(1).contiguous()
                    )
                    gt_tokens = self.accelerator.gather(all_labels[-1].contiguous())
                    pred_tokens = pred_tokens.detach().cpu().long()
                    gt_tokens = gt_tokens.detach().cpu().long()
                    valid_accuracy = (pred_tokens == gt_tokens).float().mean().item()
                    valid_loss = self.accelerator.reduce(valid_loss, "mean").item()
                    self.print(
                        f"{datetime.datetime.now()}: {steps=} {valid_loss=} {valid_accuracy=}"
                    )
                if self.is_main and self.save_predicted_tokens:
                    # interleave pred_tokens and gt_tokens and save to a text file

                    assert exists(self.tokens_folder)

                    interleave = torch.empty(
                        (
                            pred_tokens.shape[0] + gt_tokens.shape[0],
                            pred_tokens.shape[1],
                        ),
                        dtype=pred_tokens.dtype,
                    )
                    interleave[0::2] = pred_tokens
                    interleave[1::2] = gt_tokens

                    np.savetxt(
                        str(self.tokens_folder / f"{self.stage}.tokens.{steps}.txt"),
                        interleave,
                        fmt="%-6s",
                        header="predicted and ground truth tokens from the validation set. row 0%2 is predicted, 1%2 is ground truth\n ",
                    )

                if (
                    self.is_main
                    and self.save_reconstructed_wave
                    and (self.stage == "coarse" or self.stage == "fine")
                ):
                    # For coarse and fine stages, reconstruct teacher-forced wave from logits

                    assert exists(self.neural_codec)
                    assert exists(self.waves_folder)

                    pred_tokens = all_logits[-1].detach().argmax(1)[:, :-1]
                    pred_tokens[pred_tokens == self.transformer.eos_ids[-1]] = 0

                    num_quantizers = self.transformer.token_sequences[-1].num_quantizers
                    pred_tokens = rearrange(
                        pred_tokens, "b (n q) -> b n q", q=num_quantizers
                    )

                    if self.stage == "fine":
                        coarse_tokens = all_labels[-2][:, :-1]
                        coarse_quantizers = self.transformer.token_sequences[
                            -2
                        ].num_quantizers
                        coarse_tokens = rearrange(
                            coarse_tokens, "b (n q) -> b n q", q=coarse_quantizers
                        )
                        pred_tokens = torch.cat((coarse_tokens, pred_tokens), dim=-1)

                    waves = self.neural_codec.decode_from_codebook_indices(pred_tokens)
                    waves = waves.cpu()

                    file_paths = []

                    max_files_to_save = 4
                    for i, wave in enumerate(waves):
                        if i < max_files_to_save:
                            file_path = str(
                                self.waves_folder
                                / f"{self.stage}.reconstructed_wave_{i}.{steps}.wav"
                            )
                            torchaudio.save(
                                file_path, wave, self.neural_codec.sample_rate
                            )
                            file_paths.append(file_path)
                        else:
                            break

                    if "wandb" in self.log_with and exists(wandb):
                        audios = [
                            wandb.Audio(
                                file_path,
                                caption=f"reconstructed wave at {steps} steps",
                            )
                            for file_path in file_paths
                        ]
                        wandb.log({"reconstructed_wave": audios})
            # log
            self.accelerator.log(
                {
                    "train_loss": train_loss,
                    "grad_norms": grad_norms,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                },
                step=steps,
            )

            # save model every so often
            if self.is_main and not (steps % self.save_model_every):
                model_path = str(
                    self.results_folder / f"{self.stage}.transformer.{steps}.pt"
                )
                optim_path = str(
                    self.results_folder / f"{self.stage}.optimizer.{steps}.pt"
                )
                scheduler_path = str(
                    self.results_folder / f"{self.stage}.scheduler.{steps}.pt"
                )
                self.save(model_path, optim_path, scheduler_path)
                # save audio conditioner (clap) rvq checkpoint
                if exists(self.audio_conditioner) and self.audio_conditioner.learn_rvq:
                    rvq_state_dict = self.audio_conditioner.rq.state_dict()
                    rvq_path = str(
                        self.results_folder / f"{self.stage}.conditioner_rvq.{steps}.pt"
                    )
                    torch.save(rvq_state_dict, rvq_path)
                self.print(
                    f"{datetime.datetime.now()}: {steps=} done saving model to {model_path=} {optim_path=} {scheduler_path=}"
                )
            # update steps
            if self.is_main:
                self.steps += 1
            accelerate.utils.broadcast(self.steps)
            print(
                f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} update steps={self.steps.item()}"
            )
            self.accelerator.wait_for_everyone()
        self.print(
            f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} done epoch steps={self.steps.item()}"
        )
        self.accelerator.wait_for_everyone()
        return None

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print(f'{datetime.datetime.now()} training complete')


@beartype_jit
class ClapRVQTrainer(nn.Module):
    """
    Learn the residual vector quantizer to turn CLAP embeddings into discrete tokens.
    """

    def __init__(
        self,
        *,
        num_train_steps,
        batch_size,
        accumulate_batches: Optional[int] = None,
        audio_conditioner: Optional[ClapQuantized] = None,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        filelist_path: str = None,
        blacklist_path: str = None,
        wd=0.0,
        max_grad_norm=0.5,
        data_max_length_seconds: Union[float, int] = 10,
        valid_frac=0.05,
        random_split_seed=42,
        save_results_every=100,
        save_model_every=1000,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index

        self.log_with = accelerate_kwargs['log_with'] if 'log_with' in accelerate_kwargs else None

        self.audio_conditioner = audio_conditioner
        self.ds = dataset
        self.num_train_steps = num_train_steps
        self.accumulate_batches = accumulate_batches
        self.register_buffer('steps', torch.tensor(0))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'
            shm_filelist_name = get_distributed_shm_filelist_name(
                self.accelerator.num_processes,
                self.rank,
                self.local_rank,
                filelist_path,
                blacklist_path,
            )
            self.ds = SoundDataset(
                shm_filelist_name,
                folder,
                # filelist_path=filelist_path,
                # blacklist_path=blacklist_path,
                max_length_seconds=data_max_length_seconds,
                target_sample_hz=audio_conditioner.sample_rate,
                seq_len_multiple_of=None,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds, [train_size, valid_size], generator=torch.Generator().manual_seed(random_split_seed))
            self.print(
                f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader
        dataloader_kwargs = {
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "num_workers": 16,
            # "multiprocessing_context": "forkserver",
        }
        self.dl = get_dataloader(self.ds, batch_size=batch_size, **dataloader_kwargs)

        self.valid_dl = get_dataloader(
            self.valid_ds, batch_size=batch_size, **dataloader_kwargs
        )

        (self.audio_conditioner, self.dl, self.valid_dl) = self.accelerator.prepare(
            self.audio_conditioner, self.dl, self.valid_dl
        )
        print(f"dataloader len {len(self.dl)}")

        # dataloader iterators

        # self.dl_iter = iter(self.dl)
        # self.valid_dl_iter = iter(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        # if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
        #     rmtree(str(self.results_folder))

        # self.results_folder.mkdir(parents=True, exist_ok=True)
        hps = {"num_train_steps": num_train_steps, "batch_size": batch_size, "accumulate_batches": accumulate_batches}

        if 'tensorboard' in self.log_with:
            self.accelerator.init_trackers(f"clap_rvq_{int(time.time() * 1000)}", config=hps)
        else:
            self.accelerator.init_trackers(f"clap_rvq", config=hps)

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

        # ThomasLee
        self.to(self.accelerator.device)
        self.accelerator.wait_for_everyone()

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        # ThomasLee
        embed_list = list()
        acc_train_loss = 0.0
        accumulate_batches = default(self.accumulate_batches, 1)
        print(f"rank={self.rank} local_rank={self.local_rank} {accumulate_batches=}")
        for batch_idx, batch_data in enumerate(self.dl, start=1):
            raw_wave_for_clap = batch_data[0]
            loss = None
            valid_loss = None
            steps = int(self.steps.item())
            self.audio_conditioner.learn_rvq = True
            embed = self.audio_conditioner.forward(
                audio_input=raw_wave_for_clap, return_embedding=True
            )
            embed_list.append(embed)
            if batch_idx % accumulate_batches:
                print(
                    f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} {steps=} skip {batch_idx=}"
                )
                continue
            self.accelerator.wait_for_everyone()
            # train rvq
            embeds = torch.cat(embed_list, dim=0)
            gathered_embeds = self.accelerator.gather(embeds)
            print(
                f"rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} embeds={embeds.size()} gathered_embeds={gathered_embeds.size()}"
            )
            loss = self.audio_conditioner.quantize(
                gathered_embeds, return_rvq_loss=True
            )
            # gather loss
            loss_t = torch.tensor(
                loss, device=self.accelerator.device, requires_grad=False
            )
            gathered_loss_t = self.accelerator.gather(loss_t)
            print(
                f"rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} {gathered_loss_t=}"
            )
            gathered_loss_mean_t = gathered_loss_t.mean()
            print(
                f"rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} {gathered_loss_mean_t=}"
            )
            acc_train_loss = gathered_loss_mean_t.item()
            self.print(
                f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} training loss: {acc_train_loss}"
            )
            # validation
            if not (steps % self.save_results_every):
                valid_input = None
                for batch_data in self.valid_dl:
                    valid_input = batch_data[0]
                    break
                with torch.no_grad():
                    self.audio_conditioner.learn_rvq = False
                    loss = self.audio_conditioner.forward(
                        audio_input=valid_input, return_rvq_loss=True
                    )
                # gather
                valid_loss_t = torch.tensor(
                    loss, device=self.accelerator.device, requires_grad=False
                )
                gathered_valid_loss_t = self.accelerator.gather(valid_loss_t)
                print(
                    f"rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} {gathered_valid_loss_t=}"
                )
                valid_loss = gathered_valid_loss_t.mean().item()
                self.print(
                    f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} {steps=} validation loss: {valid_loss}"
                )
            # log
            self.accelerator.log({
                "train_loss": loss,
                "valid_loss": valid_loss
            }, step=steps)

            # save model every so often
            if self.is_main and not (steps % self.save_model_every):
                # save audio conditioner (clap) rvq checkpoint
                rvq_state_dict = self.accelerator.unwrap_model(
                    self.audio_conditioner
                ).rq.state_dict()
                save_path = str(self.results_folder / f"clap.rvq.{steps}.pt")
                torch.save(rvq_state_dict, save_path)
                self.print(
                    f"{datetime.datetime.now()}: {steps=} saving model to {save_path}"
                )

            # reset embeds and update
            embed_list.clear()
            if self.is_main:
                self.steps += 1
            accelerate.utils.broadcast(self.steps)
            print(
                f"rank={self.rank} local_rank={self.local_rank} update steps={self.steps.item()}"
            )
            self.accelerator.wait_for_everyone()
        self.print(
            f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} done epoch steps={self.steps.item()}"
        )
        self.accelerator.wait_for_everyone()
        return None

    def train(self, log_fn=noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')


@beartype_jit
class HfHubertKmeansTrainer(nn.Module):
    """
    Trainer for kmeans part of HfHubertWithKmeans. Consists of two parts: 1) extracting Hubert features and 2) training kmeans model on these features.
    """

    def __init__(
        self,
        *,
        feature_extraction_num_steps: int,
        feature_extraction_batch_size: int,
        hubert_kmeans: HfHubertWithKmeans,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]]=None,
        ignore_load_errors: bool=True,
        folder=None,
        filelist_path: str = None,
        blacklist_path: str = None,
        data_max_length_seconds: Union[float, int] = 1,
        results_folder='./results',
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        self.ds = dataset
        self.feature_extraction_num_steps = feature_extraction_num_steps
        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.hubert_kmeans = hubert_kmeans
        self.register_buffer('steps', torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'
            shm_filelist_name = get_distributed_shm_filelist_name(
                self.accelerator.num_processes,
                self.rank,
                self.local_rank,
                filelist_path,
                blacklist_path,
            )
            self.ds = SoundDataset(
                shm_filelist_name,
                folder,
                # filelist_path=filelist_path,
                # blacklist_path=blacklist_path,
                max_length_seconds=data_max_length_seconds,
                normalize=True,
                target_sample_hz=hubert_kmeans.target_sample_hz,
                seq_len_multiple_of=hubert_kmeans.seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors
            )
        self.print(
            f'training on {feature_extraction_num_steps * feature_extraction_batch_size} out of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size=feature_extraction_batch_size, shuffle=True)

        (
            self.hubert_kmeans,
            self.dl
        ) = self.accelerator.prepare(
            self.hubert_kmeans,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)

        self.results_folder = Path(results_folder)

        # if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
        #     rmtree(str(self.results_folder))

        # self.results_folder.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()

        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def extract_hubert_features(self):

        raw_wave = next(self.dl_iter)[0]

        embed = self.hubert_kmeans.forward(wav_input=raw_wave.to(self.device), return_embed=True)

        # get features
        embed = rearrange(embed, 'b t f -> (b t) f')
        embed = self.accelerator.gather_for_metrics(embed)
        embed = embed.detach().cpu().numpy()

        return embed

    def train(self, log_fn=noop, seed=0):

        self.print('step 1: extracting features. must wait for this to complete before training kmeans.')
        features = []
        num_steps = math.ceil(self.feature_extraction_num_steps / self.accelerator.num_processes)
        while self.steps < num_steps:
            self.print(f'{int(self.steps.item())} / {num_steps} steps')
            features.append(self.extract_hubert_features())
            self.steps += 1

        features = np.concatenate(features, axis=0)

        features = features[~np.any(np.isnan(features), axis=-1)]

        self.print('step 2: training kmeans')
        if self.is_main:
            learn_kmeans(
                features,
                seed,
                str(self.results_folder / 'kmeans.joblib'),
                n_clusters=self.accelerator.unwrap_model(self.hubert_kmeans).codebook_size)

        self.print('training complete')


@beartype_jit
class HfHubertBatchKmeansTrainer(nn.Module):
    def __init__(
        self,
        *,
        feature_extraction_num_steps: int,
        feature_extraction_batch_size: int,
        hubert_kmeans: HfHubertWithBatchKmeans,
        dataset: Optional[Dataset] = None,
        ignore_files: Optional[List[str]] = None,
        ignore_load_errors: bool = True,
        folder=None,
        filelist_path: str = None,
        blacklist_path: str = None,
        valid_frac=0.05,
        random_split_seed=42,
        data_max_length_seconds: Union[float, int] = 1,
        results_folder="./results",
        accelerate_kwargs: dict = {},
        config_paths: Optional[List[str]] = None,
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index

        self.log_with = (
            accelerate_kwargs["log_with"] if "log_with" in accelerate_kwargs else None
        )

        self.ds = dataset
        self.feature_extraction_num_steps = feature_extraction_num_steps
        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.hubert_kmeans = hubert_kmeans
        self.register_buffer("steps", torch.Tensor([0]))

        if not exists(self.ds):
            assert exists(
                folder
            ), "folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training"
            shm_filelist_name = get_distributed_shm_filelist_name(
                self.accelerator.num_processes,
                self.rank,
                self.local_rank,
                filelist_path,
                blacklist_path,
            )
            self.ds = SoundDataset(
                shm_filelist_name,
                folder,
                # filelist_path=filelist_path,
                # blacklist_path=blacklist_path,
                max_length_seconds=data_max_length_seconds,
                normalize=True,
                target_sample_hz=hubert_kmeans.target_sample_hz,
                seq_len_multiple_of=hubert_kmeans.seq_len_multiple_of,
                ignore_files=default(ignore_files, []),
                ignore_load_errors=ignore_load_errors,
            )
        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(
                self.ds,
                [train_size, valid_size],
                generator=torch.Generator().manual_seed(random_split_seed),
            )
            self.print(
                f"training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples"
            )
        else:
            self.valid_ds = self.ds
            self.print(
                f"training with shared training and valid dataset of {len(self.ds)} samples"
            )

        # dataloader
        dataloader_kwargs = {
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": True,
            "num_workers": 16,
            # "multiprocessing_context": "forkserver",
        }
        self.dl = get_dataloader(
            self.ds, batch_size=feature_extraction_batch_size, **dataloader_kwargs
        )
        self.valid_dl = get_dataloader(
            self.valid_ds, batch_size=feature_extraction_batch_size, **dataloader_kwargs
        )
        # only wrap kmeans_model for custom functions
        self.kmeans_model, self.dl, self.valid_dl = self.accelerator.prepare(
            self.hubert_kmeans.kmeans, self.dl, self.valid_dl
        )
        print(f"dataloader size per process {len(self.dl)}")
        # FIXME
        self.save_model_every = 100
        self.save_results_every = 100
        self.num_train_steps = 200001
        hps = {
            "num_train_steps": self.num_train_steps,
            "batch_size": feature_extraction_batch_size,
        }

        if "tensorboard" in self.log_with:
            self.accelerator.init_trackers(
                f"hubert_batch_kmeans_{int(time.time() * 1000)}", config=hps
            )
        else:
            self.accelerator.init_trackers(f"hubert_batch_kmeans", config=hps)

        self.results_folder = Path(results_folder)
        if self.is_main and exists(config_paths):
            configs_folder = self.results_folder / "configs"
            configs_folder.mkdir(parents=True, exist_ok=True)
            for config_path in config_paths:
                copy_file_to_folder(config_path, configs_folder)

        # ThomasLee
        self.to(self.accelerator.device)
        self.accelerator.wait_for_everyone()

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (
            self.accelerator.distributed_type == DistributedType.NO
            and self.accelerator.num_processes == 1
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def extract_features(self, raw_wave):
        embed = self.hubert_kmeans.get_embed(wav_input=raw_wave)
        embed = rearrange(embed, "b t f -> (b t) f")
        return embed

    def train(self, log_fn=noop, seed=0):
        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print("training complete")

    def train_step(self):
        # ThomasLee
        acc_dist = 0.0
        for batch_idx, batch_data in enumerate(self.dl, start=1):
            steps = int(self.steps.item())
            raw_wave = batch_data[0]
            embeds = self.extract_features(raw_wave)
            dist = self.kmeans_model(embeds)
            gathered_dist = self.accelerator.gather(dist)
            acc_dist = gathered_dist.mean().item()
            # save
            if not steps % self.save_model_every:
                # sync and update codebook
                self.hubert_kmeans.kmeans.epoch()
                if self.is_main:
                    state_dict = self.accelerator.unwrap_model(
                        self.kmeans_model
                    ).state_dict()
                    save_path = str(
                        self.results_folder / f"hubert-batch-kmeans.{steps}.pt"
                    )
                    torch.save(state_dict, save_path)
                    self.print(
                        f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} {steps=} saving model to {save_path}"
                    )
            # eval
            valid_score = None
            valid_dist = None
            if not steps % self.save_results_every:
                raw_wave_eval = None
                for eval_idx, eval_data in enumerate(self.valid_dl, start=1):
                    raw_wave_eval = eval_data[0]
                    break
                embeds = self.extract_features(raw_wave_eval)
                score, dist = self.accelerator.unwrap_model(
                    self.kmeans_model
                ).silhouette_score_and_dist(embeds)
                score_t = torch.tensor(score).to(self.accelerator.device)
                dist_t = torch.tensor(dist).to(self.accelerator.device)
                gathered_score = self.accelerator.gather(score_t)
                gathered_dist = self.accelerator.gather(dist_t)
                valid_score = gathered_score.mean().item()
                valid_dist = gathered_dist.mean().item()
                self.print(
                    f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} {steps=} {batch_idx=} {eval_idx=} {valid_score=} {valid_dist=}"
                )
            # update
            if self.is_main:
                self.steps += 1
            accelerate.utils.broadcast(self.steps)
            # log
            self.accelerator.log(
                {
                    "acc_dist": acc_dist,
                    "valid_score": valid_score,
                    "valid_dist": valid_dist,
                },
                step=steps,
            )
            self.accelerator.wait_for_everyone()
        self.print(
            f"{datetime.datetime.now()}: rank={self.rank} local_rank={self.local_rank} done epoch steps={self.steps.item()}"
        )
        self.accelerator.wait_for_everyone()
        return None
