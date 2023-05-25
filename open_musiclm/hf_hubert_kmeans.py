from pathlib import Path

import torch
from torch import nn
import numpy as np
from einops import rearrange, pack, unpack
from beartype.typing import Optional

from torchaudio.functional import resample
from .utils import exists, curtail_to_multiple, zero_mean_unit_var_norm
from transformers import HubertModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import joblib
import logging
logging.root.setLevel(logging.ERROR)


class HfHubertWithKmeans(nn.Module):
    """
    Hugging Face HubertModel + a k-means layer on top. Pretrained checkpoint for music: https://huggingface.co/m-a-p/MERT-v0
    Note: MERT-v0 outputs features at 50Hz while Wav2Vec-BERT (used in the paper) outputs at 25 Hz.
    """

    def __init__(
        self,
        *,
        hubert: HubertModel,
        kmeans: Optional[MiniBatchKMeans] = None,
        embed_layer: int=7,
        target_sample_hz=16000,
        seq_len_multiple_of=int(16000 / 50),
        normalize_embeds=True,
        codebook_size: int=1024,
        output_hz: int=50
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.output_hz = output_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.codebook_size = kmeans.n_clusters if exists(kmeans) else None

        self.codebook_size = codebook_size
        if exists(kmeans):
            assert self.codebook_size == kmeans.n_clusters, "codebook_size must match kmeans.n_clusters"

        self.normalize_embeds = normalize_embeds

        self.embed_layer = embed_layer

        self.hubert = hubert
        self.kmeans = kmeans

    @torch.no_grad()
    def forward(
        self,
        wav_input: torch.Tensor,
        flatten=True,
        return_embed=False,
        input_sample_hz=None
    ):
        assert return_embed or exists(self.kmeans), "kmeans model must be provided if return_embed==False"

        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        hubert_args = {
            'input_values': wav_input,
            'attention_mask': torch.ones_like(wav_input, device=device), # TODO: handle padding
        }

        outputs = self.hubert(**hubert_args, output_hidden_states = True)
        embed = outputs.hidden_states[self.embed_layer]

        if self.normalize_embeds:
            embed = zero_mean_unit_var_norm(embed)

        if return_embed:
            return embed

        embed, packed_shape = pack([embed], '* d')
        codebook_indices = self.kmeans.predict(embed.detach().cpu().numpy())
        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()

        if flatten:
            return codebook_indices

        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        return codebook_indices


def get_kmeans_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def learn_kmeans(
    feat,
    seed,
    km_path='./results/kmeans.joblib',
    n_clusters=1024,
    init="k-means++",
    max_iter=100,
    batch_size=10000,
    tol=0.0,
    n_init=20,
    reassignment_ratio=0.0,
    max_no_improvement=100,
):
    np.random.seed(seed)
    km_model = get_kmeans_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    print("total intertia: %.5f", inertia)
    print("finished successfully")


def get_hubert_kmeans(model_name: str="m-a-p/MERT-v0", kmeans_path: Optional[str]='./checkpoints/kmeans.joblib', **kwargs):
    wav2vec = HubertModel.from_pretrained(model_name, resume_download=True)
    kmeans = joblib.load(kmeans_path) if exists(kmeans_path) else None

    return HfHubertWithKmeans(hubert=wav2vec, kmeans=kmeans, **kwargs)


import torch.distributed as dist
from einops import repeat
import os


class BatchKmeans(nn.Module):
    def __init__(self, k: int, dim: int, ddp=False):
        super().__init__()
        self.k = k
        self.dim = dim
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("bins", torch.zeros(k, dim))
        self.register_buffer("nums", torch.zeros(k, 1))
        self.register_buffer("codebook", torch.randn(k, dim))
        self.all_reduce = dist.all_reduce if ddp else lambda *args, **kwargs: None
        self.rank = None
        self.local_rank = None

    def __init_clusters(self, x):
        from sklearn.cluster import kmeans_plusplus

        if self.is_initialized:
            return

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        # gather x
        print(f"rank={self.rank} local_rank={self.local_rank} x_size {x.size()}")
        if dist.is_initialized():
            x_list = [x.clone() for _ in range(dist.get_world_size())]
            dist.all_gather(x_list, x)
            x = torch.cat(x_list, dim=0)
            print(
                f"rank={self.rank} local_rank={self.local_rank} gathered_x_size {x.size()}"
            )
        cluster_centers_t = torch.zeros_like(self.codebook).to(x.device)
        if self.rank is None or self.rank == 0:
            x_in = x.cpu().numpy()
            # init with kmeans
            print(
                f"rank={self.rank} local_rank={self.local_rank} init cluster centers using kmeans++ ..."
            )
            cluster_centers, indices = kmeans_plusplus(x_in, n_clusters=self.k)
            print(f"rank={self.rank} local_rank={self.local_rank} done")
            cluster_centers_t = torch.from_numpy(cluster_centers).to(x.device)
        # broadcast
        if dist.is_initialized():
            dist.broadcast(cluster_centers_t, 0)
        # assign
        self.codebook.data.copy_(cluster_centers_t)
        self.is_initialized.fill_(True)

    def forward(self, x):
        self.__init_clusters(x)
        (batch_size, dim), device = x.shape, x.device
        dists = torch.cdist(x, self.codebook, p=2)
        buckets = torch.argmin(dists, dim=-1, keepdim=True)
        ones = torch.ones(buckets.size()).to(device)
        repeated_buckets = repeat(buckets, "b 1->b d", d=dim)
        self.bins.scatter_add_(dim=0, index=repeated_buckets, src=x)
        self.nums.scatter_add_(dim=0, index=buckets, src=ones)
        return dists.mean()

    @torch.no_grad()
    def predict(self, x):
        dists = torch.cdist(x, self.codebook, p=2)
        buckets = torch.argmin(dists, dim=-1)
        return buckets

    @torch.no_grad()
    def silhouette_score_and_dist(self, x):
        dists = torch.cdist(x, self.codebook, p=2)
        buckets = torch.argmin(dists, dim=-1)
        score = silhouette_score(
            dists.detach().cpu().numpy(), buckets.detach().cpu().numpy()
        )
        return score, dists.mean().item()

    def epoch(self):
        self.all_reduce(self.bins)
        self.all_reduce(self.nums)
        update_indices = (self.nums > 0).squeeze()
        self.codebook[update_indices] = (
            self.bins[update_indices] / self.nums[update_indices]
        )
        self.bins.zero_()
        self.nums.zero_()


class HfHubertWithBatchKmeans(nn.Module):
    def __init__(
        self,
        *,
        hubert: HubertModel,
        kmeans: Optional[BatchKmeans] = None,
        embed_layer: int = 7,
        target_sample_hz=16000,
        seq_len_multiple_of=int(16000 / 50),
        normalize_embeds=True,
        output_hz: int = 50,
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.output_hz = output_hz
        self.seq_len_multiple_of = seq_len_multiple_of
        self.normalize_embeds = normalize_embeds
        self.embed_layer = embed_layer
        self.hubert = hubert
        self.kmeans = kmeans
        self.codebook_size = self.kmeans.k

    def calc_kmeans(self, embeds):
        return self.kmeans.forward(embeds)

    @torch.no_grad()
    def get_embed(self, wav_input: torch.Tensor, input_sample_hz=None):
        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)
        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)
        hubert_args = {
            "input_values": wav_input,
            # TODO: handle padding
            "attention_mask": torch.ones_like(wav_input).to(wav_input.device),
        }
        outputs = self.hubert(**hubert_args, output_hidden_states=True)
        embed = outputs.hidden_states[self.embed_layer]
        if self.normalize_embeds:
            embed = zero_mean_unit_var_norm(embed)
        return embed

    def forward(
        self,
        wav_input: torch.Tensor,
        flatten=True,
        return_embed=False,
        input_sample_hz=None,
    ):
        assert return_embed or exists(
            self.kmeans
        ), "kmeans model must be provided if return_embed==False"

        embed = self.get_embed(wav_input, input_sample_hz=input_sample_hz)
        if return_embed:
            return embed

        embed, packed_shape = pack([embed], "* d")
        codebook_indices = self.kmeans.predict(embed)

        if flatten:
            return codebook_indices

        (codebook_indices,) = unpack(codebook_indices, packed_shape, "*")
        return codebook_indices


def get_hubert_batch_kmeans(
    model_name: str = "m-a-p/MERT-v0",
    kmeans_path: Optional[str] = None,
    codebook_size: int = 1024,
    emb_dim: int = 768,
    **kwargs,
):
    wav2vec = HubertModel.from_pretrained(model_name, resume_download=True)
    kmeans = BatchKmeans(k=codebook_size, dim=emb_dim)
    if kmeans_path:
        print(f"loading kmeans {kmeans_path}")
        state_dict = torch.load(kmeans_path, map_location="cpu")
        kmeans.load_state_dict(state_dict)
    return HfHubertWithBatchKmeans(hubert=wav2vec, kmeans=kmeans, **kwargs)
