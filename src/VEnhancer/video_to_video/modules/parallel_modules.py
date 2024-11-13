import logging
from typing import Any

import torch
from torch import Tensor, nn
import torch.distributed as dist

from ..context_parallel import get_context_parallel_group, get_context_parallel_rank, get_context_parallel_world_size

logger = logging.getLogger(__name__)


class ContextParallelConv3d(nn.Conv3d):
    def __init__(self, *args, padding=None, **kwargs):
        assert padding is not None
        self.f_padding = padding[0]
        super().__init__(*args, padding=padding, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        pad_num = self.f_padding

        cp_rank, cp_size, cp_group = (
            get_context_parallel_rank(),
            get_context_parallel_world_size(),
            get_context_parallel_group(),
        )
        # first, pad from the temporal 'former' side
        if cp_rank != cp_size - 1:
            dist.isend(x[:, :, -pad_num:].contiguous(), cp_rank + 1, group=cp_group, tag=100)
        if cp_rank != 0:
            former_padding = torch.empty_like(x[:, :, -pad_num:]).contiguous()
            dist.recv(former_padding, cp_rank - 1, cp_group, tag=100)
        else:
            former_padding = None

        # then, pad from the temporal 'later' side
        if cp_rank != 0:
            dist.isend(x[:, :, :pad_num].contiguous(), cp_rank - 1, group=cp_group, tag=101)
        if cp_rank != cp_size - 1:
            later_padding = torch.empty_like(x[:, :, :pad_num]).contiguous()
            dist.recv(later_padding, cp_rank + 1, cp_group, tag=101)
        else:
            later_padding = None

        if cp_rank == 0:
            x = torch.cat([x, later_padding], dim=2)
            x = super().forward(x)[:, :, :-pad_num].contiguous()
        elif cp_rank == cp_size - 1:
            x = torch.cat([former_padding, x], dim=2)
            x = super().forward(x)[:, :, pad_num:].contiguous()
        else:
            x = torch.cat([former_padding, x, later_padding], dim=2)
            x = super().forward(x)[:, :, pad_num:-pad_num].contiguous()

        logger.debug(f"paraconv out shape: {x.shape}")
        return x


class ContextParallelGroupNorm(nn.GroupNorm):
    def forward(self, x: Tensor) -> Tensor:

        cp_rank, cp_size, cp_group = (
            get_context_parallel_rank(),
            get_context_parallel_world_size(),
            get_context_parallel_group(),
        )

        b, c, f_shard, h, w = x.shape
        # f_shard may be different on the last rank and on the others because f_total % cp_world_size != 0
        f_total = torch.tensor(f_shard, device="cuda")
        dist.all_reduce(f_total, group=cp_group)
        f_total = f_total.item()  #
        x = x.view(b, self.num_groups, c // self.num_groups, f_shard, h, w)
        mean = torch.mean(x, dim=[2, 3, 4, 5], keepdim=True) * f_shard / f_total

        dist.all_reduce(mean, group=cp_group)

        var = (x - mean).square().mean(dim=[2, 3, 4, 5], keepdim=True) * f_shard / f_total
        dist.all_reduce(var, group=cp_group)

        x = (x - mean) / torch.sqrt(var + self.eps)

        x = x.view(b, c, f_shard, h, w) * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


def all_to_all(input_: torch.Tensor, gather_dim: int, scatter_dim: int, group: Any) -> torch.Tensor:
    assert gather_dim != scatter_dim
    assert 0 <= gather_dim < input_.ndim
    assert 0 <= scatter_dim < input_.ndim
    world_size = get_context_parallel_world_size()

    logger.debug(f"all_to_all_input: {input_.shape}")

    real_gather_channels = torch.tensor(input_.size(gather_dim), device="cuda")
    dist.all_reduce(real_gather_channels, group=group)
    if real_gather_channels % world_size != 0:
        pad_gather_channels = (real_gather_channels // world_size + 1) * world_size - real_gather_channels
    else:
        pad_gather_channels = 0

    if input_.size(scatter_dim) % world_size != 0:
        pad_scatter_channels = (input_.size(scatter_dim) // world_size + 1) * world_size - input_.size(scatter_dim)
    else:
        pad_scatter_channels = 0

    if world_size == 1:
        return input_

    scatter_pad_shape = list(input_.shape)
    scatter_pad_shape[scatter_dim] = pad_scatter_channels
    input_ = torch.cat([input_, torch.zeros(size=scatter_pad_shape).to(input_)], dim=scatter_dim)

    if pad_gather_channels != 0 and get_context_parallel_rank() == get_context_parallel_world_size() - 1:
        gather_pad_shape = list(input_.shape)
        gather_pad_shape[gather_dim] = pad_gather_channels
        input_ = torch.cat([input_, torch.zeros(size=gather_pad_shape).to(input_)], dim=gather_dim)

    inputs = [x.contiguous() for x in input_.chunk(world_size, dim=scatter_dim)]
    outputs = [torch.empty_like(x) for x in inputs]

    logger.debug(
        f"rank {dist.get_rank()}({get_context_parallel_rank()})"
        f"len_inputs{len(inputs)}, shape{[_.shape for _ in inputs]}"
    )
    dist.all_to_all(outputs, inputs, group=group)

    outputs = torch.cat(outputs, dim=gather_dim)
    outputs = torch.split(outputs, real_gather_channels, dim=gather_dim)[0]
    if pad_scatter_channels != 0 and get_context_parallel_rank() == get_context_parallel_world_size() - 1:
        outputs = torch.split(outputs, outputs.size(scatter_dim) - pad_scatter_channels, dim=scatter_dim)[0]

    logger.debug(f"all_to_all_output: {outputs.shape}")

    return outputs
