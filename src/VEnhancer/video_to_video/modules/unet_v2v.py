# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import abstractmethod
import logging
import math
import os
from typing import Any, Optional

from einops import rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import xformers
import xformers.ops

logger = logging.getLogger(__name__)

USE_TEMPORAL_TRANSFORMER = True


class DropPath(nn.Module):
    r"""DropPath but without rescaling and supports optional all-zero and/or all-keep."""

    def __init__(self, p):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, *args, zero=None, keep=None):
        if not self.training:
            return args[0] if len(args) == 1 else args

        # params
        x = args[0]
        b = x.size(0)
        n = (torch.rand(b) < self.p).sum()

        # non-zero and non-keep mask
        mask = x.new_ones(b, dtype=torch.bool)
        if keep is not None:
            mask[keep] = False
        if zero is not None:
            mask[zero] = False

        # drop-path index
        index = torch.where(mask)[0]
        index = index[torch.randperm(len(index))[:n]]
        if zero is not None:
            index = torch.cat([index, torch.where(zero)[0]], dim=0)

        # drop-path multiplier
        multiplier = x.new_ones(b)
        multiplier[index] = 0.0
        output = tuple(u * self.broadcast(multiplier, u) for u in args)
        return output[0] if len(args) == 1 else output

    def broadcast(self, src, dst):
        assert src.size(0) == dst.size(0)
        shape = (dst.size(0),) + (1,) * (dst.ndim - 1)
        return src.view(shape)


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(timesteps, torch.pow(10000, -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        mask = torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


class MemoryEfficientCrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, max_bs=16384, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.max_bs = max_bs
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)

        del x

        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of.
        if q.shape[0] > self.max_bs:
            q_list = torch.chunk(q, q.shape[0] // self.max_bs, dim=0)
            k_list = torch.chunk(k, k.shape[0] // self.max_bs, dim=0)
            v_list = torch.chunk(v, v.shape[0] // self.max_bs, dim=0)
            out_list = []
            for q_1, k_1, v_1 in zip(q_list, k_list, v_list):
                out = xformers.ops.memory_efficient_attention(q_1, k_1, v_1, attn_bias=None, op=self.attention_op)
                out_list.append(out)
            out = torch.cat(out_list, dim=0)
            del q_list, k_list, v_list
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        del q, k, v, context
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class RelativePositionBias(nn.Module):

    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)  # noqa
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")


class BasicTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = MemoryEfficientCrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        attn_cls2 = MemoryEfficientCrossAttention

        self.attn2 = attn_cls2(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# feedforward
class GEGLU(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeedForward(nn.Module):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = x[..., 1:-1, :]
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset
            )

    def forward(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size)

    def _forward(self, x, emb, batch_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, "(b f) c h w -> b c f h w", b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, "b c f h w -> (b f) c h w")
        return h


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=(2, 1)):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Resample(nn.Module):

    def __init__(self, in_dim, out_dim, mode):
        assert mode in ["none", "upsample", "downsample"]
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x, reference=None):
        if self.mode == "upsample":
            assert reference is not None
            x = F.interpolate(x, size=reference.shape[-2:], mode="nearest")
        elif self.mode == "downsample":
            x = F.adaptive_avg_pool2d(x, output_size=tuple(u // 2 for u in x.shape[-2:]))
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, embed_dim, out_dim, use_scale_shift_norm=True, mode="none", dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.mode = mode

        # layers
        self.layer1 = nn.Sequential(nn.GroupNorm(32, in_dim), nn.SiLU(), nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.resample = Resample(in_dim, in_dim, mode)
        self.embedding = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, out_dim * 2 if use_scale_shift_norm else out_dim)
        )
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout), nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e, reference=None):
        identity = self.resample(x, reference)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x), reference))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        only_self_att=True,
        multiply_zero=False,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        self.use_adaptor = False
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)
            if self.use_adaptor:
                self.adaptor_in = nn.Linear(frames, frames)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d], checkpoint=use_checkpoint
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
            if self.use_adaptor:
                self.adaptor_out = nn.Linear(frames, frames)
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, "b c f h w -> (b h w) c f").contiguous()
            x = self.proj_in(x)
        # [16384, 16, 320]
        if self.use_linear:
            x = rearrange(x, "b c f h w -> (b h w) f c").contiguous()
            x = self.proj_in(x)
            x = rearrange(x, "bhw f c -> bhw c f").contiguous()

        if self.only_self_att:
            x = rearrange(x, "bhw c f -> bhw f c").contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, "(b hw) f c -> b hw f c", b=b).contiguous()
        else:
            x = rearrange(x, "(b hw) c f -> b hw f c", b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(context[i], "(b f) l con -> b f l con", f=self.frames).contiguous()
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j], "f l con -> (f r) l con", r=(h * w) // self.frames, f=self.frames
                    ).contiguous()
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = rearrange(x, "b hw f c -> (b hw) f c").contiguous()
            x = self.proj_out(x)
            x = rearrange(x, "(b h w) f c -> b c f h w", b=b, h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, "b hw f c -> (b hw) c f").contiguous()
            x = self.proj_out(x)
            x = rearrange(x, "(b h w) c f -> b c f h w", b=b, h=h, w=w).contiguous()

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TemporalConvBlock_v2(nn.Module):

    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(), nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class Vid2VidSDUNet(nn.Module):

    def __init__(
        self,
        in_dim=4,
        dim=320,
        y_dim=1024,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1 / 1, 1 / 2, 1 / 4],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_checkpoint=True,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        training=False,
        inpainting=True,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(Vid2VidSDUNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(nn.Linear(dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(nn.Linear(dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                raise NotImplementedError
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            raise NotImplementedError
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        self.middle_block = nn.ModuleList(
            [
                ResBlock(
                    out_dim,
                    embed_dim,
                    dropout,
                    use_scale_shift_norm=False,
                    use_image_dataset=use_image_dataset,
                ),
                SpatialTransformer(
                    out_dim,
                    out_dim // head_dim,
                    head_dim,
                    depth=1,
                    context_dim=self.context_dim,
                    disable_self_attn=False,
                    use_linear=True,
                ),
            ]
        )

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                raise NotImplementedError

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [
                        ResBlock(
                            in_dim + shortcut_dims.pop(),
                            embed_dim,
                            dropout,
                            out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            raise NotImplementedError
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)

    def forward(self, x, t, y, x_lr=None, fps=None, mask_last_frame_num=0):

        batch, c, f, h, w = x.shape
        device = x.device
        self.batch = batch

        # embeddings
        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        context = y

        # repeat f times for spatial e and context
        e = e.repeat_interleave(repeats=f, dim=0)
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, "b c f h w -> (b f) c h w")
        # encoder
        xs = []
        for ind, block in enumerate(self.input_blocks):
            x = self._forward_single(block, x, e, context)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context)

        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, e, context, reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=batch)
        return x

    def _forward_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, context)
            x = rearrange(x, "b c f h w -> (b f) c h w")
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class ControlledV2VUNet(Vid2VidSDUNet):
    def __init__(self):
        super(ControlledV2VUNet, self).__init__()
        self.VideoControlNet = VideoControlNet()

    def forward(
        self,
        x,
        t,
        y,
        hint=None,
        t_hint=None,
        s_cond=None,
        mask_cond=None,
        x_lr=None,
        fps=None,
        mask_last_frame_num=0,
    ):

        batch, c, f, h, w = x.shape
        device = x.device
        self.batch = batch

        control = self.VideoControlNet(x, t, y, hint=hint, t_hint=t_hint, mask_cond=mask_cond, s_cond=s_cond)

        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        e = e.repeat_interleave(repeats=f, dim=0)

        context = y
        context = context.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, "b c f h w -> (b f) c h w")
        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, e, context)
            xs.append(x)
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context)

        if control is not None:
            x = control.pop() + x

        # decoder
        for block in self.output_blocks:
            if control is None:
                x = torch.cat([x, xs.pop()], dim=1)
            else:
                x = torch.cat([x, xs.pop() + control.pop()], dim=1)
            x = self._forward_single(block, x, e, context, reference=xs[-1] if len(xs) > 0 else None)

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        x = rearrange(x, "(b f) c h w -> b c f h w", b=batch)
        return x

    def _forward_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, context)
            x = rearrange(x, "b c f h w -> (b f) c h w")
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class VideoControlNet(nn.Module):

    def __init__(
        self,
        in_dim=4,
        dim=320,
        y_dim=1024,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1 / 1, 1 / 2, 1 / 4],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_checkpoint=True,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        training=False,
        inpainting=True,
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(VideoControlNet, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.training = training
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(nn.Linear(dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.hint_time_zero_linear = zero_module(nn.Linear(embed_dim, embed_dim))

        # scale prompt
        self.scale_cond = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(), zero_module(nn.Linear(embed_dim, embed_dim))
        )

        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(nn.Linear(dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        # need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                raise NotImplementedError
        self.input_blocks.append(init_block)
        self.zero_convs = nn.ModuleList([self.make_zero_conv(dim)])
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            raise NotImplementedError
                in_dim = out_dim
                self.input_blocks.append(block)
                self.zero_convs.append(self.make_zero_conv(out_dim))
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
                    self.zero_convs.append(self.make_zero_conv(out_dim))

        self.middle_block = nn.ModuleList(
            [
                ResBlock(
                    out_dim,
                    embed_dim,
                    dropout,
                    use_scale_shift_norm=False,
                    use_image_dataset=use_image_dataset,
                ),
                SpatialTransformer(
                    out_dim,
                    out_dim // head_dim,
                    head_dim,
                    depth=1,
                    context_dim=self.context_dim,
                    disable_self_attn=False,
                    use_linear=True,
                ),
            ]
        )

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_dim,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                raise NotImplementedError
        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        self.middle_block_out = self.make_zero_conv(embed_dim)

        """
        add prompt
        """
        add_dim = 320
        self.add_dim = add_dim

        self.input_hint_block = zero_module(nn.Conv2d(4, add_dim, 3, padding=1))

    def make_zero_conv(self, in_channels, out_channels=None):
        out_channels = in_channels if out_channels is None else out_channels
        return TimestepEmbedSequential(zero_module(nn.Conv2d(in_channels, out_channels, 1, padding=0)))

    def forward(
        self,
        x,
        t,
        y,
        s_cond=None,
        hint=None,
        t_hint=None,
        mask_cond=None,
    ):

        batch, x_c, f, h, w = x.shape
        device = x.device
        self.batch = batch

        if hint is not None:
            add = x.new_zeros(batch, self.add_dim, f, h, w)
            hints = rearrange(hint, "b c f h w -> (b f) c h w")
            hints = self.input_hint_block(hints)
            hints = rearrange(hints, "(b f) c h w -> b c f h w", b=batch)

            if mask_cond is not None:
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = torch.where(mask_cond_per_batch >= 0)[0]
                    hint_inds = mask_cond_per_batch[inds]
                    add[i, :, inds] += hints[i, :, hint_inds]
                    # add[i,:,inds] += hints[i]
            add = rearrange(add, "b c f h w -> (b f) c h w")

        e = self.time_embed(sinusoidal_embedding(t, self.dim))
        e = e.repeat_interleave(repeats=f, dim=0)

        if t_hint is not None:
            e_cond = self.hint_time_zero_linear(self.time_embed(sinusoidal_embedding(t_hint, self.dim)))
            if mask_cond is not None:
                e = rearrange(e, "(b f) d -> b f d", b=batch)
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = torch.where(mask_cond_per_batch >= 0)[0]
                    e[i, inds] += e_cond[i]
                e = rearrange(e, "b f d -> (b f) d")
            else:
                e_cond = e_cond.repeat_interleave(repeats=f, dim=0)
                e += e_cond

        if s_cond is not None:
            e_scale = self.scale_cond(sinusoidal_embedding(s_cond, self.dim))
            if mask_cond is not None:
                e = rearrange(e, "(b f) d -> b f d", b=batch)
                for i in range(batch):
                    mask_cond_per_batch = mask_cond[i]
                    inds = torch.where(mask_cond_per_batch >= 0)[0]
                    e[i, inds] += e_scale[i]
                e = rearrange(e, "b f d -> (b f) d")
            else:
                e_scale = e_scale.repeat_interleave(repeats=f, dim=0)
                e += e_scale

        context = y.repeat_interleave(repeats=f, dim=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, "b c f h w -> (b f) c h w")

        # encoder
        xs = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if hint is not None:
                for block in module:
                    x = self._forward_single(block, x, e, context)
                    if not isinstance(block, TemporalTransformer):
                        if hint is not None:
                            x += add
                            hint = None
            else:
                x = self._forward_single(module, x, e, context)
            xs.append(zero_conv(x, e, context))

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context)
        xs.append(self.middle_block_out(x, e, context))

        return xs

    def _forward_single(self, module, x, e, context, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, context)
            x = rearrange(x, "b c f h w -> (b f) c h w")
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, Resample):
            x = module(x, reference)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e, context, reference)
        else:
            x = module(x)
        return x


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x
