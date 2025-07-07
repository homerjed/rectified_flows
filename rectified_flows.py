from __future__ import annotations

import abc
from dataclasses import dataclass, replace as dataclass_replace
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Tuple, Callable, Optional, Sequence, Optional, List, Generator, Union, Literal
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
from jax.sharding import NamedSharding, PositionalSharding, Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import equinox as eqx
import diffrax as dfx
from jaxtyping import Key, PRNGKeyArray, PyTree, Array, Float, DTypeLike, Scalar, jaxtyped
import optax
import grain.python as grain

import numpy as np
from einops import einsum, rearrange, repeat
import einx
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
import cloudpickle
from PIL import Image
from tqdm import trange

try:
    from beartype import beartype as typechecker
    typecheck = jaxtyped(typechecker=typechecker) 
except ImportError:
    typecheck = lambda x: x


# Array types for flow

TimeArray = Float[Array, ""]

XArray = Float[Array, "_ _ _"]

QArray = Optional[Float[Array, "_ _ _"]]

AArray = Optional[Float[Array, "_"]]

ScheduleFn = Callable[[TimeArray], TimeArray]

SampleFn = Callable[[PRNGKeyArray, QArray, AArray], Float[Array, "_ _ _"]]


"""
    Mixed Precision
"""


@dataclass(frozen=True)
class StaticLossScale:
    """ Scales and unscales by a fixed constant. """

    loss_scale: Scalar

    def scale(self, tree: PyTree) -> PyTree:
        return jax.tree.map(lambda x: x * self.loss_scale, tree)

    def unscale(self, tree: PyTree) -> PyTree:
        return jax.tree.map(lambda x: x / self.loss_scale, tree)

    def adjust(self, grads_finite: Array):
        del grads_finite
        return self


def _cast_floating_to(tree: PyTree, dtype: DTypeLike) -> PyTree:
    def conditional_cast(x):
        if (
            isinstance(x, jnp.ndarray) 
            and
            jnp.issubdtype(x.dtype, jnp.floating)
        ):
            x = x.astype(dtype)
        return x
    return jax.tree.map(conditional_cast, tree)


@dataclass(frozen=True)
class Policy:
    param_dtype: Optional[DTypeLike] = None
    compute_dtype: Optional[DTypeLike] = None
    output_dtype: Optional[DTypeLike] = None

    def cast_to_param(self, x: PyTree) -> PyTree:
        if self.param_dtype is not None:
            x = _cast_floating_to(x, self.param_dtype)
        return x

    def cast_to_compute(self, x: PyTree) -> PyTree:
        if self.compute_dtype is not None:
            x = _cast_floating_to(x, self.compute_dtype) 
        return x

    def cast_to_output(self, x: PyTree) -> PyTree:
        if self.output_dtype is not None:
            x = _cast_floating_to(x, self.output_dtype)
        return x 

    def with_output_dtype(self, output_dtype: DTypeLike) -> Policy:
        return dataclass_replace(self, output_dtype=output_dtype)


def precision_cast(fn: Union[Callable, eqx.Module], x: Array) -> Array:
    return fn(x.astype(jnp.float32)).astype(x.dtype)


"""
    Sharding
"""


def get_shardings() -> Tuple[
    Optional[NamedSharding], Optional[PositionalSharding]
]:
    devices = jax.local_devices()
    n_devices = len(devices)
    print(f"Running on {n_devices} local devices: \n\t{devices}")

    if n_devices > 1:
        mesh = Mesh(devices, ('x',))
        sharding = NamedSharding(mesh, P('x'))

        devices = mesh_utils.create_device_mesh((n_devices, 1))
        replicated = PositionalSharding(devices).replicate()
    else:
        sharding = replicated = None

    return sharding, replicated


def shard_batch(
    batch: Tuple[
        Float[Array, "n _ _ _"], 
        Optional[Float[Array, "n _ _ _"]], 
        Optional[Float[Array, "n _"]]
    ], 
    sharding: Optional[NamedSharding] = None
) -> Tuple[
    Float[Array, "n _ _ _"], 
    Optional[Float[Array, "n _ _ _"]], 
    Optional[Float[Array, "n _"]]
]:
    if sharding:
        batch = eqx.filter_shard(batch, sharding)
    return batch


"""
    Miscallaneous
"""


def count_parameters(model: eqx.Module) -> int:
    n_parameters = sum(
        x.size 
        for x in jax.tree.leaves(model) 
        if eqx.is_array(x)
    )
    return n_parameters


def identity(x):
    return x


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def stop_grad(a):
    return jax.lax.stop_gradient(a)


class SubScaler(eqx.Module):
    dim: Sequence[int]
    params: Tuple[Float[Array, "..."], Float[Array, "..."]]
    use_scaling: bool 

    def __init__(
        self, Y: Float[Array, "n ..."], use_scaling: bool = True
    ):
        if exists(Y):
            self.dim = Y.shape[1:]
            self.params = (Y.mean(axis=0), Y.std(axis=0))
        else: 
            self.dim = self.params = None
        self.use_scaling = use_scaling

    @property
    def mu_and_std(self):
        return stop_grad(self.params[0]), stop_grad(self.params[1])

    def forward(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        if self.use_scaling and exists(self.dim):
            mu, std = self.params
            y = (y - stop_grad(mu)) / stop_grad(std)
        return y

    def inverse(self, y: Float[Array, "..."]) -> Float[Array, "..."]:
        if self.use_scaling and exists(self.dim):
            mu, std = self.params
            y = y * stop_grad(std) + stop_grad(mu)
        return y


class Scaler(eqx.Module):
    scaler_x: SubScaler
    scaler_q: SubScaler
    scaler_a: SubScaler
    use_scaling: bool

    @typecheck
    def __init__(
        self, 
        X: Float[Array, "n _ _ _"], 
        Q: Optional[Float[Array, "n _ _ _"]] = None, 
        A: Optional[Float[Array, "n _"]] = None, 
        use_scaling: bool = True
    ):
        self.scaler_x = SubScaler(X, use_scaling)
        self.scaler_q = SubScaler(Q, use_scaling)
        self.scaler_a = SubScaler(A, use_scaling)
        self.use_scaling = use_scaling

    @typecheck
    def forward(
        self, 
        x: XArray, 
        q: QArray, 
        a: AArray
    )-> Tuple[XArray, QArray, AArray]:
        x = self.scaler_x.forward(x)
        q = self.scaler_q.forward(q)
        a = self.scaler_a.forward(a)
        return x, q, a
    
    @typecheck
    def inverse(
        self, 
        x: XArray, 
        q: QArray, 
        a: AArray
    )-> Tuple[XArray, QArray, AArray]:
        x = self.scaler_x.inverse(x)
        q = self.scaler_q.inverse(q)
        a = self.scaler_a.inverse(a)
        return x, q, a


def apply_ema(
    ema_model: eqx.Module, 
    model: eqx.Module, 
    ema_rate: float, 
    policy: Optional[Policy] = None
) -> eqx.Module:
    if exists(policy):
        model = policy.cast_to_param(model)
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    m_, _m = eqx.partition(model, eqx.is_inexact_array) # Current model params
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array) # Old EMA params
    e_ = jax.tree_util.tree_map(ema_fn, e_, m_) # New EMA params
    return eqx.combine(e_, _m)


def split_data(X, Q, A, split=0.9):
    assert len(X) == len(Q)
    Xt, Xv = jnp.split(X, [int(split * len(X))])
    Qt, Qv = jnp.split(Q, [int(split * len(Q))])
    At, Av = jnp.split(A, [int(split * len(A))])
    return (Xt, Qt, At), (Xv, Qv, Av)


"""
    UNet
"""


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def key_split_allowing_none(key, n=2, i=None):
    if key is not None:
        if i is not None:
            key = jr.fold_in(key, i)
        keys = jr.split(key, n)
    else: 
        keys = [None] * n
    return keys


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        *, 
        key: Key
    ):
        self.conv = eqx.nn.Conv2d(
            dim, 
            default(dim_out, dim), 
            kernel_size=3, 
            padding=1, 
            key=key
        )

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape
        x = jax.image.resize(
            x, shape=(c, h * 2, w * 2), method='bilinear'
        )
        x = self.conv(x)
        return x


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: Optional[int] = None, 
        *, 
        key: Key
    ):
        self.conv = eqx.nn.Conv2d(
            dim * 4, default(dim_out, dim), kernel_size=1, key=key
        )

    def __call__(self, x: Array) -> Array:
        x = rearrange(
            x, 'c (h p1) (w p2) -> (c p1 p2) h w', p1=2, p2=2
        )
        x = self.conv(x)
        return x


class RMSNorm(eqx.Module):
    scale: float # Array, learnable?
    gamma: Float[Array, "d 1 1"]

    def __init__(self, dim: int):
        self.scale = dim ** 0.5 # This is a default scaling
        self.gamma = jnp.zeros((dim, 1, 1))

    def __call__(self, x: Array) -> Array:
        # Normalise x, shift and scale 
        return (x / jnp.linalg.norm(x, ord=2, axis=0)) * (self.gamma + 1) * self.scale 


class SinusoidalPosEmb(eqx.Module):
    dim: int
    theta: int

    def __init__(self, dim: int, theta: int = 10_000):
        self.dim = dim
        self.theta = theta

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        half_dim = self.dim // 2
        emb = jnp.log(self.theta) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb) # emb = x * emb
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)])
        return emb


class RandomOrLearnedSinusoidalPosEmb(eqx.Module):
    random: bool
    weights: Float[Array, "d"]

    def __init__(self, dim: int, is_random: bool = False, *, key: Key):
        assert divisible_by(dim, 2)
        half_dim = int(dim / 2)
        self.weights = jr.normal(key, (half_dim,))
        self.random = is_random

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x) 
        weights = jax.lax.stop_gradient(self.weights) if self.random else self.weights 
        freqs = x * weights * 2. * jnp.pi 
        fouriered = jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)])
        fouriered = jnp.concatenate([x, fouriered])
        return fouriered


class Block(eqx.Module):
    proj: eqx.nn.Conv2d
    norm: RMSNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        dropout: float = 0., 
        *, 
        key: Key
    ):
        self.proj = eqx.nn.Conv2d(
            dim, dim_out, kernel_size=3, padding=1, key=key
        )
        self.norm = RMSNorm(dim_out)
        self.dropout = eqx.nn.Dropout(dropout)

    def __call__(
        self, 
        x: Array, 
        scale_shift: Tuple[Array, Array] = None, 
        key: Optional[PRNGKeyArray] = None
    ) -> Array:

        x = self.proj(x)

        x = precision_cast(self.norm, x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1.) + shift

        return self.dropout(jax.nn.silu(x), key=key)


class ResnetBlock(eqx.Module):
    mlp: eqx.nn.Linear
    block1: Block
    block2: Block
    res_conv: eqx.nn.Conv2d

    def __init__(
        self, 
        dim: int, 
        dim_out: int, 
        *, 
        time_emb_dim: int = None, 
        dropout: float = 0., 
        key: Key
    ):
        keys = jr.split(key, 4)
        self.mlp = eqx.nn.Linear(
            time_emb_dim, dim_out * 2, key=keys[0]
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, dropout=dropout, key=keys[1])
        self.block2 = Block(dim_out, dim_out, key=keys[2])
        self.res_conv = eqx.nn.Conv2d(
            dim, dim_out, kernel_size=1, key=keys[3]
        ) if dim != dim_out else eqx.nn.Identity()

    def __call__(
        self, 
        x: Array, 
        time_emb: Optional[Array] = None, 
        *, 
        key: PRNGKeyArray
    ) -> Array:
        keys = key_split_allowing_none(key)

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(jax.nn.silu(time_emb))
            time_emb = rearrange(time_emb, 'c -> c 1 1')
            scale_shift = jnp.split(time_emb, 2)

        h = self.block1(x, scale_shift=scale_shift, key=keys[0])

        h = self.block2(h, key=keys[1])

        return h + self.res_conv(x)


class LinearAttention(eqx.Module):
    scale: float
    heads: int
    norm1: RMSNorm
    mem_kv: Float[Array, "h d kv"]
    to_qkv: eqx.nn.Conv2d
    conv: eqx.nn.Conv2d
    norm2: RMSNorm

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        *,
        key: Key
    ):
        keys = jr.split(key, 3)

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm1 = RMSNorm(dim)

        self.mem_kv = jr.normal(keys[0], (2, heads, dim_head, num_mem_kv))

        self.to_qkv = eqx.nn.Conv2d(
            dim, hidden_dim * 3, kernel_size=1, use_bias=False, key=keys[1]
        )

        self.conv = eqx.nn.Conv2d(hidden_dim, dim, kernel_size=1, key=keys[2])

        self.norm2 = RMSNorm(dim)

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape

        x = precision_cast(self.norm1, x)

        qkv = jnp.split(self.to_qkv(x), 3)
        q, k, v = tuple(
            rearrange(t, '(h c) x y -> h c (x y)', h=self.heads) for t in qkv
        )

        mk, mv = tuple(repeat(t, 'h c n -> h c n') for t in self.mem_kv)
        k, v = map(partial(jnp.concatenate, axis=-1), ((mk, k), (mv, v)))

        q = precision_cast(
            lambda q: jax.nn.softmax(q - jnp.max(q), axis=-2), q # Casting, q - q.max()
        )
        k = precision_cast(
            lambda k: jax.nn.softmax(k - jnp.max(k), axis=-1), k # Casting, k - k.max()
        )

        q = q * self.scale

        context = einsum(k, v, 'h d n, h e n -> h d e')

        out = einsum(context, q, 'h d e, h d n -> h e n')
        out = rearrange(
            out, 'h c (x y) -> (h c) x y', h=self.heads, x=h, y=w
        )
        out = self.conv(out)

        return precision_cast(self.norm2, out)


class Attention(eqx.Module):
    scale: float
    heads: int
    norm: RMSNorm
    mem_kv: Float[Array, "2 h kv d"]
    to_qkv: eqx.nn.Conv2d
    to_out: eqx.nn.Conv2d

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        flash: bool = False,
        *,
        key: Key
    ):
        keys = jr.split(key, 3)

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = jr.normal(keys[0], (2, heads, num_mem_kv, dim_head))

        self.to_qkv = eqx.nn.Conv2d(
            dim, hidden_dim * 3, kernel_size=1, use_bias=False, key=keys[1]
        )

        self.to_out = eqx.nn.Conv2d(
            hidden_dim, dim, kernel_size=1, use_bias=False, key=keys[2]
        )

    def __call__(self, x: Array) -> Array:
        c, h, w = x.shape

        x = precision_cast(self.norm, x)

        qkv = jnp.split(self.to_qkv(x), 3)
        q, k, v = map(lambda t: rearrange(t, '(h c) x y -> h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> h n d'), self.mem_kv)
        k, v = map(partial(jnp.concatenate, axis=-2), ((mk, k), (mv, v)))

        q = q * self.scale
        sim = einsum(q, k, 'h i d, h j d -> h i j')

        attn = precision_cast(
            lambda sim: jax.nn.softmax(sim - jnp.max(sim), axis=-1), sim
        )

        out = einsum(attn, v, 'h i j, h j d -> h i d')
        out = rearrange(out, 'h (x y) d -> (h d) x y', x=h, y=w)

        return self.to_out(out)
 

class TimeMLP(eqx.Module):
    embed: RandomOrLearnedSinusoidalPosEmb | SinusoidalPosEmb
    layers: List[eqx.Module]

    def __init__(self, embed, fourier_dim, time_dim, a_dim, *, key):
        keys = jr.split(key)
        self.embed = embed 
        self.layers = [
            eqx.nn.Linear(
                fourier_dim + a_dim if exists(a_dim) else fourier_dim, 
                time_dim, 
                key=keys[0]
            ),
            jax.nn.gelu,
            eqx.nn.Linear(
                time_dim + a_dim if exists(a_dim) else time_dim, 
                time_dim, 
                key=keys[1]
            )
        ]

    def __call__(self, t, a):
        t = self.embed(t)
        t = self.layers[0](jnp.concatenate([t, a]) if exists(a) else t)
        t = self.layers[1](t)
        t = self.layers[2](jnp.concatenate([t, a]) if exists(a) else t)
        return t


class UNet(eqx.Module):
    channels: int
    init_conv: eqx.nn.Conv2d
    random_or_learned_sinusoidal_cond: bool
    time_mlp: List[eqx.Module]

    downs: List[eqx.Module]
    ups: List[eqx.Module]

    mid_block1: ResnetBlock
    mid_attn: Attention | LinearAttention
    mid_block2: ResnetBlock

    out_dim: int
    final_res_block: ResnetBlock
    final_conv: eqx.nn.Conv2d

    scaler: Optional[eqx.Module]

    @typecheck
    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        q_channels: Optional[int] = None,
        a_dim: Optional[int] = None,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10_000,
        dropout: float = 0.,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        time_dim_expansion: int = 4,
        full_attn: bool = False , # Defaults to full attention only for inner most layer
        flash_attn: bool = False,
        scaler: Optional[Scaler] = None,
        *,
        key: PRNGKeyArray
    ):
        key_modules, key_down, key_mid, key_up, key_final = jr.split(key, 5)
        keys = jr.split(key_modules, 3)

        # Determine dimensions
        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = eqx.nn.Conv2d(
            channels + q_channels if exists(q_channels) else channels, 
            init_dim, 
            kernel_size=7, 
            padding=3, 
            key=keys[0]
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Time embeddings
        time_dim = dim * time_dim_expansion

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features, key=keys[1]
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(
                dim, theta=sinusoidal_pos_emb_theta
            )
            fourier_dim = dim

        self.time_mlp = TimeMLP(
            sinu_pos_emb, fourier_dim, time_dim, a_dim, key=keys[3]
        )

        # Attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # Prepare blocks
        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        # Layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        # Downsampling layers 
        for ind, (
            (dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head
        ) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            keys = jr.split(jr.fold_in(key_down, ind), 4)

            is_last = ind >= (num_resolutions - 1)

            attn_class = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(
                [
                    resnet_block(dim_in, dim_in, key=keys[0]),
                    resnet_block(dim_in, dim_in, key=keys[1]),
                    attn_class(
                        dim_in, 
                        dim_head=layer_attn_dim_head, 
                        heads=layer_attn_heads, 
                        key=keys[2]
                    ),
                    Downsample(dim_in, dim_out, key=keys[3]) 
                    if not is_last else 
                    eqx.nn.Conv2d(
                        dim_in, dim_out, kernel_size=3, padding=1, key=keys[3]
                    )
                ]
            )

        # Middle layers + attention
        mid_dim = dims[-1]
        keys = jr.split(key_mid, 3)
        self.mid_block1 = resnet_block(mid_dim, mid_dim, key=keys[0])
        self.mid_attn = FullAttention(
            mid_dim, 
            heads=attn_heads[-1], 
            dim_head=attn_dim_head[-1],
            key=keys[1]
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim, key=keys[2])

        # Upsampling layers + skip connections
        for ind, (
            (dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            keys = jr.split(jr.fold_in(key_up, ind), 4)

            is_last = ind == (len(in_out) - 1)

            attn_class = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                [
                    resnet_block(dim_out + dim_in, dim_out, key=keys[0]),
                    resnet_block(dim_out + dim_in, dim_out, key=keys[1]),
                    attn_class(
                        dim_out, 
                        dim_head=layer_attn_dim_head, 
                        heads=layer_attn_heads, 
                        key=keys[2]
                    ),
                    Upsample(dim_out, dim_in, key=keys[3]) if not is_last else eqx.nn.Conv2d(
                        dim_out, dim_in, kernel_size=3, padding=1, key=keys[3]
                    )
                ]
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        keys = jr.split(key_final)
        self.final_res_block = resnet_block(init_dim * 2, init_dim, key=keys[0])
        self.final_conv = eqx.nn.Conv2d(
            init_dim + q_channels if exists(q_channels) else init_dim, 
            self.out_dim, 
            kernel_size=1, 
            key=keys[1]
        )

        self.scaler = scaler

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    @typecheck
    def __call__(
        self, 
        t: TimeArray, 
        x: XArray, 
        q: QArray = None, 
        a: AArray = None, 
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "_ _ _"]:

        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), (
            f"Input dimensions {x.shape[-2:]} need to be divisible"
            f" by {self.downsample_factor}, given the UNet"
        )

        if exists(self.scaler):
            x, q, a = self.scaler.forward(x, q, a)

        key_down, key_mid, key_up, key_final = key_split_allowing_none(key, n=4) 

        x = self.init_conv(jnp.concatenate([x, q]) if exists(q) else x)
        r = x.copy()

        t = self.time_mlp(t, a)

        h = []
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            keys = key_split_allowing_none(key_down, i=i)

            x = block1(x, t, key=keys[0])
            h.append(x)

            x = block2(x, t, key=keys[1])
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        keys = key_split_allowing_none(key_mid)
        x = self.mid_block1(x, t, key=keys[0])
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t, key=keys[1])

        for i, (block1, block2, attn, upsample) in enumerate(self.ups):
            keys = key_split_allowing_none(key_up, i=i)

            x = jnp.concatenate([x, h.pop()])
            x = block1(x, t, key=keys[0])

            x = jnp.concatenate([x, h.pop()])
            x = block2(x, t, key=keys[1])
            x = attn(x) + x

            x = upsample(x)

        x = jnp.concatenate([x, r])
        x = self.final_res_block(x, t, key=key_final)

        return self.final_conv(jnp.concatenate([x, q]) if exists(q) else x)
    

"""
    Diffusion Transformer (DiT)
"""


class AdaLayerNorm(eqx.Module):
    norm: eqx.nn.LayerNorm
    scale_proj: eqx.nn.Linear
    shift_proj: eqx.nn.Linear

    @typecheck
    def __init__(self, embed_dim: int, *, key: PRNGKeyArray):
        keys = jr.split(key)
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.scale_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[0])
        self.shift_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])

    @typecheck
    def __call__(self, x: Float[Array, "q"], y: Float[Array, "y"]):
        gamma = self.scale_proj(y)#[jnp.newaxis, :]  # (1, D)
        beta = self.shift_proj(y)#[jnp.newaxis, :]   # (1, D)
        return self.norm(x) * (1. + gamma) + beta


class PatchEmbedding(eqx.Module):
    patch_size: int
    proj: eqx.nn.Conv2d
    cls_token: Float[Array, "1 1 e"]
    pos_embed: Float[Array, "1 s e"]

    @typecheck
    def __init__(
        self, 
        img_size: int, 
        patch_size: int, 
        in_channels: int, 
        embed_dim: int, 
        key: PRNGKeyArray
    ):
        keys = jr.split(key, 3)
        self.patch_size = patch_size
        self.proj = eqx.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, key=keys[0])
        self.cls_token = jr.normal(keys[1], (1, embed_dim)) # extra 1 before
        self.pos_embed = jr.normal(keys[2], (int(img_size / patch_size) ** 2 + 1, embed_dim))

    @typecheck
    def __call__(self, x: Float[Array, "_ _ _"]) -> Float[Array, "s q"]:
        x = self.proj(x)
        x = rearrange(x, "c h w -> (h w) c") 
        x = jnp.concatenate([self.cls_token, x], axis=0) 
        x = x + self.pos_embed  
        return x


class TimestepEmbedding(eqx.Module):
    embed_dim: int
    mlp: eqx.nn.Sequential

    @typecheck
    def __init__(self, embed_dim: int, *, key: PRNGKeyArray):
        self.embed_dim = embed_dim

        keys = jr.split(key)
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(1, embed_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(embed_dim, embed_dim, key=keys[1]),
            ]
        )

    @typecheck
    def __call__(self, t: TimeArray) -> Float[Array, "{self.embed_dim}"]:
        return self.mlp(jnp.atleast_1d(t))


class TransformerBlock(eqx.Module):
    norm1: AdaLayerNorm
    attn: eqx.nn.MultiheadAttention
    norm2: AdaLayerNorm
    mlp: eqx.nn.Sequential

    @typecheck
    def __init__(
        self, 
        embed_dim: int, 
        n_heads: int, 
        *, 
        key: PRNGKeyArray
    ):
        keys = jr.split(key, 5)
        self.norm1 = AdaLayerNorm(embed_dim, key=keys[0])
        self.attn = eqx.nn.MultiheadAttention(n_heads, embed_dim, key=keys[1]) # NOTE: Casting in here...
        self.norm2 = AdaLayerNorm(embed_dim, key=keys[2])
        self.mlp = eqx.nn.Sequential(
            [
                eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[3]),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[4])
            ]
        )

    @typecheck
    def __call__(self, x: Float[Array, "s q"], y: Float[Array, "y"]) -> Float[Array, "s q"]:
        x = precision_cast(jax.vmap(lambda x: self.norm1(x, y)), x)
        x = x + self.attn(x, x, x)
        x = precision_cast(jax.vmap(lambda x: self.norm2(x, y)), x)
        x = x + jax.vmap(self.mlp)(x)
        return x


class DiT(eqx.Module):
    img_size: int
    q_dim: int
    patch_embed: PatchEmbedding
    time_embed: TimestepEmbedding
    a_embed: Optional[eqx.nn.Linear]
    blocks: List[TransformerBlock]
    out_conv: eqx.nn.ConvTranspose2d
    scaler: Optional[Scaler] = None

    @typecheck
    def __init__(
        self, 
        img_size: int, 
        patch_size: int, 
        channels: int, 
        embed_dim: int, 
        depth: int, 
        n_heads: int, 
        q_dim: Optional[int] = None, 
        a_dim: Optional[int] = None, 
        scaler: Optional[Scaler] = None,
        *, 
        key: PRNGKeyArray
    ):
        self.img_size = img_size
        self.q_dim = q_dim

        keys = jr.split(key, 5)
        channels = channels + q_dim if (q_dim is not None) else channels

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, channels, embed_dim, key=keys[0]
        )
        self.time_embed = TimestepEmbedding(embed_dim, key=keys[1])
        
        self.a_embed = eqx.nn.Linear(a_dim, embed_dim, key=keys[2]) if (a_dim is not None) else None

        block_keys = jr.split(keys[3], depth)
        self.blocks = eqx.filter_vmap(
            lambda key: TransformerBlock(embed_dim, n_heads, key=key) 
        )(block_keys)

        self.out_conv = eqx.nn.ConvTranspose2d(
            embed_dim, channels, kernel_size=patch_size, stride=patch_size, key=keys[4]
        )

        self.scaler = scaler

    @typecheck
    def __call__(
        self, 
        t: TimeArray, 
        x: XArray, 
        q: QArray, 
        a: AArray,
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "_ _ _"]:

        if exists(self.scaler):
            x, q, a = self.scaler.forward(x, q, a)

        x = self.patch_embed(
            jnp.concatenate([x, q]) 
            if (q is not None) and (self.q_dim is not None)
            else x
        )

        t_embedding = self.time_embed(t)
        if (a is not None) and (self.a_dim is not None):
            a_embedding = self.a_embed(a)
            embedding = a_embedding + t_embedding
        else:
            embedding = t_embedding

        all_params, struct = eqx.partition(self.blocks, eqx.is_array)

        def block_fn(x, params):
            block = eqx.combine(params, struct)
            x = block(x, embedding)
            return x, None

        x, _ = jax.lax.scan(block_fn, x, all_params)

        x = x[1:] # No class token 
        x = rearrange(
            x, 
            "(h w) c -> c h w", 
            h=int(self.img_size / self.patch_embed.patch_size)
        )  
        x = self.out_conv(
            jnp.concatenate([x, q]) 
            if (q is not None) and (self.q_dim is not None)
            else x
        )

        return x


"""
    Rectified Flow
"""


class RectifiedFlow(eqx.Module):
    net: eqx.Module | UNet | DiT

    def __init__(self, net: eqx.Module | UNet | DiT):
        self.net = net

    def __call__(self, *args, **kwargs):
        return self.v(*args, **kwargs)

    @typecheck
    def alpha(self, t: TimeArray) -> Scalar:
        return 1.0 - t # t1? # t1 - t / t1?

    @typecheck
    def sigma(self, t: TimeArray) -> Scalar:
        return t

    @typecheck
    def p_t(
        self, 
        x: XArray, 
        t: TimeArray, 
        eps: Float[Array, "_ _ _"]
    ) -> Float[Array, "_ _ _"]:
        return self.alpha(t) * x + self.sigma(t) * eps

    @typecheck
    def v(
        self, 
        t: TimeArray, 
        x: XArray, 
        q: QArray, 
        a: AArray,
        key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "_ _ _"]:
        return self.net(t, x, q, a, key=key)

    @typecheck
    @eqx.filter_jit
    def sample(
        self,
        q: QArray, 
        a: AArray,
        key: PRNGKeyArray, 
        x_shape: Sequence[int], 
        *,
        t0: float, 
        t1: float, 
        dt: float, 
        solver: Optional[dfx.AbstractSolver] = None,
        ts: Optional[Float[Array, "_"]] = None,
        q_as_x_1: bool = False
    ) -> Float[Array, "_ _ _"]:
        return single_sample_fn(
            self.net, 
            q, 
            a, 
            key, 
            x_shape, 
            t0, 
            t1, 
            dt, 
            solver, 
            ts=ts, 
            q_as_x_1=q_as_x_1
        )

    @typecheck
    @eqx.filter_jit
    def sample_ode(
        self,
        q: QArray, 
        a: AArray,
        key: PRNGKeyArray, 
        x_shape: Sequence[int], 
        *,
        t0: float, 
        t1: float, 
        dt: float, 
        alpha: float = 0.1,
        ts: Optional[Float[Array, "_"]] = None,
        solver: Optional[dfx.AbstractSolver] = None,
        q_as_x_1: bool = False
    ) -> Float[Array, "_ _ _"]:
        return single_sample_fn_ode(
            self.net, 
            q, 
            a, 
            key, 
            x_shape, 
            t0, 
            t1, 
            dt, 
            alpha,
            solver,
            ts=ts,
            q_as_x_1=q_as_x_1
        )

    @typecheck
    @eqx.filter_jit
    def sample_stochastic(
        self,
        q: QArray, 
        a: AArray,
        key: PRNGKeyArray, 
        x_shape: Sequence[int], 
        *,
        t0: float, 
        t1: float, 
        g_scale: float = 0.1,
        n_steps: int = 1000,
        q_as_x_1: bool = False
    ) -> Float[Array, "_ _ _"]:
        return single_non_singular_sample_fn(
            self.net, 
            q, 
            a, 
            key, 
            x_shape, 
            t0=t0, 
            t1=t1, 
            g_scale=g_scale, 
            n_steps=n_steps,
            q_as_x_1=q_as_x_1
        )

    @typecheck
    @eqx.filter_jit
    def log_prob(
        self,
        x: XArray,
        q: QArray, 
        a: AArray,
        t0: float, 
        t1: float, 
        dt: float, 
        solver: Optional[dfx.AbstractSolver] = None,
        exact_log_prob: bool = False,
        n_eps: Optional[int] = 10
    ) -> Scalar:
        return single_likelihood_fn(
            self.net, 
            x, 
            q, 
            a, 
            key, 
            t0, 
            t1, 
            dt, 
            solver, 
            exact_log_prob=exact_log_prob,
            n_eps=n_eps
        )


"""
    Sampling
"""


@typecheck
@eqx.filter_jit
def single_sample_fn(
    v: eqx.Module, 
    q: Optional[QArray], 
    a: Optional[AArray],
    key: PRNGKeyArray, 
    x_shape: Sequence[int], 
    t0: float, 
    t1: float, 
    dt: float, 
    solver: Optional[dfx.AbstractSolver] = None,
    *,
    ts: Optional[Float[Array, "_"]] = None,
    q_as_x_1: bool = False
) -> Float[Array, "_ _ _"]:
    solver = default(solver, dfx.Euler())

    def flow(t, x, args):
        (q, a, v) = args
        t = jnp.asarray(t)
        return v(t, x, q, a)
    
    v = eqx.nn.inference_mode(v, True)

    if q_as_x_1 and exists(q):
        z, q = q.copy(), None
    else:
        z = jr.normal(key, x_shape)

    term = dfx.ODETerm(flow) 

    if exists(ts):
        saveat = dfx.SaveAt(ts=ts)
    else:
        saveat = dfx.SaveAt(t1=True)

    sol = dfx.diffeqsolve(
        term, solver, t1, t0, -dt, z, args=(q, a, v), saveat=saveat
    )

    if exists(ts):
        x_ = sol.ys
    else:
        (x_,) = sol.ys

        scaler = v.scaler
        if exists(scaler):
            x_, _, _ = scaler.inverse(x_, q, a) # Inverse scale 
    return x_ 


@typecheck
def get_sample_fn(
    v: eqx.Module, 
    x_shape: Sequence[int], 
    soln_kwargs: Optional[dict] = {}
) -> SampleFn:
    def _sample_fn(
        key: PRNGKeyArray, 
        q: QArray, 
        a: AArray
    ) -> Float[Array, "_ _ _"]:
        return v.sample(
            q, a, key, x_shape, **soln_kwargs
        )
    return _sample_fn


@typecheck
@eqx.filter_jit
def single_sample_fn_ode(
    v: eqx.Module, 
    q: QArray, 
    a: AArray,
    key: PRNGKeyArray, 
    x_shape: Sequence[int], 
    t0: Optional[float] = None, 
    t1: Optional[float] = None, 
    dt: Optional[float] = None, 
    alpha: float = 0.1,
    solver: Optional[dfx.AbstractSolver] = None,
    *,
    ts: Optional[Float[Array, "_"]] = None,
    q_as_x_1: bool = False
) -> Float[Array, "_ _ _"]:
    """ Sample ODE of SDE corresponding to Gaussian flow matching marginals """

    solver = default(solver, dfx.Euler())

    def _ode(t, x, args):
        # Non-Singular ODE; using score of Gaussian Rectified Flow
        (q, a, v) = args
        t = jnp.asarray(t)
        _v = v(t, x, q, a)
        score = -((1. - t) * _v + x) / t # Assuming mu_1, sigma_1 = 0, 1
        drift = _v + 0.5 * alpha ** 2. * t * score
        return drift
    
    v = eqx.nn.inference_mode(v, True)

    if q_as_x_1 and exists(q):
        z, q = q.copy(), None
    else:
        z = jr.normal(key, x_shape)

    term = dfx.ODETerm(_ode) 

    if exists(ts):
        saveat = dfx.SaveAt(ts=ts)
    else:
        saveat = dfx.SaveAt(t1=True)

    sol = dfx.diffeqsolve(
        term, 
        solver, 
        t1, 
        t0, 
        -dt, 
        z, 
        args=(q, a, v),
        saveat=saveat
    )

    if exists(ts):
        x_ = sol.ys
    else:
        (x_,) = sol.ys

        scaler = v.scaler
        if exists(scaler):
            x_, _, _ = scaler.inverse(x_, q, a) 
    return x_


@typecheck
@eqx.filter_jit
def single_sample_fn_sde(
    v: eqx.Module, 
    q: QArray, 
    a: AArray,
    key: PRNGKeyArray, 
    x_shape: Sequence[int], 
    *, 
    t0: Optional[float] = None, 
    t1: Optional[float] = None, 
    dt: Optional[float] = None, 
    alpha: float = 0.1, # Arbitrary constant, controls diffusion coeff...
    solver: Optional[dfx.AbstractSolver] = None,
    ts: Optional[Float[Array, "_"]] = None,
    q_as_x_1: bool = False
) -> Float[Array, "_ _ _"]:

    solver = default(solver, dfx.Euler())

    def _ode(t, x, args):
        # Singular ODE
        (q, a, v) = args
        _v = v(t, x, q, a)
        score = -((1. - t) * _v + x) / t
        drift = _v + 0.5 * alpha ** 2. * t * score
        return drift
    
    v = eqx.nn.inference_mode(v, True)

    if q_as_x_1 and exists(q):
        z, q = q.copy(), None
    else:
        z = jr.normal(key, x_shape)

    term = dfx.ODETerm(_ode) 

    if exists(ts):
        saveat = dfx.SaveAt(ts=ts)
    else:
        saveat = dfx.SaveAt(t1=True)

    sol = dfx.diffeqsolve(
        term, 
        solver, 
        t1, 
        t0, 
        -dt, 
        z, 
        args=(q, a, v),
        saveat=saveat
    )
    (x_,) = sol.ys

    scaler = v.scaler
    if exists(scaler):
        x_, _, _ = scaler.inverse(x_, q, a)
    return x_


@typecheck
def get_sample_fn_ode(
    flow: eqx.Module, 
    x_shape: Sequence[int], 
    soln_kwargs: Optional[dict] = {}
) -> SampleFn:
    def _sample_fn(
        key: PRNGKeyArray, 
        q: QArray, 
        a: AArray
    ) -> Float[Array, "_ _ _"]:
        return single_sample_fn_ode(
            flow, q, a, key, x_shape, **soln_kwargs
        )
    return _sample_fn


@typecheck
@eqx.filter_jit
def single_sample_along_time(
    v: eqx.Module, 
    q: QArray, 
    a: AArray,
    key: PRNGKeyArray, 
    x_shape: Sequence[int], 
    t0: float, 
    t1: float, 
    dt: float, 
    solver: Optional[dfx.AbstractSolver] = None,
    *,
    n_time_steps: int = 100,
    q_as_x_1: bool = False
) -> Float[Array, "t _ _ _"]:

    solver = default(solver, dfx.Euler())

    def flow(t, x, args):
        (q, a, v) = args
        t = jnp.asarray(t)
        return v(t, x, q, a)
    
    v = eqx.nn.inference_mode(v, True)

    if q_as_x_1 and exists(q):
        z, q = q.copy(), None
    else:
        z = jr.normal(key, x_shape)

    ts = jnp.linspace(t1, t0, n_time_steps)

    term = dfx.ODETerm(flow) 

    sol = dfx.diffeqsolve(
        term, 
        solver, 
        t1, 
        t0, 
        -dt, 
        z, 
        args=(q, a, v), 
        saveat=dfx.SaveAt(ts=ts)
    )
    (x_,) = sol.ys

    scaler = v.scaler
    if exists(scaler):
        x_, _, _ = scaler.inverse(x_, q, a) # Inverse scale 
    return x_


@typecheck
@eqx.filter_jit
def single_non_singular_sample_fn(
    v: eqx.Module, 
    q: QArray, 
    a: AArray,
    key: PRNGKeyArray, 
    x_shape: Sequence[int],
    *,
    t0: float, 
    t1: float, 
    g_scale: float = 0.1, 
    n_steps: int = 1000,
    n: int = 1, 
    m: int = 0,
    q_as_x_1: bool = False
) -> Float[Array, "_ _ _"]:

    key_z, key_sample = jr.split(key)

    t = jnp.linspace(t0, t1, n_steps + 1) 

    v = eqx.nn.inference_mode(v)

    def sample_step(i, z):
        z, q, a, key = z

        key, key_eps, key_apply = jr.split(key, 3)

        _t = t[i] 
        dt = t[i + 1] - t[i]

        eps = jr.normal(key_eps, z.shape)

        z_hat = v(1. - _t, z, q, a, key_apply) # t1 - t?
        _z_hat = -z_hat
        g = g_scale * jnp.power(_t, 0.5 * n) * jnp.power(1. - _t, 0.5 * m)
        s_u = -((1. - _t) * _z_hat + z)
        fr = (_z_hat - jnp.square(g_scale) * jnp.power(_t, n - 1.) * jnp.power(1 - _t, m) * 0.5 * s_u)

        dbt = jnp.sqrt(jnp.abs(dt)) * eps
        z = z + fr * dt + g * dbt

        return z, q, a, key 

    if q_as_x_1 and exists(q):
        z, q = q.copy(), None
    else:
        z = jr.normal(key_z, x_shape)

    x, *_ = jax.lax.fori_loop(
        lower=0, 
        upper=n_steps, 
        body_fun=sample_step, 
        init_val=(z, q, a, key_sample)
    )

    return x


@typecheck
def get_sample_fn_non_singular(
    flow: eqx.Module, 
    x_shape: Sequence[int], 
    stochastic_kwargs: Optional[dict] = {},
    soln_kwargs: Optional[dict] = {}
) -> SampleFn:
    if not soln_kwargs:
        _soln_kwargs = soln_kwargs.copy()
        _soln_kwargs.pop("dt")

    def _sample_fn(
        key: PRNGKeyArray, 
        q: QArray, 
        a: AArray
    ) -> Float[Array, "_ _ _"]:
        return single_non_singular_sample_fn(
            flow, q, a, key, x_shape, **stochastic_kwargs, **_soln_kwargs
        )
    return _sample_fn


"""
    Likelihood
"""


def normal_log_likelihood(z):
    return jnp.sum(jax.scipy.stats.norm.pdf(z))


@typecheck
def log_prob_approx(
    t: TimeArray,
    y: XArray, 
    args: Tuple[
        QArray, 
        AArray, 
        Optional[Float[Array, "_ _ _"]], # Eps
        Union[
            eqx.Module,
            Callable[
                [TimeArray, XArray, QArray, AArray], 
                Float[Array, "_ _ _"]
            ]
        ]
    ]
) -> Tuple[Float[Array, "_ _ _"], Scalar]:
    """ 
        Approx. trace using Hutchinson's trace estimator. 
        - optional multiple-eps sample to average estimated log_prob over
    """
    y, _ = y 
    (q, a, eps, v) = args
    
    f, f_vjp = jax.vjp(lambda y: v(t, y, q, a), y) # f = f(*primals)
    
    # Expectation over multiple eps
    estimator = lambda eps_dfdy, eps: jnp.sum(eps_dfdy * jnp.flatten(eps))
    if eps.ndim == (len((1,) + y.shape)):
        (eps_dfdy,) = jax.vmap(f_vjp)(eps.reshape(eps.shape[0], -1))
        log_probs = jax.vmap(estimator)(eps_dfdy, eps)
        log_prob = jnp.mean(log_probs, axis=0)
    else:
        (eps_dfdy,) = f_vjp(jnp.flatten(eps))
        log_prob = estimator(eps_dfdy, eps) #jnp.sum(eps_dfdy * jnp.flatten(eps))
        
    return f, log_prob


@typecheck
def log_prob_exact(
    t: TimeArray, 
    y: XArray,
    args: Tuple[
        QArray,
        AArray, 
        Optional[Float[Array, "_ _ _"]], # Eps
        Union[
            eqx.Module,
            Callable[
                [TimeArray, XArray, QArray, AArray], 
                Float[Array, "_ _ _"]
            ]
        ]
    ]
) -> Tuple[Float[Array, "_ _ _"], Scalar]:
    """ 
        Compute trace directly. 
    """
    y, _ = y
    (q, a, _, v) = args

    f, f_vjp = jax.vjp(lambda y: v(t, y, q, a), y) 

    (dfdy,) = jax.vmap(f_vjp)(jnp.eye(y.size)) 
    log_prob = jnp.trace(dfdy)

    return f, log_prob


@typecheck
@eqx.filter_jit(donate="all-except-first")
def single_likelihood_fn(
    v: eqx.Module, 
    x: XArray, 
    q: QArray, 
    a: AArray, 
    key: PRNGKeyArray, 
    t0: float, 
    t1: float, 
    dt: float, 
    solver: Optional[dfx.AbstractSolver] = None,
    exact_log_prob: bool = False,
    n_eps: Optional[int] = 10
)-> Scalar:

    solver = default(solver, dfx.Euler())

    v = eqx.nn.inference_mode(v, True)

    term = dfx.ODETerm(
        log_prob_exact if exact_log_prob else log_prob_approx
    )

    if exists(key) and not exact_log_prob:
        if exists(n_eps):
            eps_shape = (n_eps,) + x.shape
        else: 
            eps_shape = x.shape
        eps = jr.normal(key, eps_shape)
    else:
        eps = None

    delta_log_likelihood = 0.
    x0_p0 = (x, delta_log_likelihood)

    sol = dfx.diffeqsolve(
        term, solver, t0, t1, dt, x0_p0, args=(q, a, eps, v)
    )
    (y1,), (delta_log_likelihood,) = sol.ys

    return delta_log_likelihood + normal_log_likelihood(y1) # NOTE: not for q_as_x1


"""
    Loss
"""


@typecheck
def loss_fn(
    model: RectifiedFlow, 
    ema_model: RectifiedFlow, 
    key: PRNGKeyArray, 
    # Data and times
    x: XArray, 
    q: QArray, 
    a: AArray, 
    t: TimeArray, 
    q_as_x_1: bool,
    # Flow matching
    t0: float, 
    t1: float, 
    sigma_0: float, 
    policy: Optional[Policy] = None
) -> Tuple[Scalar, Tuple[Scalar]]:

    key_noise, key_apply = jr.split(key)

    if q_as_x_1 and exists(q):
        x_1, q = q.copy(), None
    else:
        x_1 = jr.normal(key_noise, x.shape, x.dtype) # t=1 => unit noise

    if exists(policy):
        model = policy.cast_to_compute(model) 
        x, q, a, t, x_1 = policy.cast_to_compute((x, q, a, t, x_1))

    x_t = model.p_t(x, t, x_1) 

    v = model.v(t, x_t, q, a, key=key_apply) 
    l = jnp.mean(jnp.square(jnp.subtract(v, x_1 - x))) # NOTE: wasn't sum before...

    if exists(policy):
        l = policy.cast_to_output(l)

    return l, (l,) # Signature for metrics


def cosine_time(t: TimeArray) -> TimeArray:
    return 1. - (1. / (jnp.tan(0.5 * jnp.pi * t) + 1.)) # t1?


def get_time(
    key: PRNGKeyArray, 
    n: int, 
    t0: float,
    t1: float, 
    noise_schedule: Optional[ScheduleFn] = identity
) -> Float[Array, "n"]: 
    # Use lower variance but clip to [t0 + dt, t1]?
    t = jr.uniform(key, (n,), minval=t0, maxval=t1 / n) 
    t = t + (t1 / n) * jnp.arange(n)
    t = noise_schedule(t)
    return t 


@typecheck
def batch_loss_fn(
    model: RectifiedFlow, 
    ema_model: RectifiedFlow, 
    key: PRNGKeyArray, 
    x: Float[Array, "n _ _ _"], 
    q: Optional[Float[Array, "n _ _ _"]], 
    a: Optional[Float[Array, "n _"]], 
    t: Float[Array, "n"],
    q_as_x_1: bool = False,
    policy: Optional[Policy] = None,
    cfm_kwargs: Optional[dict] = {}
) -> Tuple[Scalar, List[Scalar]]:
    keys = jr.split(key, x.shape[0])
    _fn = partial(
        loss_fn, 
        model, 
        ema_model, 
        q_as_x_1=q_as_x_1, 
        policy=policy, 
        **cfm_kwargs
    )
    L, m = jax.vmap(_fn)(keys, x, q, a, t)
    return jnp.mean(L), [jnp.mean(_m) for _m in m if exists(_m)]


"""
    Train
"""


def load_model_and_opt_state(
    model: eqx.Module, 
    name: Optional[str] = None, 
    *, 
    run_dir: Path
) -> Tuple[
    eqx.Module, optax.GradientTransformation, optax.OptState, int
]:

    def load_model(model, filename):
        model = eqx.tree_deserialise_leaves(filename, model)
        return model

    def load_opt_state(filename="state.obj"):
        f = open(filename, 'rb')
        state = cloudpickle.load(f)
        f.close()
        return state

    name = default(name, model.__class__.__name__)

    model = load_model(model, run_dir / name + ".eqx")
    opt, opt_state, i = load_opt_state(run_dir / name + "_state.obj")

    return model, opt, opt_state, i


def save_model_and_opt_state(
    model: eqx.Module, 
    opt: optax.GradientTransformation, 
    opt_state: optax.OptState, 
    i: int, 
    name: Optional[str] = None, 
    *, 
    run_dir: Path 
) -> None:

    def save_model(model, filename):
        eqx.tree_serialise_leaves(str(filename), model)

    def save_opt_state(opt, opt_state, i, filename="state.obj"):
        """ Save an optimiser and its state for a model, to train later """
        state = {
            "opt" : opt, "opt_state" : opt_state, "step" : i
        }
        f = open(str(filename), 'wb')
        cloudpickle.dump(state, f)
        f.close()

    name = default(name, model.__class__.__name__)

    save_model(
        model, filename=run_dir / ("out/" + name + ".eqx")
    )
    # save_opt_state(
    #     opt, opt_state, i, filename=run_dir / ("out/" + name + "_state.obj")
    # )


@typecheck
def accumulate_gradients_scan(
    model: eqx.Module,
    ema_model: eqx.Module,
    key: PRNGKeyArray,
    x: Float[Array, "b _ _ _"], 
    q: Optional[Float[Array, "b _ _ _"]], 
    a: Optional[Float[Array, "b _"]], 
    t: Float[Array, "b"],
    n_minibatches: int,
    *,
    grad_fn: Callable = None
) -> Tuple[Tuple[Scalar, Scalar], PyTree]:

    batch_size = x.shape[0]
    minibatch_size = int(batch_size / n_minibatches)
    keys = jr.split(key, n_minibatches)

    def _minibatch_step(minibatch_idx):
        # Gradients and metrics for a single minibatch. 
        xqat = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(  
                x, 
                start_index=minibatch_idx * minibatch_size, 
                slice_size=minibatch_size, 
                axis=0
            ),
            (x, q, a, t), # This works for tuples of batched data e.g. (x, q, a)
        )
        (x, q, a, t) = xqat
        (_, step_metrics), step_grads = grad_fn(
            model, ema_model, keys[minibatch_idx], x, q, a, t
        )
        return step_grads, step_metrics

    def _scan_step(carry, minibatch_idx):
        # Scan step function for looping over minibatches.
        step_grads, step_metrics = _minibatch_step(minibatch_idx)
        carry = jax.tree.map(jnp.add, carry, (step_grads, step_metrics))
        return carry, None

    # Determine initial shapes for gradients and metrics (NOTE: x.dtype may be lower precision).
    grads_shapes, metrics_shape = jax.eval_shape(_minibatch_step, 0)
    grads = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grads_shapes)
    metrics = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)

    # Loop over minibatches to determine gradients and metrics.
    (grads, metrics), _ = jax.lax.scan(
        _scan_step, 
        init=(grads, metrics), 
        xs=jnp.arange(n_minibatches), 
        length=n_minibatches
    )

    # Average gradients over minibatches.
    grads = jax.tree.map(lambda g: g / n_minibatches, grads)
    metrics = jax.tree.map(lambda m: m / n_minibatches, metrics)

    return (metrics[0], metrics), grads # Same signature as unaccumulated 


@typecheck
@eqx.filter_jit(donate="all-except-first")
def evaluate(
    model: RectifiedFlow, 
    ema_model: RectifiedFlow,
    x: Float[Array, "n _ _ _"], 
    q: Optional[Float[Array, "n _ _ _"]], 
    a: Optional[Float[Array, "n _"]], 
    t: Float[Array, "n"],
    key: PRNGKeyArray, 
    q_as_x_1: bool = False,
    cfm_kwargs: dict = {},
    *,
    policy: Optional[Policy] = None,
    replicated_sharding: Optional[PositionalSharding] = None
) -> Tuple[Scalar, List[Scalar]]:

    if exists(replicated_sharding):
        model, ema_model = eqx.filter_shard(
            (model, ema_model), replicated_sharding
        )

    model, ema_model = eqx.nn.inference_mode((model, ema_model))

    L, metrics = batch_loss_fn(
        model, 
        ema_model, 
        key, 
        x, 
        q, 
        a, 
        t, 
        q_as_x_1=q_as_x_1, 
        policy=policy,
        cfm_kwargs=cfm_kwargs
    )

    return L, metrics


@typecheck
@eqx.filter_jit(donate="all")
def make_step(
    model: RectifiedFlow, 
    ema_model: RectifiedFlow,
    x: Float[Array , "n _ _ _"], 
    q: Optional[Float[Array , "n _ _ _"]], 
    a: Optional[Float[Array, "n _"]], 
    t: Float[Array, "n"], 
    key: PRNGKeyArray, 
    q_as_x_1: bool,
    cfm_kwargs: dict,
    opt_state: PyTree, 
    opt: optax.GradientTransformation,
    *,
    policy: Optional[Policy] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
    accumulate_gradients: bool = False,
    n_minibatches: Optional[int] = 4
) -> Tuple[Scalar, List[Scalar], RectifiedFlow, PyTree]:

    if exists(sharding):
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    grad_fn = eqx.filter_value_and_grad(
        partial(
            batch_loss_fn, 
            q_as_x_1=q_as_x_1, 
            policy=policy,     
            cfm_kwargs=cfm_kwargs
        ),
        has_aux=True
    )

    if accumulate_gradients and n_minibatches > 0:
        (loss, metrics), grads = accumulate_gradients_scan(
            model, ema_model, key, x, q, a, t, n_minibatches=n_minibatches, grad_fn=grad_fn
        ) 
    else:
        (loss, metrics), grads = grad_fn(
            model, ema_model, key, x, q, a, t
        )

    if exists(policy):
        grads = policy.cast_to_param(grads) 

    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    if exists(replicated_sharding):
        model, opt_state = eqx.filter_shard(
            (model, opt_state), replicated_sharding
        )

    return loss, metrics, model, opt_state


def get_noise_schedule_fn(
    noise_schedule: Union[Literal["cosine"], ScheduleFn]
) -> ScheduleFn:
    if not isinstance(noise_schedule, Callable):
        if noise_schedule == "cosine":
            noise_schedule_fn = cosine_time
    else:
        noise_schedule_fn = noise_schedule
    return noise_schedule_fn


@typecheck
def train(
    key: PRNGKeyArray,
    # Model
    flow: RectifiedFlow,
    # Data
    dataset: Dataset,
    # Training
    n_steps: int = 100_000,
    n_batch: int = 1000,
    lr: float = 1e-4,
    optimiser: optax.GradientTransformation = optax.adamw,
    patience: Optional[int] = None,
    use_ema: bool = False,
    accumulate_gradients: bool = False,
    n_minibatches: int = 4,
    q_as_x_1: bool = False,
    # FM
    t0: float = 0.,
    t1: float = 1.,
    dt: float = 0.01,
    noise_schedule: Union[Literal["cosine"], ScheduleFn] = identity,
    sigma_0: float = 1e-4,
    ema_rate: float = 0.9995,
    solver: dfx.AbstractSolver = dfx.Euler(),
    # Policy
    policy: Optional[Policy] = None,
    # Sharding
    sharding: Optional[NamedSharding] = None,
    replicated_sharding: Optional[PositionalSharding] = None,
    # Sampling
    n_sample: int = 64,
    # Other
    model_name: Optional[str] = None,
    reload_opt_and_model: bool = False,
    cmap: str = "coolwarm",
    run_dir: Path | str = "./" 
) -> Tuple[RectifiedFlow, np.ndarray]:

    if isinstance(run_dir, str):
        run_dir = Path(run_dir)

    if isinstance(flow, dict):
        flow = RectifiedFlow(**flow)
        
    print("Model has {:.3E} parameters.".format(count_parameters(flow)))

    if not run_dir.exists():
        run_dir.mkdir()
    for _sub_dir in ["imgs/", "out/"]:
        _dir = run_dir / _sub_dir
        if not _dir.exists():
            _dir.mkdir(exist_ok=True)

    key, key_sample, key_valid = jr.split(key, 3)

    Xs, Qs, As = next(dataset.valid_dataloader.loop(n_sample)) # Fixed sample

    if exists(sharding):
        Xs, Qs, As = eqx.filter_shard((Xs, Qs, As), sharding)

    initial_step = 0
    if reload_opt_and_model:
        (
            flow, opt, opt_state, initial_step
        ) = load_model_and_opt_state(
            flow, name=model_name, run_dir=run_dir
        )
    else:
        opt = optimiser(lr)
        opt_state = opt.init(eqx.filter(flow, eqx.is_array))

    # Consistency conditional flow matching parameters
    cfm_kwargs = dict(t0=t0, t1=t1, sigma_0=sigma_0)

    # Sampling parameters
    soln_kwargs = dict(t0=t0, t1=t1, dt=dt, solver=solver, q_as_x_1=q_as_x_1)

    # Noise schedule
    noise_schedule = get_noise_schedule_fn(noise_schedule)

    # Shard model and optimiser state
    if exists(replicated_sharding):
        flow, opt_state = eqx.filter_shard((flow, opt_state), replicated_sharding)

    # Use EMA with diffusion, but required for consistency
    if use_ema:
        ema_flow = deepcopy(flow)
        if exists(replicated_sharding):
            ema_flow = eqx.filter_shard(ema_flow, replicated_sharding)

    if accumulate_gradients:
        n_batch_train = n_batch * n_minibatches
    else:
        n_batch_train = n_batch  

    metrics_t, metrics_v = [], []
    with trange(initial_step, n_steps, colour="green") as steps:
        for step, xqa_t, xqa_v in zip(
            steps, 
            dataset.train_dataloader.loop(n_batch_train), 
            dataset.valid_dataloader.loop(n_batch)
        ):
            key, key_time, key_step = jr.split(key, 3)

            # Train
            x, q, a = shard_batch(xqa_t, sharding=sharding)
            t = get_time(key_time, n_batch, t0, t1, noise_schedule)

            Lt, _metrics_t, flow, opt_state = make_step(
                flow,
                ema_flow if use_ema else flow,
                x,
                q,
                a,
                t,
                key_step,
                q_as_x_1=q_as_x_1,
                cfm_kwargs=cfm_kwargs,
                opt_state=opt_state,
                opt=opt,
                policy=policy,
                replicated_sharding=replicated_sharding,
                accumulate_gradients=accumulate_gradients,
                n_minibatches=n_minibatches
            )

            if use_ema:
                ema_flow = apply_ema(ema_flow, flow, ema_rate, policy)

            # Validate
            x, q, a = shard_batch(xqa_v, sharding=sharding)
            t = get_time(key_time, n_batch, t0, t1, noise_schedule)

            Lv, _metrics_v = evaluate(
                ema_flow if use_ema else flow,
                ema_flow if use_ema else flow,
                x, 
                q, 
                a, 
                t, 
                key_valid, 
                q_as_x_1=q_as_x_1,
                cfm_kwargs=cfm_kwargs,
                policy=policy,
                replicated_sharding=replicated_sharding
            )

            # Record
            steps.set_postfix_str(f"{Lt=:.3E}, {Lv=:.3E}")
            metrics_t.append(_metrics_t)
            metrics_v.append(_metrics_v)

            # Early stopping
            if exists(patience):
                if len(metrics_v) - np.argmin(metrics_v[step][0]) > patience: # NOTE: fix
                    save_model_and_opt_state(
                        ema_flow if use_ema else flow, 
                        opt,
                        opt_state,
                        step,
                        name=model_name,
                        run_dir=run_dir
                    )
                    break

            # Sample model and plot metrics
            if step % 1000 == 0 or step == 100:

                sample_model(
                    key_sample, 
                    ema_flow if use_ema else flow, 
                    Xs,
                    Qs, 
                    As, 
                    x_shape=dataset.data_shape, 
                    soln_kwargs=soln_kwargs, 
                    cmap=cmap,
                    sharding=sharding,
                    filename=run_dir / "imgs/{:07d}.png".format(step)
                )

                stochastic_sample_model(
                    key_sample, 
                    ema_flow if use_ema else flow, 
                    Xs,
                    Qs, 
                    As, 
                    x_shape=dataset.data_shape, 
                    soln_kwargs=soln_kwargs, 
                    cmap=cmap,
                    sharding=sharding,
                    filename=run_dir / "imgs/stochastic_{:07d}.png".format(step)
                )

                metrics = np.stack([metrics_t, metrics_v])
                if step > 0:
                    plot_metrics(metrics, filename=run_dir / "L.png")
            
            # Save model and optimiser state
            if step % 10_000 == 0:
                save_model_and_opt_state(
                    ema_flow if use_ema else flow, 
                    opt,
                    opt_state,
                    step,
                    name=model_name,   
                    run_dir=run_dir
                )

    return ema_flow if use_ema else flow, metrics


"""
    Plots
"""


def plot_metrics(metrics: np.ndarray, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5., 4.))
    metrics_t, metrics_v = metrics
    ax.loglog(metrics_t, label="Lt")
    ax.loglog(metrics_v, label="Lv")
    ax.legend(frameon=False)
    ax.set_ylabel("L")
    plt.savefig(filename)
    plt.close()


def sample_model(
    key: Key, 
    flow: RectifiedFlow, 
    X: Array, 
    Q: Array, 
    A: Array, 
    x_shape: Sequence[int], 
    soln_kwargs: dict, 
    cmap: str, 
    sharding: Optional[NamedSharding] = None,
    *,
    filename: str
) -> None:

    n_channels, n_pix, n_pix = x_shape

    vs = dict(vmin=X.min(), vmax=X.max())

    if exists(sharding):
        X, Q, A = eqx.filter_shard((X, Q, A), sharding)

    sample_fn = lambda q, a, key: flow.sample(
        q, a, key, x_shape, **soln_kwargs
    )

    sample_keys = jr.split(key, X.shape[0])
    x_sample = eqx.filter_vmap(sample_fn)(Q, A, sample_keys)

    n_side = int(jnp.sqrt(len(x_sample)))

    x_sample = rearrange(
        x_sample, 
        "(r c) s h w -> (r h) (c w) s", 
        s=n_channels, r=n_side, c=n_side, h=n_pix, w=n_pix
    )
    X = rearrange(
        X, 
        "(r c) s h w -> (r h) (c w) s", 
        s=n_channels, r=n_side, c=n_side, h=n_pix, w=n_pix
    )

    fig, axs = plt.subplots(1, 2, dpi=200, figsize=(8., 4.))
    ax = axs[0]
    im = ax.imshow(jnp.clip(x_sample, 0., 1.), cmap=cmap, **vs)
    ax = axs[1]
    im = ax.imshow(jnp.clip(X, 0., 1.), cmap=cmap, **vs)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def stochastic_sample_model(
    key: Key, 
    flow: RectifiedFlow, 
    X: Array, 
    Q: Array, 
    A: Array, 
    x_shape: Sequence[int], 
    soln_kwargs: dict = {}, 
    cmap: Optional[str] = None, 
    sharding: Optional[NamedSharding] = None,
    *,
    filename: str
) -> None:

    n_channels, n_pix, n_pix = x_shape

    vs = dict(vmin=X.min(), vmax=X.max())

    # Don't need the ODE kwargs here
    if soln_kwargs:
        stochastic_kwargs = soln_kwargs.copy()
        stochastic_kwargs.pop("dt")
        stochastic_kwargs.pop("solver")

    if exists(sharding):
        X, Q, A = eqx.filter_shard((X, Q, A), sharding)

    sample_fn = lambda q, a, key: flow.sample_stochastic(
        q, a, key, x_shape, g_scale=0.1, **stochastic_kwargs
    )

    sample_keys = jr.split(key, X.shape[0])
    x_sample = eqx.filter_vmap(sample_fn)(Q, A, sample_keys)

    n_side = int(jnp.sqrt(len(x_sample)))

    x_sample = rearrange(
        x_sample, 
        "(r c) s h w -> (r h) (c w) s", 
        s=n_channels, r=n_side, c=n_side, h=n_pix, w=n_pix
    )
    X = rearrange(
        X, 
        "(r c) s h w -> (r h) (c w) s", 
        s=n_channels, r=n_side, c=n_side, h=n_pix, w=n_pix
    )

    fig, axs = plt.subplots(1, 2, dpi=200, figsize=(8., 4.))
    ax = axs[0]
    im = ax.imshow(jnp.clip(x_sample, 0., 1.), cmap=cmap, **vs)
    ax = axs[1]
    im = ax.imshow(jnp.clip(X, 0., 1.), cmap=cmap, **vs)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


"""
    Datasets & loaders
"""


class _AbstractDataLoader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, dataset, *, key):
        pass

    def __iter__(self):
        raise RuntimeError("Use `.loop` to iterate over the data loader.")

    @abc.abstractmethod
    def loop(self, batch_size):
        pass


class InMemoryDataLoader(_AbstractDataLoader):
    def __init__(
        self, X: Array, Q: Array = None, A: Array = None, *, key: Key
    ):
        self.X = X 
        self.Q = Q 
        self.A = A 
        self.key = key

    def loop(
        self, batch_size: int, *, key: Optional[Key] = None
    ) -> Generator[Array, Array | None, Array | None]:
        dataset_size = self.X.shape[0]
        if batch_size > dataset_size:
            raise ValueError("Batch size larger than dataset size")

        key = key if exists(key) else self.key
        indices = jnp.arange(dataset_size)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, indices)
            start = 0
            end = batch_size
            while end < dataset_size:
                batch_perm = perm[start:end]
                yield (
                    self.X[batch_perm], 
                    self.Q[batch_perm] if exists(self.Q) else None, 
                    self.A[batch_perm] if exists(self.A) else None 
                )
                start = end
                end = start + batch_size


class GrainDataLoader(_AbstractDataLoader):
    def __init__(
        self, 
        dataset: grain.RandomAccessDataSource, 
        n_workers: int, 
        key: Key
    ):
        self.dataset = dataset
        self.key = key
        self.n_workers = n_workers

    def loop(
        self, batch_size: int, num_workers: int = 4,  *, key: Optional[Key] = None
    ) -> Generator[Array, Array | None, Array | None]:

        if isinstance(self.key, Key[Scalar, ""]):
            seed = jnp.sum(jr.key_data(self.key))

        sampler = grain.SequentialSampler(
            num_records=len(self.dataset),
            shard_options=grain.NoSharding(),
            seed=seed
        ) 
        loader = grain.DataLoader(
            data_source=self.dataset,
            sampler=sampler,
            worker_count=self.n_workers,
            operations=[
                AddChannel(), 
                Normalize(), 
                # Rotate(), 
                grain.Batch(batch_size)
            ]
        )
        while True:
            for arrays in loader:
                yield arrays


@dataclass
class Dataset:
    name: str
    train_dataloader: InMemoryDataLoader | GrainDataLoader
    valid_dataloader: InMemoryDataLoader | GrainDataLoader
    data_shape: Tuple[int]
    context_shape: Optional[Tuple[int]] = None
    parameter_dim: Optional[int] = None


class FFHQDataset(grain.RandomAccessDataSource):
    def __init__(
        self,
        image_folder: str | Path,
        image_size: int,
        exts: List[str] = ['jpg', 'jpeg', 'png', 'tiff'],
        idx: Optional[List] = None,
        convert_image_to = None
    ):
        super().__init__()

        if isinstance(image_folder, str):
            image_folder = Path(image_folder)

        assert image_folder.is_dir() 

        self.image_folder = image_folder
        self.image_size = image_size

        image_paths = [
            p for ext in exts for p in image_folder.glob(f'**/*.{ext}')
        ]

        if exists(idx):
            self.image_paths = [image_paths[i] for i in idx]
        else:
            self.image_path = image_paths

        def convert_image_to_fn(img_type, image):
            if image.mode == img_type:
                return image
            return image.convert(img_type)

        if exists(convert_image_to):
            maybe_convert_fn = partial(convert_image_to_fn, convert_image_to)  
        else: 
            maybe_convert_fn = identity

        self.transform = lambda x: maybe_convert_fn(x)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        x = jnp.asarray(
            Image.open(self.image_paths[index]).convert("RGB")
        )
        x = x.astype(jnp.float32)
        # X, Q, A convention
        return (
            self.transform(x), 
            None, 
            None
        ) 


@dataclass
class AddChannel(grain.MapTransform):
    def map(self, xqa: Tuple[Array]) -> Tuple[Array]:
        def _maybe_add_channel(u):
            if u.ndim == 3:
                return u
            else:
                return u[jnp.newaxis, ...]
        x, q, a = xqa
        return (
            _maybe_add_channel(x), 
            _maybe_add_channel(q) if exists(q) else q, 
            a
        )


@dataclass
class Normalize(grain.MapTransform):
    def map(self, xqa: Tuple[Array]) -> Tuple[Array]:
        x, q, a = xqa
        x = x / 255.
        q = q / 255. if exists(q) else q
        return x, q, a


@dataclass
class Rotate(grain.RandomMapTransform):
    def random_map(self, xqa: Tuple[Array], rng: np.random.Generator) -> Tuple[Array]:
        x, q, a = xqa
        if exists(rng):
            k = rng.integers(0, 4)
            x = jnp.rot90(x, k=k, axes=(1, 2))
            q = jnp.rot90(q, k=k, axes=(1, 2)) if exists(q) else q
        return x, q, a


def ffhq(
    key: Key, 
    n_pix: int = 128, 
    split: float = 0.5,
    n_data: int = 100,
    use_grain: bool = False,
    n_workers: int = 4,
    data_dir: str | Path = "./",
    image_file_ext: str = ".png"
) -> Dataset:

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        
    key_train, key_valid = jr.split(key)

    assert n_data <= 70_000
    assert n_pix <= 128

    data_shape = (3, n_pix, n_pix)

    train_idx, valid_idx = np.split(
        np.arange(n_data), [int(split * n_data)]
    )

    if use_grain:
        train_dataloader = GrainDataLoader(
            FFHQDataset(
                image_folder=data_dir, 
                image_size=n_pix, 
                idx=train_idx
            ),
            n_workers=n_workers,
            key=key_train
        )
        valid_dataloader = GrainDataLoader(
            FFHQDataset(
                image_folder=data_dir, 
                image_size=n_pix, 
                idx=valid_idx
            ),
            n_workers=n_workers,
            key=key_valid
        )
    else:
        image_paths = [
            data_dir / p for p in data_dir.glob("*{}".format(image_file_ext))
        ][:n_data]

        X = jnp.asarray(
            [
                jax.image.resize(
                    np.asarray(Image.open(image_path).convert("RGB")),
                    reversed(data_shape),
                    method="bilinear"
                )
                for image_path in image_paths
            ]
        )
        X = jnp.transpose(X, (0, 3, 1, 2))
        X = X.astype(jnp.float32) / 255.
        X = (X - X.min()) / (X.max() - X.min())

        print("DATA:", X.shape, X.dtype, X.min(), X.max())

        # Scaler(...) doesn't implement forward scaling in this Loader?
        train_dataloader = InMemoryDataLoader(X[train_idx], key=key_train)
        valid_dataloader = InMemoryDataLoader(X[valid_idx], key=key_valid)

        del X

    return Dataset(
        name="ffhq",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=data_shape
    )


def get_config():
    config = ConfigDict()
    config.seed                   = 0
    config.run_dir                = "/project/ls-gruen/users/jed.homer/zurich/runs/ffhq_{}/"
    config.data_dir               = "/project/ls-gruen/users/jed.homer/1pt_pdf/little_studies/sgm_lib/datasets/ffhq/thumbnails128x128/"

    config.data = data = ConfigDict()
    data.n_pix                    = 64 
    data.n_channels               = 3
    data.n_data                   = 10_000
    data.split                    = 0.9
    data.use_grain                = False

    config.model_type = "DiT"

    if config.model_type == "UNet":
        config.model_constructor = UNet

        config.model = model = ConfigDict()
        model.dim                     = 128
        model.channels                = data.n_channels
        model.q_channels              = None
        model.a_dim                   = None
        model.dim_mults               = (1, 2, 4, 8)
        model.learned_sinusoidal_cond = True
        model.random_fourier_features = True
        model.attn_dim_head           = 64
        model.dropout                 = 0.1

    if config.model_type == "DiT":
        config.model_constructor = DiT

        config.model = model = ConfigDict()
        model.img_size                = config.data.n_pix
        model.channels                = config.data.n_channels
        model.patch_size              = 2
        model.embed_dim               = 128
        model.q_dim                   = None 
        model.a_dim                   = None
        model.depth                   = 8
        model.n_heads                 = 4

    config.train = train = ConfigDict()
    train.reload                  = False # Auto-load from config.run_dir
    train.n_steps                 = 400_000
    train.n_batch                 = 4 * jax.local_device_count()
    train.patience                = None
    train.lr                      = 5e-5
    train.optimiser               = optax.adamw
    train.accumulate_gradients    = True
    train.n_minibatches           = 4
    train.use_ema                 = True 
    train.ema_rate                = 0.9995
    train.q_as_x_1                = False # Use q samples as x_1 samples
    train.noise_schedule          = "cosine"

    config.train.policy = policy = ConfigDict()
    policy.param_dtype            = jnp.float32
    policy.compute_dtype          = jnp.bfloat16
    policy.output_dtype           = jnp.float32 #bfloat16

    config.train.sampling = sampling = ConfigDict() 
    sampling.n_sample             = 4 
    sampling.solver               = dfx.Euler()
    sampling.t0                   = 0.
    sampling.t1                   = 1.
    sampling.dt                   = 0.01

    config.run_dir = config.run_dir.format(data.n_pix)

    return config


if __name__ == "__main__":

    config = get_config()

    key = jr.key(config.seed)
    key, key_data, key_train, key_model = jr.split(key, 4)

    dataset = ffhq(
        key_data, 
        n_data=config.data.n_data,
        n_pix=config.data.n_pix, 
        split=config.data.split,
        use_grain=config.data.use_grain, # Bug...
        data_dir=config.data_dir
    )

    v = config.model_constructor(**config.model, key=key_model) 

    flow = RectifiedFlow(v)

    policy = Policy(**config.train.policy)

    sharding, replicated_sharding = get_shardings()

    ema_v, metrics = train(
        key_train,
        # Model
        flow,
        # Data
        dataset,
        # Training
        n_steps=config.train.n_steps,
        n_batch=config.train.n_batch,
        n_minibatches=config.train.n_minibatches,
        optimiser=config.train.optimiser,
        lr=config.train.lr,
        patience=config.train.patience,
        use_ema=config.train.use_ema,
        q_as_x_1=config.train.q_as_x_1, # Force this to be corrected in sampling, possible soln_kwarg?
        # FM
        t0=config.train.sampling.t0,
        t1=config.train.sampling.t1,
        dt=config.train.sampling.dt,
        noise_schedule=config.train.noise_schedule,
        solver=config.train.sampling.solver,
        ema_rate=config.train.ema_rate, 
        # Shardings, sampling etc.
        n_sample=config.train.sampling.n_sample,
        reload_opt_and_model=config.train.reload,
        sharding=sharding,
        replicated_sharding=replicated_sharding,
        run_dir=config.run_dir 
    )