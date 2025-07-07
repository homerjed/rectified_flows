
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr 
import diffrax as dfx
from jaxtyping import PRNGKeyArray, jaxtyped
import optax
from ml_collections import ConfigDict
from datasets import load_dataset
from beartype import beartype as typechecker

from rectified_flows import (
    RectifiedFlow, DiT, 
    Dataset, InMemoryDataLoader, 
    Policy, train, 
    exists, get_shardings
)

typecheck = jaxtyped(typechecker=typechecker) 


@typecheck
def cifar10(
    key: PRNGKeyArray,
    img_size: int, 
    split: float = 0.9,
    use_y: bool = False,
    use_integer_labels: bool = True,
    *,
    dataset_path: Optional[str | Path] = None
) -> Dataset:
    
    key_train, key_valid = jr.split(key)

    target_type = jnp.int32 if use_integer_labels else jnp.float32

    dataset = load_dataset("cifar10").with_format("jax")

    data = jnp.concatenate([dataset["train"]["img"], dataset["test"]["img"]])
    data = data / 255.
    data = data.transpose(0, 3, 1, 2)
    data = data.astype(jnp.float32)

    targets = jnp.concatenate([dataset["train"]["label"], dataset["test"]["label"]])
    targets = targets[:, jnp.newaxis]
    targets = targets.astype(target_type)

    data = jax.image.resize(
        data, 
        shape=(data.shape[0], 3, img_size, img_size),
        method="bilinear"
    )

    a, b = jnp.min(data), jnp.max(data)
    print(a, b)
    # data = 2. * (data - a) / (b - a) - 1.

    print(
        "DATA:\n> {:.3E} {:.3E} {}\n> {} {}".format(
            data.min(), data.max(), data.dtype, 
            data.shape, targets.shape if exists(targets) else None
        )
    )

    n_train = int(split * data.shape[0])
    x_train, x_valid = jnp.split(data, [n_train])

    if use_y:
        y_train, y_valid = jnp.split(targets, [n_train])
    else:
        y_train = y_valid = None

    train_dataloader = InMemoryDataLoader(x_train, A=y_train, key=key_train)
    valid_dataloader = InMemoryDataLoader(x_valid, A=y_valid, key=key_valid)

    return Dataset(
        name="cifar10",
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        data_shape=x_train.shape[1:]
    )


def get_config():

    config = ConfigDict()
    config.seed                   = 0
    config.run_dir                = Path.cwd() / "runs" / "cifar10"

    config.data = data = ConfigDict()
    data.n_pix                    = 32
    data.n_channels               = 3
    data.n_data                   = 10_000
    data.split                    = 0.9
    data.use_grain                = False

    config.model_type = "DiT"
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
    policy.output_dtype           = jnp.float32 

    config.train.sampling = sampling = ConfigDict() 
    sampling.n_sample             = 4 
    sampling.solver               = dfx.Euler()
    sampling.t0                   = 0.
    sampling.t1                   = 1.
    sampling.dt                   = 0.01

    return config


if __name__ == "__main__":

    config = get_config()

    key = jr.key(config.seed)
    key, key_data, key_train, key_model = jr.split(key, 4)

    dataset = cifar10(
        key_data, 
        img_size=config.data.n_pix, 
        split=config.data.split
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