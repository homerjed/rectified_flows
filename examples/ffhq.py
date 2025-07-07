
from pathlib import Path
from typing import Optional, List
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr 
import equinox as eqx
import diffrax as dfx
from jaxtyping import Key, jaxtyped
import optax
import grain.python as grain
import numpy as np
from ml_collections import ConfigDict

from rectified_flows import (
    RectifiedFlow, UNet, DiT, 
    Dataset, GrainDataLoader, InMemoryDataLoader, 
    Policy, train, exists, identity, get_shardings
)

try:
    from beartype import beartype as typechecker
    typecheck = jaxtyped(typechecker=typechecker) 
except ImportError:
    typecheck = lambda x: x


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
    policy.output_dtype           = jnp.float32 

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