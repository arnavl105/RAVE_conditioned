import pytorch_lightning as pl
import torch
import os
import sys

import rave
import rave.core
import rave.dataset
import gin
from prior.model import Model as Prior
from absl import flags
from absl import app

from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, help='Name of the run', required=True)
flags.DEFINE_string('db_path',
                    None,
                    help='Preprocessed dataset path',
                    required=True)
flags.DEFINE_integer('max_steps',
                     6000000,
                     help='Maximum number of training steps')
flags.DEFINE_integer('val_every', 10000, help='Checkpoint model every n steps')
flags.DEFINE_integer('n_signal',
                     2048,
                     help='Number of audio samples to use during training')
flags.DEFINE_integer('batch', 8, help='Batch size')
flags.DEFINE_string('ckpt',
                    None,
                    help='Path to previous checkpoint of the run')
flags.DEFINE_string('pretrained_vae',
                    None,
                    help='Path to trained VAE')
flags.DEFINE_multi_string('override', default=[], help='Override gin binding')
flags.DEFINE_integer('workers',
                     default=8,
                     help='Number of workers to spawn for dataset loading')
flags.DEFINE_multi_integer('gpu', default=None, help='GPU to use')
flags.DEFINE_bool('progress',
                  default=True,
                  help='Display training progress bar')

def main(argv):
    # Load the model
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    RESOLUTION = 32 
    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10

    model = Prior(
        resolution=RESOLUTION,
        res_size=RES_SIZE,
        skp_size=SKP_SIZE,
        kernel_size=KERNEL_SIZE,
        cycle_size=CYCLE_SIZE,
        n_layers=N_LAYERS,
        pretrained_vae=FLAGS.pretrained_vae,
    )

    dataset = rave.dataset.get_dataset(FLAGS.db_path,
                                       model.sr,
                                       FLAGS.n_signal,
                                       conditioning=True,
                                       )
    train, val = rave.dataset.split_dataset(dataset, 98)
    num_workers = FLAGS.workers

    train = DataLoader(train,
                       FLAGS.batch,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, FLAGS.batch, False, num_workers=num_workers)

    print("train shape:", len(train))

    print("Dataset loaded")

    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(monitor="validation",
                                                         filename="best")
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    val_check = {}
    if len(train) >= FLAGS.val_every:
        val_check["val_check_interval"] = FLAGS.val_every
    else:
        nepoch = FLAGS.val_every // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    RUN_NAME = f"{FLAGS.name}" 

    os.makedirs(os.path.join("prior_runs", RUN_NAME), exist_ok=True)

    if FLAGS.gpu == [-1]:
        gpu = 0
    else:
        gpu = FLAGS.gpu or rave.core.setup_gpu()

    print('selected gpu:', gpu)

    accelerator = None
    devices = None
    if FLAGS.gpu == [-1]:
        pass
    elif torch.cuda.is_available():
        accelerator = "cuda"
        devices = FLAGS.gpu or rave.core.setup_gpu()
    elif torch.backends.mps.is_available():
        print(
            "Training on mac is not available yet. Use --gpu -1 to train on CPU (not recommended)."
        )
        exit()
        accelerator = "mps"
        devices = 1

    callbacks = [
        validation_checkpoint,
        last_checkpoint,
        rave.core.LoggerCallback(rave.core.ProgressLogger(RUN_NAME)),
    ]

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            "prior_runs",
            name=RUN_NAME,
        ),
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        max_epochs=100000,
        max_steps=FLAGS.max_steps,
        profiler="simple",
        enable_progress_bar=FLAGS.progress,
        **val_check,
    )
    
    run = rave.core.search_for_run(FLAGS.ckpt)
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step
    # Train the model
    trainer.fit(model, train, val, ckpt_path=run)

if __name__ == "__main__":
    app.run(main)
