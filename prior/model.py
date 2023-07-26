import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm

from .residual_block import ResidualBlock
from .core import DiagonalShift, QuantizedNormal
from rave import pqmf
from rave import core

import cached_conv as cc

class Model(pl.LightningModule):

    def __init__(self, resolution, res_size, skp_size, kernel_size, cycle_size,
                 n_layers, pretrained_vae):
        super().__init__()
        self.save_hyperparameters()
        
        self.diagonal_shift = DiagonalShift()
        self.quantized_normal = QuantizedNormal(resolution)

        self.synth = torch.jit.load(pretrained_vae)

        self.sr = 44100 
        data_size = 1

        N_BAND = 16

        self.weights = {'fullband_spectral_distance': 1.0,}

        self.pqmf = pqmf.CachedPQMF(attenuation=100, n_band=N_BAND) 

        multiscale_stft = core.MultiScaleSTFT(scales=[2048, 1024, 512, 256, 128], sample_rate=44100, magnitude=True)

        self.audio_distance = core.AudioDistanceV1(multiscale_stft=lambda: multiscale_stft, log_epsilon=1e-7)

        self.multiband_audio_distance = core.AudioDistanceV1(multiscale_stft=lambda: multiscale_stft, log_epsilon=1e-7)


        self.warmed_up = True
        self.automatic_optimization = False

        self.pre_net = nn.Sequential(
            cc.Conv1d(
                resolution * data_size,
                res_size,
                kernel_size,
                padding=cc.get_padding(kernel_size, mode="causal"),
                groups=data_size,
            ),
            nn.LeakyReLU(.2),
        )

        self.residuals = nn.ModuleList([
            ResidualBlock(
                res_size,
                skp_size,
                kernel_size,
                2**(i % cycle_size),
            ) for i in range(n_layers)
        ])

        self.post_net = nn.Sequential(
            cc.Conv1d(skp_size, skp_size, 1),
            nn.LeakyReLU(.2),
            cc.Conv1d(
                skp_size,
                resolution * data_size,
                1,
                groups=data_size,
            ),
        )

        self.data_size = data_size

        self.val_idx = 0

    def configure_optimizers(self):
        p = []
        p.extend(list(self.pre_net.parameters()))
        p.extend(list(self.residuals.parameters()))
        p.extend(list(self.post_net.parameters()))
        return torch.optim.Adam(p, lr=1e-4)

    @torch.no_grad()
    def encode(self, x):
        self.synth.eval()
        return self.synth.encode(x)

    @torch.no_grad()
    def decode(self, z):
        flipped_z = z.permute(1, 0, 2)
        self.synth.eval()
        return self.synth.decode(flipped_z)

    def forward(self, x, onset_strength=None):
        res = self.pre_net(x)
        skp = torch.tensor(0.).to(x)
        for layer in self.residuals:
            res, skp = layer(res, skp, onset_strength)
        x = self.post_net(skp)
        y = self.decode(x).detach()
        return y

    @torch.no_grad()
    def generate(self, x, argmax: bool = False):
        for i in tqdm(range(x.shape[-1] - 1)):
            if cc.USE_BUFFER_CONV:
                start = i
            else:
                start = None

            pred = self.forward(x[..., start:i + 1])
            if not cc.USE_BUFFER_CONV:
                pred = pred[..., -1:]

            pred = self.post_process_prediction(pred, argmax=argmax)

            x[..., i + 1:i + 2] = pred
        return x

    def split_classes(self, x):
        # B x D*C x T
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], x.shape[1], self.data_size, -1)
        x = x.permute(0, 2, 1, 3)  # B x D x T x C
        return x

    def post_process_prediction(self, x, argmax: bool = False):
        x = self.split_classes(x)
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        if argmax:
            x = torch.argmax(x, -1)
        else:
            x = torch.softmax(x - torch.logsumexp(x, -1, keepdim=True), -1)
            x = torch.multinomial(x, 1, True).squeeze(-1)

        x = x.reshape(shape[0], shape[1], shape[2])
        x = self.quantized_normal.to_stack_one_hot(x)
        return x

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        audio = batch[:, 0, :]
        onset_strength = batch[:, 1, :]

        x_multiband = self.pqmf(audio[None, :, :])

        x = self.encode(audio)
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        y_multiband = self.forward(x, onset_strength)

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}
        
        fullband_distance = self.audio_distance(audio[None, :, :], y_multiband)
        for k, v in fullband_distance.items():
            distances[f'fullband_{k}'] = v


        loss = {}
        loss.update(distances)

        optimizer.zero_grad()
        loss_value = 0.
        dummy_param = next(self.parameters())
        for k, v in loss.items():
            loss_value += v + (dummy_param.sum() * 0.)
        self.manual_backward(loss_value)
        optimizer.step()

        print("loss", loss_value)
        self.log_dict(loss)

    
    def validation_step(self, batch, batch_idx):
        audio = batch[:, 0, :]
        onset_strength = batch[:, 1, :]

        x = self.encode(audio)
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        y = self.forward(x, onset_strength)

        distance = self.audio_distance(audio[None, :, :], y)

        full_distance = sum(distance.values())

        self.log("validation", full_distance)

        return audio

    def validation_epoch_end(self, out):
        x = torch.randn_like(self.encode(out[0]))
        x = self.quantized_normal.encode(self.diagonal_shift(x))
        z = self.generate(x)
        z = self.diagonal_shift.inverse(self.quantized_normal.decode(z))
        y = self.decode(z)
        self.logger.experiment.add_audio(
            "generation",
            y.reshape(-1),
            self.val_idx,
            44100,
        )
        self.val_idx += 1
