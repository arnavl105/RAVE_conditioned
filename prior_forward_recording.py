import os
import torch
import librosa
import numpy as np
import soundfile as sf
import argparse
import math
import torch.nn as nn
from prior import Prior
from prior.core import QuantizedNormal, DiagonalShift
from tqdm import tqdm


def generate(model, x):
    for i in tqdm(range(x.shape[-1] - 1)):
        start = None

        pred = model.forward(x[..., start:i + 1])

        pred = pred[..., -1:]

        pred = model.post_process_prediction(pred, argmax=False)

        x[:, :-32, i + 1:i + 2] = pred[:, :-32, :]
    return x

def generate_audio(input_path, output_path, model_checkpoint_path, pretrained_vae_path ,chunk_size):
    # Load the model from the checkpoint
    RESOLUTION = 32
    RES_SIZE = 544
    SKP_SIZE = 272
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10
    LATENT_DIM = 16

    MODEL_SR = 44100

    model = Prior(
        resolution=RESOLUTION,
        res_size=RES_SIZE,
        skp_size=SKP_SIZE,
        kernel_size=KERNEL_SIZE,
        cycle_size=CYCLE_SIZE,
        n_layers=N_LAYERS,
        pretrained_vae=pretrained_vae_path,
    )

    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # set the model to evaluation mode

    # Load audio file
    audio, sr = librosa.load(input_path)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SR)
    onset_strength = librosa.onset.onset_strength(y=audio, sr=MODEL_SR, hop_length=2048, n_fft=2048)

    x = torch.from_numpy(audio)[None, None, :]
    onset_strength = torch.from_numpy(onset_strength)[None, None, :]
    x = torch.randn_like(model.encode(x))
    x = model.quantized_normal.encode(x)
    onset_strength = model.quantized_normal.encode_onset(onset_strength)
    x = torch.cat([x, onset_strength], dim=1)
    z  = generate(model, x)
    z = model.quantized_normal.decode(z)
    
    #remove onset channel
    z = z[:, :-1, :]
    y = model.decode(z)
    output = y.reshape(-1).detach().numpy()

    sf.write(output_path, output, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using specified parameters.")
    parser.add_argument("--input_path", type=str, help="Path to the input audio file or dir", required=True)
    parser.add_argument("--output_path", type=str, default="forward", help="Path to save the output processed audio.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the model checkpoint.", required=True)
    parser.add_argument("--pretrained_vae_path", type=str,  help="Path to the pre-trained VAE checkpoint.", required=True)
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for processing the audio.")
    parser.add_argument("--is_dir", type=bool, default=False, help="Whether the input path is a directory or not.")

    args = parser.parse_args()

    if args.is_dir:
        for file in os.listdir(args.input_path):
            input_path = os.path.join(args.input_path, file)
            output_path = os.path.join(args.output_path, file)
            generate_audio(
                input_path=input_path,
                output_path=output_path,
                model_checkpoint_path=args.model_checkpoint_path,
                pretrained_vae_path=args.pretrained_vae_path,
                chunk_size=args.chunk_size,
            )
    else:
        generate_audio(
            input_path=args.input_path,
            output_path=args.output_path,
            model_checkpoint_path=args.model_checkpoint_path,
            pretrained_vae_path=args.pretrained_vae_path,
            chunk_size=args.chunk_size,
        )
