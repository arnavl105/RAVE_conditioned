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

def calculate_ce(input_path, model_checkpoint_path, pretrained_vae_path ,chunk_size):
    # Load the model from the checkpoint
    RESOLUTION = 32
    RES_SIZE = 544
    SKP_SIZE = 272
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10
    LATENT_DIM = 16

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

    losses = []
    # Load audio file
    audio, sr = librosa.load(input_path)

    # time_steps = math.ceil(len(audio) / chunk_size)
    #seperate audio into chunks
    chunks = librosa.util.frame(audio, frame_length=chunk_size, hop_length=2048)

    #iterate over chunks
    for chunk in chunks.T:
        onset_strength = librosa.onset.onset_strength(y=chunk, sr=sr, hop_length=2048, n_fft=2048)
        x = model.encode(torch.from_numpy(chunk)[None, None, :])
        x = model.quantized_normal.encode(x)
        onset_quantized = model.quantized_normal.encode_onset(torch.from_numpy(onset_strength)[None, None, 1:])
        x = torch.cat((x, onset_quantized), dim=1)
        pred = model.forward(x)

        x = torch.argmax(model.split_classes(x[...,1:]), -1)
        pred = model.split_classes(pred[...,:-1])

        #slice onset off of pred and x
        pred = pred[:, :-1, :, :]
        x = x[:, :-1,  :]   

        with torch.no_grad():
            loss = nn.functional.cross_entropy(
                pred.reshape(-1, model.quantized_normal.resolution),
                x.reshape(-1),
            )
        losses.append(loss.item())

    print("Mean cross-entropy loss {} : {}".format(input_path, np.mean(losses)))
    return np.mean(losses)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using specified parameters.")
    parser.add_argument("--input_folder", type=str, help="Path to the input audio file.", required=True)
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the model checkpoint.", required=True)
    parser.add_argument("--pretrained_vae_path", type=str,  help="Path to the pre-trained VAE checkpoint.", required=True)
    parser.add_argument("--chunk_size", type=int, default=131072, help="Chunk size for processing the audio.")

    args = parser.parse_args()
    losses = []

    # Iterate over all the .wav files in the input folder and process them
    for audio_file in tqdm(os.listdir(args.input_folder)):
        if audio_file.endswith('.wav'):
            input_path = os.path.join(args.input_folder, audio_file)

            loss = calculate_ce(
                input_path=input_path,
                model_checkpoint_path=args.model_checkpoint_path,
                pretrained_vae_path=args.pretrained_vae_path,
                chunk_size=args.chunk_size,
            )

            losses.append(loss)

    print("Mean cross-entropy loss: {}".format(np.mean(losses)))
