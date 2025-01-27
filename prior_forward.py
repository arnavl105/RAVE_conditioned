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

def generate_audio(input_path, output_folder, model_checkpoint_path, pretrained_vae_path ,chunk_size):
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

    # Load audio file
    audio, sr = librosa.load(input_path)

    time_steps = math.ceil(len(audio) / chunk_size)

    x = torch.randn_like(torch.zeros(1, (LATENT_DIM + 1), time_steps))
    print("x shape:", x.shape)
    x = model.quantized_normal.encode(model.diagonal_shift(x))
    z  = model.generate(x)
    z = model.diagonal_shift.inverse(model.quantized_normal.decode(z))
    
    #remove onset channel
    z = z[:, :-1, :]
    print("z shape:", z.shape)
    y = model.decode(z)
    output = y.reshape(-1).detach().numpy()

    # Save the output audio to the output folder using soundfile
    output_filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, output_filename)
    sf.write(output_path, output, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using specified parameters.")
    parser.add_argument("--input_folder", type=str, help="Path to the input audio file.", required=True)
    parser.add_argument("--output_folder", type=str, default="forward", help="Path to save the output processed audio.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the model checkpoint.", required=True)
    parser.add_argument("--pretrained_vae_path", type=str,  help="Path to the pre-trained VAE checkpoint.", required=True)
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for processing the audio.")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Iterate over all the .wav files in the input folder and process them
    for audio_file in tqdm(os.listdir(args.input_folder)):
        if audio_file.endswith('.wav'):
            input_path = os.path.join(args.input_folder, audio_file)

            generate_audio(
                input_path=input_path,
                output_folder=args.output_folder,
                model_checkpoint_path=args.model_checkpoint_path,
                pretrained_vae_path=args.pretrained_vae_path,
                chunk_size=args.chunk_size,
            )
