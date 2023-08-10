import os
import torch
import librosa
import numpy as np
import soundfile as sf
import argparse
import math
import torch.nn as nn
from prior import Prior
from prior.core import QuantizedNormal  
from tqdm import tqdm

def process_audio(input_path, output_path, model_checkpoint_path, pretrained_vae_path ,chunk_size):
    # Load the model from the checkpoint
    RESOLUTION = 32
    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10

    quantized_normal = QuantizedNormal(RESOLUTION)

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

    # Split audio into chunks
    chunks = librosa.util.frame(audio, frame_length=chunk_size, hop_length=chunk_size).T

    # Initialize output array
    output = []

    # Pass each chunk through the model and reconstruct the signal
    for i in tqdm(range(chunks.shape[0])):
        # Get input chunk
        chunk = chunks[i:i+1, :]
        #chunk = np.expand_dims(chunk, 0)
        x = torch.from_numpy(chunk)

        # Pass chunk through model and get output
        with torch.no_grad():
            x_encoded = model.encode(x)
            x_encoded = quantized_normal.encode(x_encoded.unsqueeze(0))
            y = model.forward(x_encoded)
            y = quantized_normal.decode(y)
            y = model.decode(y)

        # Store output
        
        output.append(y.detach().numpy())

    # Reconstruct signal from output chunks
    output = np.concatenate(output, axis=-1)[0]

    # Save the output audio to the output folder using soundfile
    sf.write(output_path, output.T, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using specified parameters.")
    parser.add_argument("--input_path", type=str, help="Path to the input audio file.", required=True)
    parser.add_argument("--output_path", type=str, default="output.wav", help="Path to save the output processed audio.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the model checkpoint.", required=True)
    parser.add_argument("--pretrained_vae_path", type=str,  help="Path to the pre-trained VAE checkpoint.", required=True)
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for processing the audio.")

    args = parser.parse_args()

    process_audio(
        input_path=args.input_path,
        output_path=args.output_path,
        model_checkpoint_path=args.model_checkpoint_path,
        pretrained_vae_path=args.pretrained_vae_path,
        chunk_size=args.chunk_size,
    )
