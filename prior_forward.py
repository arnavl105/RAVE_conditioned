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

def process_audio(input_path, output_path, model_checkpoint_path, pretrained_vae_path ,chunk_size):
    # Load the model from the checkpoint
    RESOLUTION = 32
    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10

    # quantized_normal = QuantizedNormal(RESOLUTION)
    # diagonal_shift = DiagonalShift()

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

    # # Split audio into chunks
    # chunks = librosa.util.frame(audio, frame_length=chunk_size, hop_length=2048).T

    # # Initialize output array
    # output = []

    # # Pass each chunk through the model and reconstruct the signal
    # for i in tqdm(range(chunks.shape[0])):
    #     # Get input chunk
    #     chunk = chunks[None, i:i+1, :]
    #     x = torch.from_numpy(chunk)

    #     # Pass chunk through model and get output
    #     with torch.no_grad():
    #         # x_encoded = model.encode(x)
    #         # x_encoded = model.quantized_normal.encode(x_encoded)
    #         # y = model.forward(x_encoded)
    #         # y = model.post_process_prediction(y)
    #         # y = model.quantized_normal.decode(y)
    #         # y = y[:, :, -1:None]
    #         # y = model.decode(y)
    #         x_encoded = model.encode(x)
    #         print(x_encoded.shape)
    #         x_encoded = model.quantized_normal.encode(x_encoded)
    #         z = model.generate(x_encoded)

    #         z = model.quantized_normal.decode(z)

    #         y = model.decode(z)
    #         # Store output
        
    #         output.append(y.detach().numpy())

    # # Reconstruct signal from output chunks
    # output = np.concatenate(output, axis=-1)[0]
    x_encoded = model.encode(torch.from_numpy(audio)[None, None, :])
    x_encoded = model.quantized_normal.encode(x_encoded)
    preds = []
    for i in tqdm(range(x_encoded.shape[-1] - 1)):
        start = None

        pred = model.forward(x_encoded[..., start:i + 1])

        pred = model.post_process_prediction(pred)
        preds.append(pred[..., -1:None])

    preds = torch.cat(preds, dim=0)
    print("preds shape", preds.shape)
    z = model.quantized_normal.decode(preds)

    y = model.decode(z)
    # Store output

    print("y shape", y.shape)
    y=y.reshape(1,-1)
    print("y shape post reshape", y.shape)

    output = y.squeeze().detach().numpy()


    # Save the output audio to the output folder using soundfile
    sf.write(output_path, output, sr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio using specified parameters.")
    parser.add_argument("--input_path", type=str, help="Path to the input audio file.", required=True)
    parser.add_argument("--output_path", type=str, default="output.wav", help="Path to save the output processed audio.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the model checkpoint.", required=True)
    parser.add_argument("--pretrained_vae_path", type=str,  help="Path to the pre-trained VAE checkpoint.", required=True)
    parser.add_argument("--chunk_size", type=int, default=2048*8, help="Chunk size for processing the audio.")

    args = parser.parse_args()

    process_audio(
        input_path=args.input_path,
        output_path=args.output_path,
        model_checkpoint_path=args.model_checkpoint_path,
        pretrained_vae_path=args.pretrained_vae_path,
        chunk_size=args.chunk_size,
    )
