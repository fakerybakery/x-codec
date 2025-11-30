import os
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder_vocos import CodecDecoderVocos
from argparse import ArgumentParser
from time import time
import torch.nn as nn
from collections import OrderedDict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='test_audio/input_test')
    parser.add_argument('--ckpt', type=str, default='ckpt/epoch=4-step=1400000.ckpt')
    parser.add_argument('--output-dir', type=str, default='test_audio/output_test')
    parser.add_argument('--vq-dim', type=int, default=1024, help='VQ dimension (1024 for no-semantic, 2048 for original)')

    args = parser.parse_args()
    sr = 16000

    print(f'Load codec ckpt from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ckpt = ckpt['state_dict']

    state_dict = ckpt
    # Extract and filter model components
    filtered_state_dict_codec = OrderedDict()
    filtered_state_dict_gen = OrderedDict()
    filtered_state_dict_fc_post_a = OrderedDict()
    filtered_state_dict_fc_prior = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith('CodecEnc.'):
            new_key = key[len('CodecEnc.'):]
            filtered_state_dict_codec[new_key] = value
        elif key.startswith('generator.'):
            new_key = key[len('generator.'):]
            filtered_state_dict_gen[new_key] = value
        elif key.startswith('fc_post_a.'):
            new_key = key[len('fc_post_a.'):]
            filtered_state_dict_fc_post_a[new_key] = value
        elif key.startswith('fc_prior.'):
            new_key = key[len('fc_prior.'):]
            filtered_state_dict_fc_prior[new_key] = value

    # Load encoder
    encoder = CodecEncoder()
    encoder.load_state_dict(filtered_state_dict_codec)
    encoder = encoder.eval().cuda()

    # Load decoder
    decoder = CodecDecoderVocos(vq_dim=args.vq_dim)
    decoder.load_state_dict(filtered_state_dict_gen)
    decoder = decoder.eval().cuda()

    # Load projection layers
    fc_post_a = nn.Linear(args.vq_dim, 1024)
    fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
    fc_post_a = fc_post_a.eval().cuda()

    fc_prior = nn.Linear(1024, args.vq_dim)
    fc_prior.load_state_dict(filtered_state_dict_fc_prior)
    fc_prior = fc_prior.eval().cuda()

    wav_dir = args.output_dir
    os.makedirs(wav_dir, exist_ok=True)

    # Find all audio files
    wav_paths = glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
    flac_paths = glob(os.path.join(args.input_dir, '**', '*.flac'), recursive=True)
    mp3_paths = glob(os.path.join(args.input_dir, '**', '*.mp3'), recursive=True)

    wav_paths = wav_paths + flac_paths + mp3_paths
    print(f'Found {len(wav_paths)} audio files in {args.input_dir}')

    st = time()
    for wav_path in tqdm(wav_paths):
        target_wav_path = join(wav_dir, basename(wav_path))
        wav = librosa.load(wav_path, sr=sr)[0]
        wav_cpu = torch.from_numpy(wav)

        wav = wav_cpu.unsqueeze(0).cuda()
        # Pad to multiple of 320
        pad_for_wav = (320 - (wav.shape[1] % 320)) % 320
        if pad_for_wav > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_for_wav))

        with torch.no_grad():
            # Encode
            vq_emb = encoder(wav.unsqueeze(1))  # [B, 1024, T]

            # Project to VQ dimension
            vq_emb = fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)  # [B, vq_dim, T]

            # Quantize
            _, vq_code, _ = decoder(vq_emb, vq=True)

            # Decode from codes
            vq_post_emb = decoder.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
            vq_post_emb = vq_post_emb.transpose(1, 2)
            vq_post_emb = fc_post_a(vq_post_emb.transpose(1, 2)).transpose(1, 2)
            recon = decoder(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze().detach().cpu().numpy()

        sf.write(target_wav_path, recon, sr)

    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')
