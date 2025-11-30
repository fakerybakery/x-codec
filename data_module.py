import os

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        # ds = FSDataset_add_STFT(phase, self.cfg)
        dl = DataLoader(ds, 
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=28,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        pass

class FSDataset(Dataset):
    """Dataset batching wav, mel
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()

        self.sr = cfg.preprocess.audio.sr

        self.filelist = self.get_filelist(self.phase_cfg.filelist)
        self.min_audio_length = cfg.dataset.min_audio_length
    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def get_filelist(self, fpath):
        with open(fpath, 'r') as f:
            # flist = [l.strip() for l in f if l.strip()]
            flist = [l.strip().split('\t')[0] for l in f if l.strip()]
        return flist

    def __getitem__(self, idx):
        wavpath = self.filelist[idx]
        wavpath_full = join(self.cfg.preprocess.datasets.LibriSpeech.root, wavpath)

        wav, sr = torchaudio.load(wavpath_full)

        if sr != self.sr:
            wav = Resample(sr, self.sr)(wav)
        wav = wav[0, :]
        length = wav.shape[0]

        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]
        i = random.randint(0, length - self.min_audio_length)
        wav = wav[i:i + self.min_audio_length]

        out = {
            'wav': wav,
        }

        return out

    def collate_fn(self, bs):
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        out = {
            'wav': wavs,
        }
        return out

@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):
 
    data_module = DataModule(cfg)

 
    train_loader = data_module.val_dataloader()

 
    valid_filelist = []

 
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
 
        wavs = batch['wav']
 

if __name__ == "__main__":
    main()

