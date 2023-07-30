# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from scipy.io.wavfile import read
import numpy as np
import soundfile as sf


MAX_WAV_VALUE = 32768.0

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    # files = [f.rstrip() for f in files]
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def load_wav_from_npy(full_path):
    # print(full_path+'.npy')
    wav = torch.FloatTensor(np.load(full_path+'.npy'))
    return wav


class Mel2Samp(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, vc_mel_path, embeds_path, segment_length, filter_length,
                 hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        self.audio_files = files_to_list(vc_mel_path) 
        self.embeds_path = embeds_path
        random.seed(1234)
        random.shuffle(self.audio_files)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.spk_list = sorted(os.listdir(embeds_path))[:100]


    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index][:-1]
        spk = os.path.basename(filename)[:4]

        mp3 = torch.zeros([16384])
        tmp = torch.FloatTensor(np.load(filename))
        if len(tmp) > len(mp3):
            mp3 = tmp[:16384]
        else:
            mp3[:len(tmp)] = tmp
        src_emb = torch.FloatTensor(np.loadtxt(os.path.join(self.embeds_path, spk[:4], spk[:4] + '.txt'), delimiter=','))
        src_emb = src_emb.unsqueeze(0).unsqueeze(0).repeat(1, 32, 1).reshape(1, 4, 256*8)

        spk_id = self.spk_list.index(spk[:4])


        return (mp3, src_emb, spk_id)

    def __len__(self):
        return len(self.audio_files)

