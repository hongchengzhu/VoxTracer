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
# *****************************************************************************
import argparse
import json
import os
import sys
from utils.meldataset import mel_spectrogram
import torch
import numpy as np
from glow_7_23_slice_trn import WaveGlow
import soundfile as sf
from tqdm import tqdm


global device
device = 'cuda:1'


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ############################ Load checkpoint if one exists #####################################
    waveglow_emb = WaveGlow(**waveglow_config).to(device)
    checkpoint_dict = torch.load('pretrained_models/waveglow/waveglow_76000.pt',
                                 map_location='cpu')
    waveglow_emb.load_state_dict(checkpoint_dict)
    print("Loaded checkpoint '{}')".format(checkpoint_path))
    waveglow_emb.eval()

    for _, param in waveglow_emb.named_parameters():
        param.requires_grad = False
    print('we have set all params of waveglow_emb FALSE!')
    # ################################## Load checkpoint over #####################################


    val_path = ''       # converted mel rootDir
    z_path = 'pretrained_models/z'
    target = 'output/speech_with_hidden'
    if not os.path.exists(target):
        os.mkdir(target)

    with torch.no_grad():
        mel_list = sorted(os.listdir(val_path))
        for melname in tqdm(mel_list):
            src_spk = melname[:4]
            z_mappingflow = torch.FloatTensor(np.load(os.path.join(z_path, src_spk + '.npy'))).to(device)
            mel = torch.FloatTensor(np.load(os.path.join(val_path, melname))).to(device).unsqueeze(0).transpose(1, 2)

            patch_len = 64
            mel_padding = mel[:, :, :int((mel.shape[-1]//patch_len) * patch_len)]
            mel_padding = mel_padding.to(device)

            for j in range(0, mel_padding.shape[2]//patch_len, 1):
                audio_before_mp3, _ = waveglow_emb.infer(mel_padding[:, :, j*patch_len:(j+1)*patch_len], 
                                                         z_mappingflow, 
                                                         z_flag=True
                                                         )
                sf.write('{}/{}_{}.wav'.format(target, 
                                               melname[:-4], str(j).rjust(3, '0')),
                                               audio_before_mp3[0].detach().cpu().numpy(), 
                                               22050
                                               )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config_jointly/config_bk_7_23_slice_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    os.chdir(sys.path[0])
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global waveglow_config
    waveglow_config = config["waveglow_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(num_gpus, args.rank, args.group_name, **train_config)
