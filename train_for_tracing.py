import argparse
import json
import os
import sys
from utils.meldataset import mel_spectrogram
import torch
import time
import datetime
import numpy as np
#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======
from torch.utils.data import DataLoader
from glow_7_23_slice_trn import WaveGlow
from mel2samp_ms_7_23_slice_trn import Mel2Samp
from decoder.model_decoder import FVAEDecoder


global device
device = 'cuda:1'


def train(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          sigma, iters_per_checkpoint, batch_size, seed, fp16_run,
          checkpoint_path, with_tensorboard):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    # ####################################### load decoder ###########################################
    decoder = FVAEDecoder().to(device)
    # decoder.load_state_dict(torch.load(
    #     ckpt_path,
    #     map_location='cpu'
    # ))
    print("Loaded decoder finished ... \n")
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    decoder.train()
    # ################################## load mapping flow over ######################################

    # ############################ Load checkpoint if one exists #####################################
    # extr
    waveglow_extr = WaveGlow(**waveglow_config).to(device)
    waveglow_extr.load_state_dict(torch.load('pretrained/waveglow/waveglow_76000.pt',
        map_location='cpu'
    ))
    print("Loaded checkpoint '{}'".format('waveglow_76000'))
    optimizer_waveglow_extr = torch.optim.Adam(waveglow_extr.parameters(), lr=learning_rate)
    waveglow_extr.train()
    # ################################## Load checkpoint over #####################################

 
    iteration = 0

    # train
    trainset = Mel2Samp(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=False)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    epoch_offset = max(0, int(iteration / len(train_loader)))

    ref_embeds = torch.zeros([100, 256])
    emb_root = 'pretrained/spk_emb_VCTK_ge2e'
    spk_list = sorted(os.listdir(emb_root))[:100]
    for i, spk in enumerate(spk_list):
        ref_embeds[i, :] = torch.FloatTensor(np.loadtxt(os.path.join(emb_root, spk, spk+'.txt'), delimiter=','))
    ref_embeds = ref_embeds.to(device)
    ref_embeds = ref_embeds.unsqueeze(1).repeat(1, 32, 1).reshape(100, 4, 2048)


    os.chdir(sys.path[0])
    with open('output/tracing/training/log/output_vctk_aac_32kbps.txt', 'a') as f_to_write:
    # ================ MAIN TRAINNIG LOOP! ===================
        start_time = time.time()
        for epoch in range(epoch_offset, epochs):
            print("Epoch: {}".format(epoch))
            for index, batch in enumerate(train_loader):

                optimizer_waveglow_extr.zero_grad()
                optimizer_decoder.zero_grad()

                mp3s, embeds, spk_id = batch
                batch_size = mp3s.shape[0]

                mp3s = torch.autograd.Variable(mp3s.to(device))
                embeds = torch.autograd.Variable(embeds.to(device)).squeeze(1)

                audio_after_mp3 = mp3s[:, :16384]
                mels_after_mp3 = torch.zeros([mp3s.shape[0], 80, 64])
                for i in range(len(mp3s)):
                    mels_after_mp3[i, :, :] = mel_spectrogram(audio_after_mp3[i, :].unsqueeze(0), 1024, 80, 22050,
                                                              256, 1024, 0, 8000, center=False).squeeze(0)[:, :64]
                mels_after_mp3 = mels_after_mp3.to(device)

                # # extract
                # step 3: audio + mel --> z (extractor)
                outputs = waveglow_extr((mels_after_mp3, audio_after_mp3))  # audio + mel --> z

                z_mappingflow_rec = outputs[0][:, 4:, :]

                # step 4: get emb
                embeds_hat = decoder(z_mappingflow_rec)

                embeds_dif = torch.abs(embeds_hat - embeds).sum() / (embeds.shape[0] * embeds.shape[1] * embeds.shape[2])

                # classifying accuracy
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                print("time:{}, epoch: {}, iter:{}, emb_dif:{:.4f}".format(
                    et, epoch, iteration, embeds_dif))
                f_to_write.write("time:{}, epoch: {}, iter:{}, emb_dif:{:.4f}".format(
                    et, epoch, iteration, embeds_dif))
                f_to_write.write('\n')
                f_to_write.flush()

                if iteration % 100 == 0 and iteration > 25000:
                    checkpoint_path_waveglow = "{}/waveglow_extr_{}.pt".format(
                        output_directory, iteration)
                    torch.save(waveglow_extr.state_dict(), checkpoint_path_waveglow)
                    checkpoint_path_mapping_flow = "{}/decoder_{}.pt".format(
                        output_directory, iteration)
                    torch.save(decoder.state_dict(), checkpoint_path_mapping_flow)

                loss = embeds_dif

                loss.backward()
                optimizer_decoder.step()
                optimizer_waveglow_extr.step()

                iteration += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config_jointly/config_bk_7_23_slice_trn.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
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
