import argparse
import json
import os
import sys
from utils.meldataset import mel_spectrogram
import torch
import numpy as np
#=====START: ADDED FOR DISTRIBUTED======
from distributed import init_distributed
#=====END:   ADDED FOR DISTRIBUTED======
from glow_7_23_slice_val import WaveGlow
from decoder.model_decoder import FVAEDecoder
from tqdm import tqdm
import soundfile as sf


global device
device = 'cuda:1'


def train(num_gpus, rank, group_name, batch_size=1, seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #=====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    #=====END:   ADDED FOR DISTRIBUTED======

    # ####################################### load decoder ###########################################
    decoder = FVAEDecoder()
    decoder.load_state_dict(torch.load('pretrained_models/infer/decoder_35599.pt',
        map_location='cpu'))
    print("Loaded decoder finished ... \n")
    decoder.to(device)
    decoder.eval()
    # ################################## load mapping flow over ######################################

    # ############################ Load checkpoint if one exists #####################################
    # extr
    waveglow_extr = WaveGlow(**waveglow_config)
    waveglow_extr.load_state_dict(torch.load('pretrained_models/infer/waveglow_extr_35599.pt',
        map_location='cpu'))
    print("Loaded checkpoint '{}'".format('waveglow_35599'))
    waveglow_extr.to(device)
    waveglow_extr.eval()
    # ################################## Load checkpoint over #####################################

    # for VCTK Corpus
    ref_embeds = torch.zeros([100, 256])
    emb_root = 'pretrained_models/spk_emb_VCTK_ge2e'
    spk_list = sorted(os.listdir(emb_root))[:100]
    for i, spk in enumerate(spk_list):
        ref_embeds[i, :] = torch.FloatTensor(np.loadtxt(os.path.join(emb_root, spk, spk+'.txt'), delimiter=','))
    ref_embeds = ref_embeds.to(device)
    ref_embeds = ref_embeds.unsqueeze(1).repeat(1, 32, 1).reshape(100, 4, 2048)

    val_path = '/mnt1/hongcz/waveglow_fn_encoder_10_13/vctk_100spk/wav_after_aac/kbps/val_32kbps_npy'
    mp3_list = sorted(os.listdir(val_path))

    os.chdir(sys.path[0])
    with open('output/test_vctk_aac_32kbps_35599.txt', 'a') as f_to_write:
        for i in tqdm(range(len(mp3_list))):
            mp3s = torch.FloatTensor(np.load(os.path.join(val_path, mp3_list[i]))).unsqueeze(0)
            if mp3s.shape[1] < 16384:
                tmp = torch.zeros([1, 16384])
                tmp[:, :mp3s.shape[1]] = mp3s
                mp3s = tmp
            else:
                mp3s = mp3s[:, :16384]

            spk_id = os.path.basename(mp3_list[i])[:4]
            spk_id_index = spk_list.index(spk_id)

            mp3s = torch.autograd.Variable(mp3s.to(device))

            audio_after_mp3 = mp3s[:, :16384]
            mels_after_mp3 = mel_spectrogram(audio_after_mp3,
                                             1024, 80, 22050, 256, 1024, 0, 8000, center=False)[:, :, :64]
            mels_after_mp3 = mels_after_mp3.to(device)

            outputs = waveglow_extr((mels_after_mp3, audio_after_mp3))  # audio + mel --> z

            z_mappingflow_rec = outputs[0][:, 4:, :]

            embeds_hat = decoder(z_mappingflow_rec)

            cos_dis = torch.zeros([batch_size, 100]).to(device)
            for m in range(batch_size):
                for n in range(100):
                    cos_dis[m, n] = (ref_embeds[n, :, :] * embeds_hat[m, :, :]).sum() / \
                                    torch.sqrt(((ref_embeds[n, :, :] * ref_embeds[n, :, :]).sum()) * \
                                               (embeds_hat[m, :, :] * embeds_hat[m, :, :]).sum())

            max_indexes = torch.argmax(cos_dis, dim=1)

            f_to_write.write("cos:{:.4f}, accuracy:{}\n".format(torch.max(cos_dis),
                                                                    spk_id_index == int(max_indexes)))
            f_to_write.flush()




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
    train(num_gpus, args.rank, args.group_name)
