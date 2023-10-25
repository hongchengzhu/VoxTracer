# VoxTracer

Official Implementation of VoxTracer (MM' 23, Who is Speaking Actually? Robust and Versatile Speaker Traceability for Voice Conversion). In this implementation, you can complete the tracing stage of our framework.

## Environment

 You can install our necessary python3 environment by running this command:

```bash
pip install -r requirements.txt
```

You need to install pytorch as well. In our case, torch 1.13.1 is installed to support the deep learning.

## Pre-trained models

In this implementation, we provide necessary pre-trained models for the tracing stage. The storage locations for all models provided are as follows:

```
pretrained_models/infer/decoder_35599.pt # ID Decoder which converts hidden variable z to embedding

pretrained_models/infer/waveglow_extr_35599.pt # Speech Inverter which generates hidden variable z

pretrained_models/spk_emb_VCTK_ge2e # Original Speaker Embeddings

wav_after_aac/kbps/val_32kbps_npy # Audio after lossy transmission
```

## Running

You can execute the tracing stage by running :

```bash
python inference_for_tracing.py
```

The source code will load all pre-trained models above and generate corresponding restored speaker embeddings, based on the compressed audio provided.

It should be noted that the provided audios are compressed into .m4a format using **AAC compression**, and then **stored in .npy format** using librosa, which makes the loading period much faster. If you want to test other compression (like MP3, SILK), the .npy audios and their original embeddings should be replaced.

After having the restored embeddings, we calculate cosine similarity of each audio and their original embedding, and determine whether the restored embedding is accurate (cosine similarity must exceed threshold, 0.9513). And the output will be written in ***output/test_vctk_aac_32kbps_35599.txt***



















