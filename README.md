# VoxTracer

Official Implementation of VoxTracer (ACM MM 2023). In this implementation, you can partially reproduce the tracing stage of our framework.

Thanks [@Demo36days](https://github.com/Demo36days) for helping build this project.

## Environment

 You can install our necessary python3 environment by running this command:

```bash
pip install -r requirements.txt
```

You need to install pytorch as well. In our case, torch 1.13.1 is installed to support the deep learning.


## What you should download before running our code

### Pre-trained models

In this implementation, we provide necessary pre-trained models for the tracing stage. The storage locations for all models provided are as follows:

```
pretrained_models/infer/decoder_35599.pt # ID Decoder which converts hidden variable z to embedding

pretrained_models/infer/waveglow_extr_35599.pt # Speech Inverter which generates hidden variable z

pretrained_models/spk_emb_VCTK_ge2e # Original Speaker Embeddings

wav_after_aac/kbps/val_32kbps_npy # Audio after lossy transmission
```
To download our models, please visit: https://drive.google.com/drive/folders/133O8WGeslIsfcdRmlcrCGYuO67ARJUPx?usp=drive_link. You should download the *pretrained_models* and place it in the current path.

### Data for testing
Due to the very large amount of data, we just provide partial data for testing the traceability in 32kbps-AAC compression. The provided data is randomly selected from our testing set which isn't overlapped with the training set.

First, you should create a new direction in the current path, i.e., ```mkdir testdata/wav_after_aac/kbps```. Then, download *val_32kbps_npy* from the above link and place it to the newly created path.

It should be noted that the provided audios are compressed into .m4a format using **AAC compression**, and then **decompressed and stored in .npy format**, which makes the loading period much faster. If you want to test other compression (like MP3, Opus, and SILK), you should self-prepare your data and self-train your model.


## Running

### Testing

You can execute the tracing stage by running :

```bash
python inference_for_tracing.py
```

The source code will load all pre-trained models above and generate corresponding restored speaker embeddings, based on the compressed audio provided.


After having the restored embeddings, we calculate cosine similarity of each audio and their original embedding, and determine whether the restored embedding is accurate (cosine similarity must exceed the threshold which is calculated by the training set, 0.9513). And the output will be written in ***output/test_vctk_aac_32kbps_35599.txt***.

### Training
We provide the training code of the tracing stage in *train_for_tracing.py*. However, if you'd like to build your own model, we strongly advise you self-prepare your data and model, and self-train your own model.


## Citation
If you find this useful for your research, please cite our paper:
```
Yanzhen Ren, Hongcheng Zhu, Liming Zhai, Zongkun Sun, Rubing Shen, and Lina Wang. 2023. Who is Speaking Actually? Robust and Versatile Speaker Traceability for Voice Conversion. In Proceedings of the 31st ACM International Conference on Multimedia (MM ’23), October 29–November 3, 2023, Ottawa, ON, Canada. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3581783.3612333
```




## Acknowledgments
Our codes are influenced by the following repos:
- [WaveGlow](https://github.com/NVIDIA/waveglow)
- [PortaSpeech](https://github.com/NATSpeech/NATSpeech)
- [AutoVC](https://github.com/auspicious3000/autovc)



















