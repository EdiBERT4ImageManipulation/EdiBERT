# EdiBERT, a generative model for image editing

We follow the implementation of Taming-Transformers (https://github.com/CompVis/taming-transformers).
Modifications of the  can be found in: taming/models/bert_transformer.py

## Requirements
A suitable [conda](https://conda.io/) environment named `edibert` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate edibert
```

### FFHQ
Download FFHQ dataset (https://github.com/NVlabs/ffhq-dataset) and put it into `data/ffhq/`.

### Training

In the logs/ folder, download and extract the FFHQ VQGAN:
```
gdown --id '1P_wHLRfdzf1DjsAH_tG10GXk9NKEZqTg'
tar -xvzf 2021-04-23T18-19-01_ffhq_vqgan.tar.gz
```

Training on 1 GPUs:
```
python main.py --base configs/ffhq_transformer_bert_2D.yaml -t True --gpus 0,
```
Training on 2 GPUs:
```
python main.py --base configs/ffhq_transformer_bert_2D.yaml -t True --gpus 0,1
```


### Running pre-trained BERT on composite/scribble-edited images

In the logs/ folder, download and extract  the FFHQ BERT:
```
gdown --id '18fF0sTArNjD-h4fVvp2T3vulDgbPf_5P'
tar -xvzf 2021-10-14T16-32-28_ffhq_transformer_bert_2D.tar.gz
```
folders and place them into logs.

Then, launch the following script for composite images:
```
python scripts/sample_mask_likelihood_maximization.py -r logs/2021-10-14T16-32-28_ffhq_transformer_bert_2D/checkpoints/epoch=000019.ckpt \
--image_folder data/ffhq_collages/ --mask_folder data/ffhq_collages_masks/ --image_list data/ffhq_collages.txt --keep_img \
--dilation_sampling 1 -k 100 -t 1.0 --batch_size 5 --bert --epochs 2  \
--device 0 --random_order \
--mask_collage --collage_frequency 3 --gaussian_smoothing_collage
```

Then, launch the following script for edits images:
```
python scripts/sample_mask_likelihood_maximization.py -r logs/2021-10-14T16-32-28_ffhq_transformer_bert_2D/checkpoints/epoch=000019.ckpt \
--image_folder data/ffhq_edits/ --mask_folder data/ffhq_edits_masks/ --image_list data/ffhq_edits.txt --keep_img \
--dilation_sampling 1 -k 100 -t 1.0 --batch_size 5 --bert --epochs 2  \
--device 0 --random_order \
--mask_collage --collage_frequency 3 --gaussian_smoothing_collage
```

### Notebooks for playing with completion/denoising with BERT

Notebooks can be found in `scripts/`.
