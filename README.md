# EdiBERT, a generative model for image editing

EdiBERT is a generative model based on a bidirectional transformer, suited for image manipulation. The same EdiBERT model, derived from a single training, can be used on a wide variety of tasks.

For a quick start, a Colab Demo for editing tasks is available here: https://colab.research.google.com/github/EdiBERT4ImageManipulation/EdiBERT/blob/main/EdiBERT_demo.ipynb 

![edibert_example](https://user-images.githubusercontent.com/94860822/157225722-fab1a9a5-a7f2-4cd2-952c-367e9c9d8e3e.png)

We follow the implementation of Taming-Transformers (https://github.com/CompVis/taming-transformers).
Main modifications can be found in: `taming/models/bert_transformer.py` ; `scripts/sample_mask_likelihood_maximization.py`.

## Requirements
A suitable [conda](https://conda.io/) environment named `edibert` can be created
and activated with:


```
conda env create -f environment.yaml
conda activate edibert
```

### FFHQ
Download FFHQ dataset (https://github.com/NVlabs/ffhq-dataset) and put it into `data/ffhq/`.

### Training BERT

In the logs/ folder, download and extract the FFHQ VQGAN:
```
gdown '1P_wHLRfdzf1DjsAH_tG10GXk9NKEZqTg'
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

In the logs/ folder, download and extract the FFHQ VQGAN:
```
gdown '1P_wHLRfdzf1DjsAH_tG10GXk9NKEZqTg'
tar -xvzf 2021-04-23T18-19-01_ffhq_vqgan.tar.gz
```

In the logs/ folder, download and extract  the FFHQ BERT:
```
gdown '1YGDd8XyycKgBp_whs9v1rkYdYe4Oxfb3'
tar -xvzf 2021-10-14T16-32-28_ffhq_transformer_bert_2D.tar.gz
```
folders and place them into logs.

Then, launch the following script for composite images:
```
python scripts/sample_mask_likelihood_maximization.py -r logs/2021-10-14T16-32-28_ffhq_transformer_bert_2D/checkpoints/epoch=000019.ckpt \
--image_folder data/ffhq_collages/ --mask_folder data/ffhq_collages_masks/ --image_list data/ffhq_collages.txt --keep_img \
--dilation_sampling 1 -k 100 -t 1.0 --batch_size 5 --bert --epochs 2  \
--device 0 --random_order \
--mask_collage --collage_frequency 3 --gaussian_smoothing_collage \
--num_optim_steps 200
```

Then, launch the following script for edits images:
```
python scripts/sample_mask_likelihood_maximization.py -r logs/2021-10-14T16-32-28_ffhq_transformer_bert_2D/checkpoints/epoch=000019.ckpt \
--image_folder data/ffhq_edits/ --mask_folder data/ffhq_edits_masks/ --image_list data/ffhq_edits.txt --keep_img \
--dilation_sampling 1 -k 100 -t 1.0 --batch_size 5 --bert --epochs 2  \
--device 0 --random_order \
--mask_collage --collage_frequency 3 --gaussian_smoothing_collage \
--num_optim_steps 200
```

The samples can then be found in `logs/my_model/samples/`.
Here, the `--batch_size` argument corresponds to the number of EdiBERT generations per image.

### Notebooks for playing with completion/denoising with BERT

Notebooks for image denoising and image inpainting can also be found in the main folder.
