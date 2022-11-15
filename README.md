# Pseudo-Labels Post Process (PLPP)
PLPP has been used as a post process for unsupervised domain adaptation (UDA) tasks, improving significantly the results for some state of the art methods.
Watch the video for an example of segmentation results on cc359 dataset, using PLPP after adaBN finetune:
https://www.youtube.com/watch?v=owcztoLX1Zo

Below is the python implementation for our method, applied on cc359 and MSM datasets.


## Installation

* Install wandb using https://docs.wandb.ai/quickstart
* git clone https://github.com/TomerBarNatan/PLPP.git
* cd PLPP
* pip3 install -r requirements.txt
* cd ..
* git clone https://github.com/deepmind/surface-distance.git
* pip install surface-distance/

## Dataset

* Download CC359 from: https://www.ccdataset.com/download
* Download MultiSiteMri (msm) from: https://liuquande.github.io/SAML/
* point paths to the downloaded directories at paths.py
* run ```python3 -m dataset create_all_images_pickle```

## Pre-train Models
* the results will be visible at https://wandb.ai/
* source can be any number between 0 and 5. 
### cc359

```
python3 trainer.py --source {source} --target {source} --mode pretrain --gpu {device}
```

### msm

```
python3 trainer.py --source {source} --target {source} --mode pretrain --gpu {device} --msm
```



## fine-tune model
* the results will be visible at https://wandb.ai/

### cc359
* source and be target can be any number between 0 and 5.
* source and target should not be the same
```
python3 trainer.py --source {source} --target {target} --mode clustering_finetune --gpu {device}
```

### msm
* target can be any number between 0 and 5.
```
python3 trainer.py  --target {target} --mode clustering_finetune --gpu {device} --msm
```
