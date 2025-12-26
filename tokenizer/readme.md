## Installation
```shell
pip3 install -r requirements.txt
```

## Training Preparation
We use [webdataset](https://github.com/webdataset/webdataset) format for data loading. To begin with, it is needed to convert the dataset into webdataset format. An example script to convert ImageNet to wds format is provided [here](./data/convert_imagenet_to_wds.py).

## Training
We provide example commands to train vgt-ae(intervl3) as follows:
```bash
cd /path/to/VGT/tokenizer
# Training for TiTok-B64
# Stage 1 Single machine (8 GPUs)
bash scripts/vlvae_intervl3_p28_448px_stage1.sh

# Stage 2 Single machine (8 GPUs)
bash scripts/vlvae_intervl3_p28_448px_stage2.sh
```

## Eval

```bash
cd /path/to/VGT/tokenizer
bash scripts/eval/run_eval.sh
```

## Citing
If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@misc{guo2025vgt,
      title={Visual Generation Tuning}, 
      author={Jiahao Guo and Sinan Du and Jingfeng Yao and Wenyu Liu and Bo Li and Haoxiang Cao and Kun Gai and Chun Yuan and Kai Wu and Xinggang Wang},
      year={2025},
      eprint={2511.23469},
      archivePrefix={arXiv},
}
```


## Acknowledgement

[TiTok](https://github.com/bytedance/1d-tokenizer/blob/main/README_TiTok.md)