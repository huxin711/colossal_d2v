## Background
This project is the reproduction of [data2vec_vision model](https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language) with [ColossalAI](https://github.com/hpcaitech/ColossalAI) tool.

## Envirionment setup
```
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
# install dependency
pip install -r requirements/requirements.txt

# install colossalai
pip install .
```

## How to run
```
$ DATA=/path/to/data/ python -m torch.distributed.launch --nproc_per_node=nproc_per_node
                                                         --master_addr MASTER_ADDR
                                                         --master_port MASTER_PORT
                                                         train_engine.py
                                                         --config=CONFIG_FILE
                                                         --world_size=WORLD_SIZE
                                                         --rank=RANK
                                                         --local_rank=LOCAL_RANK
```

## Cite us
```
@article{bian2021colossal,
  title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
  author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
  journal={arXiv preprint arXiv:2110.14883},
  year={2021}
}
```