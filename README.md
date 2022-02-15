# colossal_d2v
Reproduction of data2vec_vision model with colossal-ai


## Background
This project is the reproduction of data2vec_vision model (http://www.baidu.com) with colossal-ai version.

## Envirionment Setup
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

7、换行（建议直接在前一行后面补两个空格）
直接回车不能换行，
可以在上一行文本后面补两个空格，
这样下一行的文本就换行了。
或者就是在两行文本直接加一个空行。
也能实现换行效果，不过这个行间距有点大。

8、引用
> 第一行引用文字
> 第二行引用文字
