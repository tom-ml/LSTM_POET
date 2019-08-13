# LSTM_POET

标签： Pytorch LSTM Python Deep-learning
------

使用两层的LSTM网络实现了基于字的中文神经网络写诗模型，支持自动生成唐诗宋词或自动生成藏头诗功能。


## 实验环境
- [x] [torch - 1.0.1](https://pytorch.org/)
- [x] torchvision - 0.2.2
- [x] torchnet - 0.0.4
- [x] tqdm - 4.19.9
- [x] [Python - 3.6.3](https://www.python.org/)

## 实验数据
本实验的数据均来自于[中华最全古诗词数据库](https://github.com/chinese-poetry/chinese-poetry),使用其[json文件夹中的诗词文件](https://github.com/chinese-poetry/chinese-poetry/tree/master/json)即可,当前程序在处理的过程中将读取[data](https://github.com/Niutranser-Li/LSTM_POET/tree/master/data)中的原始文件，程序在训练的过程中将读取已经编译好的二进制文件，若data文件夹中指定的二进制文件**（training-picket.npz）**已经存在，程序将自动读取二进制文件，否则程序将重新进行二进制文件的生成。

## 模型参数
```
dataset_path = "data/"  # The path of poetry dataset
picket_file_path = "data/training-picket.npz"   # Binary files after pre-processing can be used directly for model training
author_limit = None     # Author limit, if not None, will only learn the author's verse
length_limit = None     # length limit, if it is not None, only the verses of the specified length will be learned.
class_limit = "poet.tang"       # class limit, value choose[poet.tang, poet.song]
learning_rate = 1e-3    # The model learning rate
weight_decay = 1e-4
use_gpu = True  # is or not use gpu
epoch = 20      # The model training epoch.
batch_size = 128        # model training batch size.
max_length = 125
plot_every = 20
use_env = False # if or not use visodm
env = 'poetry'  # visdom env
generatea_max_length_limit = 200        # generate poetry max length.
debug_file_path = "debugp"
pre_training_model_path = None  # The path of pre-training model.
prefix_words = "细雨鱼儿出,微风燕子斜。"        # Control poetry
start_words = "闲云潭影日悠悠"  # poetry start
acrostic = False        # Is it a Tibetan poem?
model_prefix = "checkpoints/"
```

## 模型训练
默认参数训练：
```
python3 main.py train
```
指定参数训练:
```
python3 main.py train --batch_size = 128 --picket_file_path="data/training-picket.npz" --learning_rate = 1e-3 --epoch = 100
```

## 模型测试
生成诗歌：
```
python3 main.py generate --pre_training_model_path=checkpoints/poet.tang_20.pth --start-words="江梳天地 外" --prefix-words="江流天地外，山色有无中。"

output > 江梳天地外，水闊海門前。有時不可見，此地無何如。我來不可見，此地無所求。我來不可見，不得無所求。我來不可見，我亦不可尋。我來不可見，我亦不可尋。我來不可見，此地無所求。一朝不可見，一日不可歸。我來不可見，況乃君子歸。我來不可見，此地無所求。一朝不可見，一日不可歸。我來不可見，此地無所求。一爲一杯酒，一笑不可尋。
```

生成藏头诗：
```
python3 main.py generate --pre_training_model_path=checkpoints/poet.tang_19.pth --start-words='小牛小牛' --prefix-words='江流天地外，山色有无中。' --acrostic=True

output > 小人不可見，此地無所求。牛羊不可見，不敢問其情。小人不可見，所以不可忘。牛羊不敢問，不敢問其情。
```

## 备注
向`hello_xiaojian@126.com`发送邮件了解更多模型信息

