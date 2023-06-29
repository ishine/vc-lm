# vc-lm
[**中文**](./README.md) | [**English**](./README_en.md)

vc-lm是一个可以将任意人的音色转换为成千上万种不同音色的音频的项目。

## 🔄 最近更新
* [2023/06/09] 新增Any-to-One声音转换模型训练.

## 算法架构
该项目参考论文 [Vall-E](https://arxiv.org/abs/2301.02111)

使用[encodec](https://github.com/facebookresearch/encodec),
将音频离散化成tokens, 在tokens上构建transformer语言模型。
该项目包含两阶段模型 AR模型和NAR模型。

输入: 3s音色prompt音频 + 被转换音频

输出: 转换后音频

在训练阶段，采用了自监督的方式，其中源音频和目标音频是相同的。
### AR阶段
输入: prompt音频 + 源音频

输出: 目标音频 0 level tokens

![ar](res/vclm-ar.png)

### NAR阶段
输入: 目标音频(0~k)level tokens

输出: 目标音频k+1 level tokens

![nar](res/vclm-nar.png)

## 构造数据集

```
# 所有wav文件先处理成长度10~24s的文件, 参考文件[tools/construct_wavs_file.py]
python tools/construct_dataset.py
```
## 转换whisper encoder模型

```
python tools/extract_whisper_encoder_model.py --input_model=../whisper/medium.pt --output_model=../whisper-encoder/medium-encoder.pt
```
## 训练
```
bash ./sh/train_ar_model.sh
bash ./sh/train_nar_model.sh
```
## 推理
```
from vc_lm.vc_engine import VCEngine
engine = VCEngine('/root/autodl-tmp/vc-models/ar.ckpt',
                  '/root/autodl-tmp/vc-models/nar.ckpt',
                  '/root/project/vc-lm/configs/ar_model.json',
                  '/root/project/vc-lm/configs/nar_model.json')
output_wav = engine.process_audio(content_wav,
                                  style_wav, max_style_len=3, use_ar=True)           
```

## 样例展示
[输入音频](res/test-in.wav)

[输出音频1](res/o1.wav)

[输出音频2](res/o2.wav)

[输出音频3](res/o3.wav)

[输出音频4](res/o4.wav)

[输出音频5](res/o5.wav)

---
```
本项目模型可以生成大量one-to-any的平行数据(也就是any-to-one)。这些平行数据可以被用来训练 Any-to-One 的变声模型。
```
---
## 训练Any-to-One VC模型
目标人数据仅需10分钟，即可达到很好的效果。

### 构造数据集
```
# 所有wav文件先处理成长度10~24s的文件, 参考文件[tools/construct_wavs_file.py]
python tools/construct_dataset.py
```

### 构造Any-to-one平行数据
```
# 需要构造train, val, test数据
python tools.construct_parallel_dataset.py
```
### 训练模型
加载上面的预训练模型，在指定人数据上训练。
```
bash ./sh/train_finetune_ar_model.sh
bash ./sh/train_finetune_nar_model.sh
```

### 推理
```
from vc_lm.vc_engine import VCEngine
engine = VCEngine('/root/autodl-tmp/vc-models/jr-ar.ckpt',
                  '/root/autodl-tmp/vc-models/jr-nar.ckpt',
                  '/root/project/vc-lm/configs/ar_model.json',
                  '/root/project/vc-lm/configs/nar_model.json')
output_wav = engine.process_audio(content_wav,
                                  style_wav, max_style_len=3, use_ar=True)           
```
### DEMO
#### 输入音频:  
https://github.com/nilboy/vc-lm/assets/17962699/d9c7fb99-7d34-468b-a376-1c8c882d97e2
#### 输出音频:
https://github.com/nilboy/vc-lm/assets/17962699/7a7620d7-e71b-4655-8ad4-2fb543c92960
