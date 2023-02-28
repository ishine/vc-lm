# vc-lm
## 构造数据集

```
python tools/construct_dataset.py
```
## 转换whisper encoder模型

```
python tools/extract_whisper_encoder_model.py --input_model=../whisper/small.pt --output_model=../whisper-encoder/small-encoder.pt
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