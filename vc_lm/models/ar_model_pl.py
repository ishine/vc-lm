import pytorch_lightning as pl
import torch
import json

from typing import Any, List
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import AdamW

from vc_lm.models.base import VCLMConfig
from vc_lm.models.models.ar_model import ARModel, ARModelForConditionalGeneration
from vc_lm.models.misc import CosineWarmupScheduler

from transformers.optimization import get_polynomial_decay_schedule_with_warmup


class ARModelPL(pl.LightningModule):
    def __init__(self, config_file: str,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 warmup_step: int = 10000,
                 max_iters: int = 800000):
        super().__init__()
        self.save_hyperparameters()
        with open(config_file) as f:
            config = json.load(f)
        config = VCLMConfig(**config)
        self.model = ARModelForConditionalGeneration(config)
        # 加载whisper模型参数.
        self.model.model.encoder.load_pretrained_whisper_params()

        self.loss_fct = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass",
                                       num_classes=self.model.model.shared.num_embeddings,
                                       average='micro',
                                       ignore_index=-100)
        self.val_accuracy = Accuracy(task="multiclass",
                                     num_classes=self.model.model.shared.num_embeddings,
                                     average='micro',
                                     ignore_index=-100)
        self.test_accuracy = Accuracy(task="multiclass",
                                      num_classes=self.model.model.shared.num_embeddings,
                                      average='micro',
                                      ignore_index=-100)

    def load_bart_decoder_params(self):
        decoder = self.model.model.decoder
        bart_state_dict = torch.load('/root/autodl-tmp/pretrained-models/bart-large/pytorch_model.bin')
        filtered_state_dict = {}
        for k, v in bart_state_dict.items():
            if 'decoder.layers' in k:
                filtered_state_dict[".".join(k.split('.')[1:])] = v
        decoder.load_state_dict(filtered_state_dict, strict=False)

    def forward(self,
                input_mels=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None):
        outputs = self.model(input_ids=input_mels,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask)
        return outputs[0]

    def step(self, batch: Any):
        mel = batch['mel']
        content_mask = batch['content_mask']
        input_code = batch['input_code']
        output_code = batch['output_code']
        output_code[output_code == self.model.config.pad_token_id] = -100
        code_mask = batch['code_mask']
        lm_logits = self.forward(input_mels=mel,
                     attention_mask=content_mask,
                     decoder_input_ids=input_code,
                     decoder_attention_mask=code_mask)
        lm_loss = self.loss_fct(lm_logits.view(-1, self.model.config.vocab_size), output_code.view(-1))
        preds = torch.argmax(lm_logits, dim=-1)
        return lm_loss, preds, output_code

    def training_step(self,
                      batch: Any,
                      batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_accuracy(preds, targets)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        # return {
        #     'loss': loss,
        #     'preds': preds,
        #     'targets': targets
        # }
        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        # return {"loss": loss, "preds": preds, "targets": targets}
        return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self) -> Any:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        # scheduler = CosineWarmupScheduler(optimizer=optimizer,
        #                                   warmup=self.hparams.warmup_step,
        #                                   max_iters=self.hparams.max_iters)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                              num_warmup_steps=self.hparams.warmup_step,
                                                              num_training_steps=self.hparams.max_iters)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
