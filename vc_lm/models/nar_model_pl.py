import pytorch_lightning as pl
import torch
import json

from typing import Any, List
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import AdamW

from vc_lm.models.base import VCLMConfig
from vc_lm.models.models.nar_model import NARModel

from vc_lm.models.misc import CosineWarmupScheduler
from transformers.optimization import get_polynomial_decay_schedule_with_warmup


class NARModelPL(pl.LightningModule):
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
        self.model = NARModel(config)
        # load whisper parameter
        self.model.encoder.load_pretrained_whisper_params()
        self.loss_fct = nn.CrossEntropyLoss()

        loaded_state = torch.load('/root/autodl-tmp/vc-models/nar-1024.ckpt')['state_dict']
        # for k, v in list(loaded_state.items()):
        #     if '.encoder.' in k:
        #         del loaded_state[k]
        self.load_state_dict(loaded_state,
                             strict=False)

        self.train_accuracy = Accuracy(task="multiclass",
                                       num_classes=self.model.shared.num_embeddings,
                                       average='micro',
                                       ignore_index=-100)
        self.val_accuracy = Accuracy(task="multiclass",
                                     num_classes=self.model.shared.num_embeddings,
                                     average='micro',
                                     ignore_index=-100)
        self.test_accuracy = Accuracy(task="multiclass",
                                      num_classes=self.model.shared.num_embeddings,
                                      average='micro',
                                      ignore_index=-100)

    def forward(self,
                input_mels=None,
                attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                encoder_outputs=None,
                style_code=None,
                nar_stage=None):
        outputs = self.model(input_ids=input_mels,
                             attention_mask=attention_mask,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             encoder_outputs=encoder_outputs,
                             style_code=style_code,
                             nar_stage=nar_stage)
        return outputs

    def step(self, batch: Any):
        _, lm_logits = self.forward(input_mels=batch['mel'],
                                    attention_mask=batch['content_mask'],
                                    decoder_input_ids=batch['input_code'],
                                    decoder_attention_mask=batch['code_mask'],
                                    encoder_outputs=batch.get('encoder_outputs', None),
                                    style_code=batch['style_code'],
                                    nar_stage=batch['nar_stage'])
        output_code = batch['output_code']
        output_code[output_code == self.model.config.pad_token_id] = -100

        lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), output_code.view(-1))
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
        #return {"loss": loss, "preds": preds, "targets": targets}
        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        #return {"loss": loss, "preds": preds, "targets": targets}
        return {'loss': loss}

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
