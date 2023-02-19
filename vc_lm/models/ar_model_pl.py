import pytorch_lightning as pl
import torch
import json

from typing import Any, List
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam

from vc_lm.models.base import VCLMConfig
from vc_lm.models.models.ar_model import ARModel, ARModelForConditionalGeneration

class ARModelPL(pl.LightningModule):
    def __init__(self, config_file: str,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005):
        super().__init__()
        self.save_hyperparameters()
        with open(config_file) as f:
            config = json.load(f)
        config = VCLMConfig(**config)
        self.model = ARModelForConditionalGeneration(config)
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
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {
            'loss': loss,
            'preds': preds,
            'targets': targets
        }

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self) -> Any:
        return Adam(self.parameters(),
                    lr=self.hparams.lr,
                    weight_decay=self.hparams.weight_decay)
