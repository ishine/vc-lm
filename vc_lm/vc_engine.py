from encodec import EncodecModel
import torch
import math
from vc_lm.models.ar_model_pl import ARModelPL
from vc_lm.models.nar_model_pl import NARModelPL
from vc_lm.utils.data_utils import get_code, get_mel_spectrogram, pad_or_trim
import soundfile as sf
import librosa

class VCEngine(object):
    def __init__(self,
                 ar_model_path: str,
                 nar_model_path: str,
                 ar_config_file: str,
                 nar_config_file: str,
                 device: str = 'cuda:0'):
        self.device = device
        # Load AR model.
        ar_model = ARModelPL.load_from_checkpoint(ar_model_path,
                                                  config_file=ar_config_file)
        self.ar_model = ar_model.eval().to(self.device).model
        # Load NAR model.
        nar_model = NARModelPL.load_from_checkpoint(nar_model_path,
                                                    config_file=nar_config_file)
        self.nar_model = nar_model.eval().to(self.device).model

        self.config = self.nar_model.config
        # load encodec model
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model.set_target_bandwidth(6.0)
        self.encodec_model = encodec_model.to(self.device)
        self.max_mel_audio_time = 30
        self.max_mel_len = 100 * self.max_mel_audio_time
        self.max_content_len = math.ceil(self.max_mel_len/2)

    def process_ar(self, content_mel, content_code, style_mel, style_code):
        # Process ARModel
        style_code_len = style_code.shape[1]
        total_code_len = style_code.shape[1] + content_code.shape[1]
        # (80, len)
        content_mel = torch.cat([style_mel, content_mel], 1)
        mel_len = content_mel.shape[1]
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long).to(self.device)
        content_mel = pad_or_trim(content_mel, self.max_mel_len)
        # (style_code_len,)
        style_code = style_code[0]
        # batch input data
        content_mel = content_mel.unsqueeze(0)
        content_mask = content_mask.unsqueeze(0)
        style_code = style_code.unsqueeze(0)

        with torch.no_grad():
            outputs = self.ar_model.generate(content_mel,
                                             attention_mask=content_mask,
                                             decoder_input_ids=style_code,
                                             min_length=total_code_len+1,
                                             max_length=total_code_len+1)
        return outputs[0, style_code_len:total_code_len]

    def process_nar(self, content_mel, style_code, codes_0):
        style_code = style_code[:, 0:75 * 3]
        # codes_0: (code_len,)
        mel_len = content_mel.shape[1]
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long).to(self.device)
        content_mel = pad_or_trim(content_mel, self.max_mel_len)
        #
        content_mel = content_mel.unsqueeze(0)
        content_mask = content_mask.unsqueeze(0)
        style_code = style_code.unsqueeze(0)
        target_len = codes_0.shape[0]
        target_mask = torch.ones((1, target_len), dtype=torch.int64).to(self.device)
        #
        encoder_outputs = None
        codes_list = [codes_0]

        for i in range(0, self.config.n_q - 1):
            # prepare data.
            decoder_input_ids = torch.stack(codes_list, 0)
            # (1, n_q, code_len)
            decoder_input_ids = decoder_input_ids[None]
            nar_stage = torch.LongTensor([i]).to(self.device)
            _, logits = self.nar_model(input_ids=content_mel,
                                       attention_mask=content_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=target_mask,
                                       encoder_outputs=encoder_outputs,
                                       style_code=style_code,
                                       nar_stage=nar_stage)
            preds = torch.argmax(logits, dim=-1)
            codes_list.append(preds[0])
        full_codes = torch.stack(codes_list, 0)
        return full_codes

    def process_audio(self, content_audio: str, style_audio: str,
                      max_style_len=3, max_content_len=15, use_ar=True):
        dtype = torch.float32
        # (80, content_mel_len)
        content_mel = torch.tensor(get_mel_spectrogram(content_audio), dtype=dtype).to(self.device)[:, 0:100 * max_content_len]
        # (n_q, content_code_len)
        content_code = torch.tensor(get_code(content_audio, 'cuda:0', self.encodec_model), dtype=torch.int64).to(self.device)[:, 0:75 * max_content_len]
        # (80, style_mel_len)
        style_mel = torch.tensor(get_mel_spectrogram(style_audio), dtype=dtype).to(self.device)[:, 0:100 * max_style_len]
        # (n_q, style_code_len)
        style_code = torch.tensor(get_code(style_audio, 'cuda:0', self.encodec_model), dtype=torch.int64).to(self.device)[:, 0:75 * max_style_len]
        # Process ARModel
        if use_ar:
            codes_0 = self.process_ar(content_mel, content_code, style_mel, style_code)
        else:
            codes_0 = content_code[0]
        # Process NARModel
        full_codes = self.process_nar(content_mel, style_code, codes_0)
        # Decode encodec
        with torch.no_grad():
            outputs = self.encodec_model.decode([(full_codes.unsqueeze(0), None)])
        return outputs[0, 0].detach().cpu().numpy()

class VCEngineDataFactory(object):
    def __init__(self,
                 ar_model_path: str,
                 nar_model_path: str,
                 ar_config_file: str,
                 nar_config_file: str,
                 device: str = 'cuda:0'):
        self.device = device
        # Load AR model.
        ar_model = ARModelPL.load_from_checkpoint(ar_model_path,
                                                  config_file=ar_config_file)
        self.ar_model = ar_model.eval().to(self.device).model
        # Load NAR model.
        nar_model = NARModelPL.load_from_checkpoint(nar_model_path,
                                                    config_file=nar_config_file)
        self.nar_model = nar_model.eval().to(self.device).model

        self.config = self.nar_model.config
        # load encodec model
        encodec_model = EncodecModel.encodec_model_24khz()
        encodec_model.set_target_bandwidth(6.0)
        self.encodec_model = encodec_model.to(self.device)
        self.max_mel_audio_time = 30
        self.max_mel_len = 100 * self.max_mel_audio_time
        self.max_content_len = math.ceil(self.max_mel_len/2)

    def process_ar(self, content_mel, content_code, style_mel, style_code):
        # Process ARModel
        style_code_len = style_code.shape[1]
        total_code_len = style_code.shape[1] + content_code.shape[1]
        # (80, len)
        content_mel = torch.cat([style_mel, content_mel], 1)
        mel_len = content_mel.shape[1]
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long).to(self.device)
        content_mel = pad_or_trim(content_mel, self.max_mel_len)
        # (style_code_len,)
        style_code = style_code[0]
        # batch input data
        content_mel = content_mel.unsqueeze(0)
        content_mask = content_mask.unsqueeze(0)
        style_code = style_code.unsqueeze(0)

        with torch.no_grad():
            outputs = self.ar_model.generate(content_mel,
                                             attention_mask=content_mask,
                                             decoder_input_ids=style_code,
                                             min_length=total_code_len+1,
                                             max_length=total_code_len+1)
        return outputs[0, style_code_len:total_code_len]

    def process_nar(self, content_mel, style_code, codes_0):
        style_code = style_code[:, 0:75 * 3]
        # codes_0: (code_len,)
        mel_len = content_mel.shape[1]
        content_mask = torch.lt(torch.arange(0, self.max_content_len), math.ceil(mel_len//2)).type(torch.long).to(self.device)
        content_mel = pad_or_trim(content_mel, self.max_mel_len)
        #
        content_mel = content_mel.unsqueeze(0)
        content_mask = content_mask.unsqueeze(0)
        style_code = style_code.unsqueeze(0)
        target_len = codes_0.shape[0]
        target_mask = torch.ones((1, target_len), dtype=torch.int64).to(self.device)
        #
        encoder_outputs = None
        codes_list = [codes_0]

        for i in range(0, self.config.n_q - 1):
            # prepare data.
            decoder_input_ids = torch.stack(codes_list, 0)
            # (1, n_q, code_len)
            decoder_input_ids = decoder_input_ids[None]
            nar_stage = torch.LongTensor([i]).to(self.device)
            _, logits = self.nar_model(input_ids=content_mel,
                                       attention_mask=content_mask,
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=target_mask,
                                       encoder_outputs=encoder_outputs,
                                       style_code=style_code,
                                       nar_stage=nar_stage)
            preds = torch.argmax(logits, dim=-1)
            codes_list.append(preds[0])
        full_codes = torch.stack(codes_list, 0)
        return full_codes

    def process_audio(self, mel1, code1, mel2, code2,
                      max_style_len=3, max_content_len=21):
        dtype = torch.float32
        mel1 = torch.tensor(mel1, dtype=dtype).to(self.device)
        code1 = torch.tensor(code1, dtype=torch.int64).to(self.device)
        mel2 = torch.tensor(mel2, dtype=dtype).to(self.device)
        code2 = torch.tensor(code2, dtype=torch.int64).to(self.device)
        content_mel = mel1[:, 0:100 * max_content_len]
        content_code = code1[:, 0:75 * max_content_len]
        style_mel = mel2[:, 0:100 * max_style_len]
        style_code = code2[:, 0:75 * max_style_len]
        # Process ARModel
        codes_0 = self.process_ar(content_mel, content_code, style_mel, style_code)
        # Process NARModel
        full_codes = self.process_nar(content_mel, style_code, codes_0)
        code_alpha = full_codes.unsqueeze(0).detach()
        # Decode encodec
        with torch.no_grad():
            outputs = self.encodec_model.decode([(full_codes.unsqueeze(0), None)])
        wav0_gen = outputs[0, 0].detach().cpu().numpy()
        wav0_gen = librosa.resample(wav0_gen, orig_sr=24000, target_sr=16000)
        mel1_alpha = get_mel_spectrogram(wav0_gen)
        return {
            'mel': mel1,
            'code': code1,
            'mel_alpha': mel1_alpha,
            'wav_alpha': wav0_gen,
            'code_alpha': code_alpha[0]
        }

    def process_multistep_audio(self, mel1, code1, mel2, code2, max_style_len=3, max_content_len=21,
                                step_num=3):
        mel_alpha, code_alpha = mel1, code1
        outputs_list = []
        for i in range(step_num):
            outputs = self.process_audio(mel_alpha, code_alpha, mel2, code2, max_style_len, max_content_len)
            mel_alpha, code_alpha = outputs['mel_alpha'], outputs['code_alpha']
            outputs_list.append(outputs)
        return outputs_list
