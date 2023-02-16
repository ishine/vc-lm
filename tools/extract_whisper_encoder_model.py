import fire
import torch
import whisper

def extract_whisper_encoder_model(input_model='/root/autodl-tmp/cache/whisper/small.pt',
                                  output_model=None):
    checkpoint = torch.load('/root/autodl-tmp/cache/whisper/small.pt')
    dims = checkpoint['dims']
    model = whisper.load_model(input_model)
    model_state_dict = model.encoder.state_dict()
    torch.save({
        'dims': dims,
        'model_state_dict': model_state_dict
    }, output_model)


if __name__ == '__main__':
    fire.Fire(extract_whisper_encoder_model)