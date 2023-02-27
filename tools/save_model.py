import torch
import fire

def save_model(input_model,
               output_model):
    m = torch.load(input_model, map_location=torch.device('cpu'))
    del m['optimizer_states']
    torch.save(m, output_model)


if __name__ == '__main__':
    fire.Fire(save_model)