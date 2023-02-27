import torch
import fire

def save_model(input_model,
               output_model):
    m = torch.load(input_model, map_location=torch.device('cpu'))
    del m['optimizer_states']
    for k in list(m['state_dict'].keys()):
        v = m['state_dict'][k]
        del m['state_dict'][k]
        m['state_dict'][k.replace('linear3', 'linear1').replace('linear4', 'linear2')] = v
    torch.save(m, output_model)


if __name__ == '__main__':
    fire.Fire(save_model)