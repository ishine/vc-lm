import webdataset as wds

from vc_lm.vc_engine import VCEngineDataFactory
from tqdm.auto import tqdm
import numpy as np

def process_records(record_list, device_id=0):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    device_id = 0
    engine = VCEngineDataFactory('/root/autodl-tmp/vc-models/ar-1024.ckpt',
                                 '/root/autodl-tmp/vc-models/nar-1024.ckpt',
                                 'configs/ar_model.json',
                                 'configs/nar_model.json',
                                 device=f'cuda:{device_id}')
    output_records = []
    for record in tqdm(record_list):
        mel1, code1, mel2, code2 = record['mel1'], record['code1'], record['mel2'], record['code2']
        outputs_list = engine.process_multistep_audio(mel1, code1, mel2, code2)
        for outputs in outputs_list:
            output_mel = outputs['mel_alpha']
            output_code = outputs['code']
            mel_len = output_mel.shape[1] / 100
            code_len = output_code.shape[1] / 75
            if code_len > mel_len:
                output_code = output_code[:, 0:int(mel_len * 75)]
            else:
                output_mel = output_mel[:, 0:int(code_len * 100)]
            sub_record_idx = len(output_records)
            output_records.append(
                {
                    '__key__': "%011d" % sub_record_idx,
                    'data.pyd': {
                        'mel': output_mel,
                        'code': output_code.detach().cpu().numpy().astype(np.int16)
                    }
                }
            )
    return output_records


dataset1 = wds.WebDataset("/root/autodl-tmp/data/lyh-wds/train/shard-000000.tar")
dataset1 = dataset1.decode()

dataset2 = wds.WebDataset("/root/autodl-tmp/data/wds/train/shard-000000.tar")
dataset2 = dataset2.decode()


dataset2 = iter(dataset2)

import os

index = 0

records = []

for i in range(18):
    for record_idx, item1 in tqdm(enumerate(dataset1)):
        # if record_idx >= 40/2:
        #     continue
        item2 = next(dataset2)
        obj1, obj2 = item1['data.pyd'], item2['data.pyd']
        mel1, code1, mel2, code2 = obj1['mel'], obj1['code'], obj2['mel'], obj2['code']
        records.append({
            'index': index,
            'mel1': mel1,
            'code1': code1,
            'mel2': mel2,
            'code2': code2
        })
        index += 1

print(index)

from joblib.parallel import Parallel, delayed

n_jobs = 4
num_devices = 2
segment_num = int(len(records)/n_jobs)

result_list = Parallel(n_jobs=n_jobs)(delayed(process_records)(records[i*segment_num:(i+1)*segment_num],
                                                 i % num_devices) \
                        for i in range(n_jobs))
outputs = []
for item in result_list:
    outputs.extend(item)

with wds.ShardWriter(os.path.join("/root/autodl-tmp/data/lyh-p-wds/train", 'shard-%06d.tar'),
                     maxcount=10000000, maxsize=1 << 32) as sink:
    for record in outputs:
        sink.write(record)

