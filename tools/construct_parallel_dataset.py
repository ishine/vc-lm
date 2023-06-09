import webdataset as wds
import fire
import os
from vc_lm.vc_engine import VCEngineDataFactory
from tqdm.auto import tqdm
import numpy as np

def process_records(record_list,
                    device_id=0,
                    ar_model_path='/root/autodl-tmp/vc-models/ar-1024.ckpt',
                    nar_model_path='/root/autodl-tmp/vc-models/nar-1024.ckpt',
                    ar_config_file='configs/ar_model.json',
                    nar_config_file='configs/nar_model.json'):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    device_id = 0
    engine = VCEngineDataFactory(ar_model_path,
                                 nar_model_path,
                                 ar_config_file,
                                 nar_config_file,
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

def construct_parallel_dataset(input_data_path: str ="/root/autodl-tmp/jr_dataset/shard-000000.tar",
                               ref_data_path: str ="/root/autodl-tmp/shard-000000.tar",
                               repeat_num: int = 3,
                               num_devices: int = 1,
                               output_dir: str = "/root/autodl-tmp/data/jr-wds-pair/train",
                               ar_model_path = '/root/autodl-tmp/vc-models/ar-1024.ckpt',
                               nar_model_path = '/root/autodl-tmp/vc-models/nar-1024.ckpt',
                               ar_config_file = 'configs/ar_model.json',
                               nar_config_file = 'configs/nar_model.json'):
    """
    Args:
        input_data_path: str. The target person's voice audio file.
        ref_data_path: str. files consisting of a large number of different voices, used for prompts.
        repeat_num: int. The number of repetitions in constructing the dataset.
        output_dir: str.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset1 = wds.WebDataset(input_data_path)
    dataset1 = dataset1.decode()

    dataset2 = wds.WebDataset(ref_data_path)
    dataset2 = dataset2.decode()

    dataset2 = iter(dataset2)


    index = 0

    records = []

    for i in range(repeat_num):
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

    n_jobs = num_devices
    segment_num = int(len(records) / n_jobs)

    result_list = Parallel(n_jobs=n_jobs)(delayed(process_records)(records[i * segment_num:(i + 1) * segment_num],
                                                                   i % num_devices,
                                                                   ar_model_path,
                                                                   nar_model_path,
                                                                   ar_config_file,
                                                                   nar_config_file) \
                                          for i in range(n_jobs))
    outputs = []
    for item in result_list:
        outputs.extend(item)

    with wds.ShardWriter(os.path.join(output_dir, 'shard-%06d.tar'),
                         maxcount=10000000, maxsize=1 << 32) as sink:
        for record in outputs:
            sink.write(record)

if __name__ == '__main__':
    fire.Fire(construct_parallel_dataset)
