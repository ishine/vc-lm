import glob
import librosa
import os
import random
import soundfile as sf

from joblib.parallel import Parallel, delayed
from tqdm.auto import tqdm

min_time, max_time = 10, 24

def process_audio(input_file,
                  audio_id, output_dir):
    x, sr = librosa.load(input_file,
                         sr=16000)
    min_segment_size, max_segment_size = min_time * sr, max_time * sr
    x_len = x.shape[0]
    if x_len < min_segment_size:
        return 0
    segments = []
    pos = 0
    while pos < x_len:
        cur_segment_size = random.randint(min_segment_size, max_segment_size)
        if pos + cur_segment_size > x_len:
            cur_pos = min(pos, x_len - min_segment_size)
            segment = x[cur_pos:]
            segments.append(segment)
            break
        segment = x[pos:pos+cur_segment_size]
        segments.append(segment)
        pos += cur_segment_size
    for sub_id, segment in enumerate(segments):
        sf.write(os.path.join(output_dir,
                              f"{audio_id}_{sub_id}.wav"),
                 segment, sr, subtype='PCM_16')
    return sum([item.shape[0]/sr for item in segments])

files = []

for line in open('/home1/jiangxinghua/data/files.txt'):
    files.append(line.strip())

output_dir = '/home/jiangxinghua/data/audios'

r = Parallel(n_jobs=64)(delayed(process_audio)(filename, idx, output_dir) for idx, filename in tqdm(enumerate(files)))

print(f"total time: {sum(r)/3600} h")

with open('outputs.txt', 'w') as fout:
    fout.write(f"total time: {sum(r)/3600} h")


