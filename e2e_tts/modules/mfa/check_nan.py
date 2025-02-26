import os
import sys
import tqdm
import numpy as np

if __name__ == '__main__':
    input_folder = sys.argv[1]
    with open(os.path.join(input_folder, 'file_list.txt'), 'r', encoding='utf8') as f:
        list_segments = [_ for _ in f.read().split('\n') if _]

    output_segments = []
    for segment in tqdm.tqdm(list_segments):
        file_name = segment.split('|')[0]
        if any(True in np.isnan(np.load(f"{file_name.split('.')[0].replace('/wavs/', pros)}.npy")) for pros in ['/pitch/', '/energy/']):
            # print('{} has NaN value'.format(file_name))
            continue
        output_segments.append(segment)
    
    print(f'from {len(list_segments)} to {len(output_segments)} segments!!!')
    with open(os.path.join(input_folder, 'file_list.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(output_segments))

