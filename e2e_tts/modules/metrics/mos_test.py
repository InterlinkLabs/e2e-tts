import os
import tqdm
import json
import argparse
import speechmetrics


if __name__ == '__main__':
    # get information
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default=None, help='Path to test file')
    parser.add_argument('--eval_path', default=None, help='Path to audio path')
    parser.add_argument('--output_file', default=None, help='Path to output file')
    args = parser.parse_args()

    # load models
    window_length = 10  # seconds
    metrics = speechmetrics.load('mosnet', window_length)

    f = open(args.test_file, 'r', encoding='utf8')
    meta = [_.split('|') for _ in f.read().split('\n') if _]
    if len(meta[0]) == 4:
        # config
        MOS = {spk: {} for spk in set([_[0] for _ in meta])}
        # processing
        speakers = os.listdir(args.eval_path)
        for line in tqdm.tqdm(meta):
            MOS[line[0]][line[1]] = float(metrics(os.path.join(args.eval_path, line[0], line[1]))['mosnet'][0])
        for spk, score in MOS.items():
            score = sum(score.values()) / len(score)
            print('{}: {}'.format(spk, round(score, 2)))
    else:
        MOS = {line[0]: float(metrics(os.path.join(args.eval_path, os.path.basename(line[0])))['mosnet'][0]) for line in tqdm.tqdm(meta)}
        score = sum(MOS.values()) / len(MOS)
        print('MOS score: {}'.format(round(score, 2)))

    f = open(args.output_file, 'w', encoding='utf8')
    json.dump(MOS, f, ensure_ascii=False, indent=4)
