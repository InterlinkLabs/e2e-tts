import os
import sys

sys.path.append(".")
import json
import string
import shutil
from tqdm import tqdm

from g2p_en import G2p
from models.g2p.g2p import normalize_phonemes as vi_convert


def build_lexicon(list_words: list, foreign_dicts: dict):
    list_words = sorted(list_words)
    lexicon = {k: [f"@{ph}" for ph in v[0]] for k, v in en_convert.cmu.items() if all(c  in string.ascii_letters for c in k)}
    lexicon.update({w: vi_convert(w)[0] for w in vn_words if w})

    for w in list_words:
        if w not in vn_words:
            lexicon[w] = [f"@{ph}" for ph in foreign_dicts[w].split()]
            # print(f"{w} -> {lexicon[w]}")

    return ["\t".join([k, " ".join(v)]) for k, v in lexicon.items()]


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    os.makedirs(output_folder, exist_ok=True)

    with open("models/g2p/dict/fix_wordss.txt", "r", encoding="utf8") as f:
        vn_words = [x for x in f.read().split("\n") if x]
    en_convert = G2p()
    with open(os.path.join(input_folder, "speakers.json"), "r", encoding="utf8") as f:
        list_speakers = json.load(f)
    print(f"Build mfa folder for {len(list_speakers)} speakers: {', '.join(list(list_speakers.keys()))}")
    list_words, list_foreigns = [], {}

    for spk in list_speakers:
        os.makedirs(os.path.join(output_folder, spk), exist_ok=True)
        with open(os.path.join(input_folder, spk, "metadata.csv"), "r", encoding="utf8") as f:
            metadata = [_.split("|") for _ in f.read().split("\n") if _]

        for line in tqdm(metadata, desc=f"Format {spk}", position=0):
            if not os.path.exists(os.path.join(input_folder, spk, "wavs", line[0])):
                # print(f"missing {os.path.join(input_folder, spk, "wavs", line[0])}")
                continue
            line[1] = " ".join([_.replace("-", " ") for _ in line[1].strip().split() if _ not in [",", "."]])
            list_words.extend(line[1].split())
            if not os.path.exists(os.path.join(output_folder, spk, line[0])):
                shutil.copy(os.path.join(input_folder, spk, "wavs", line[0]), 
                            os.path.join(output_folder, spk))
            with open(os.path.join(output_folder, spk, f"{line[0].split('.')[0]}.lab"), "w", encoding="utf8") as f:
                f.write(line[1].strip())

        with open(os.path.join(input_folder, spk, "foreign_words.json"), "r", encoding="utf8") as f:
            foreign_words = json.load(f)
        for w, ex in foreign_words.items():
            if w in list_foreigns and list_foreigns[w] != ex:
                print(f"{w}: {list_foreigns[w]} - {ex}")
                exit()

            list_foreigns[w] = ex

    list_words = list(set(list_words))
    with open(os.path.join(output_folder, "lexicon.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(build_lexicon(list_words, list_foreigns)))
