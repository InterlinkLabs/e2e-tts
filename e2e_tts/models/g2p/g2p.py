import os
import json
import string
import itertools
from g2p_en import G2p
from typing import Union
from unidecode import unidecode


# config
with open(os.path.join(os.path.dirname(__file__), "dict/fix_words.txt"), "r", encoding="utf8") as f:
    vn_words = [x for x in f.read().split("\n") if x]
with open(os.path.join(os.path.dirname(__file__), "dict/foreign_words.json"), "r", encoding="utf8") as f:
    en_words = json.load(f)

vowels = ["a", "e", "i", "o", "u", "y"]
g2p_consonants = {"b": "b", "ch": "ch", "đ": "dd", "ph": "ph", "h": "h", "d": "d", "k": "k", "qu": "kw", "q": "k",
                  "c": "k", "l": "l", "m": "m", "n": "n", "nh": "nh", "ng": "ng", "ngh": "ng", "p": "p", "x": "x",
                  "s": "s", "t": "t", "th": "th", "tr": "tr", "v": "v", "kh": "kh", "g": "g", "gh": "g", "gi": "d",
                  "r": "r"}
g2p_medial = {"u": "wu", "o": "wo"}
g2p_monophthongs = {"ă": "aw", "ê": "ee", "e": "e", "â": "aa", "ơ": "ow", "y": "i", "i": "i", "ư": "uw", "ô": "oo",
                    "u": "u", "oo": "o", "o": "oa", "a": "a"}
g2p_diphthongs = {"yê": "ie", "iê": "ie", "ya": "ie", "ia": "ie", "ươ": "wa", "ưa": "wa", "uô": "uo", "ua": "uo"}
g2p_coda = {"m": "mz", "n": "nz", "ng": "ngz", "nh": "nhz", "p": "pz", "t": "tz", "ch": "kz", "k": "cz", "c": "cz",
            "u": "uz", "o": "oz", "y": "yz", "i": "iz"}

g2p_tone = {u"á": 1, u"à": 2, u"ả": 3, u"ã": 4, u"ạ": 5,
            u"ấ": 1, u"ầ": 2, u"ẩ": 3, u"ẫ": 4, u"ậ": 5,
            u"ắ": 1, u"ằ": 2, u"ẳ": 3, u"ẵ": 4, u"ặ": 5,
            u"é": 1, u"è": 2, u"ẻ": 3, u"ẽ": 4, u"ẹ": 5,
            u"ế": 1, u"ề": 2, u"ể": 3, u"ễ": 4, u"ệ": 5,
            u"í": 1, u"ì": 2, u"ỉ": 3, u"ĩ": 4, u"ị": 5,
            u"ó": 1, u"ò": 2, u"ỏ": 3, u"õ": 4, u"ọ": 5,
            u"ố": 1, u"ồ": 2, u"ổ": 3, u"ỗ": 4, u"ộ": 5,
            u"ớ": 1, u"ờ": 2, u"ở": 3, u"ỡ": 4, u"ợ": 5,
            u"ú": 1, u"ù": 2, u"ủ": 3, u"ũ": 4, u"ụ": 5,
            u"ứ": 1, u"ừ": 2, u"ử": 3, u"ữ": 4, u"ự": 5,
            u"ý": 1, u"ỳ": 2, u"ỷ": 3, u"ỹ": 4, u"ỵ": 5,
            }
remove_tone = {u"á": u"a", u"à": u"a", u"ả": u"a", u"ã": u"a", u"ạ": u"a",
               u"ấ": u"â", u"ầ": u"â", u"ẩ": u"â", u"ẫ": u"â", u"ậ": u"â",
               u"ắ": u"ă", u"ằ": u"ă", u"ẳ": u"ă", u"ẵ": u"ă", u"ặ": u"ă",
               u"é": u"e", u"è": u"e", u"ẻ": u"e", u"ẽ": u"e", u"ẹ": u"e",
               u"ế": u"ê", u"ề": u"ê", u"ể": u"ê", u"ễ": u"ê", u"ệ": u"ê",
               u"í": u"i", u"ì": u"i", u"ỉ": u"i", u"ĩ": u"i", u"ị": u"i",
               u"ó": u"o", u"ò": u"o", u"ỏ": u"o", u"õ": u"o", u"ọ": u"o",
               u"ố": u"ô", u"ồ": u"ô", u"ổ": u"ô", u"ỗ": u"ô", u"ộ": u"ô",
               u"ớ": u"ơ", u"ờ": u"ơ", u"ở": u"ơ", u"ỡ": u"ơ", u"ợ": u"ơ",
               u"ú": u"u", u"ù": u"u", u"ủ": u"u", u"ũ": u"u", u"ụ": u"u",
               u"ứ": u"ư", u"ừ": u"ư", u"ử": u"ư", u"ữ": u"ư", u"ự": u"ư",
               u"ý": u"y", u"ỳ": u"y", u"ỷ": u"y", u"ỹ": u"y", u"ỵ": u"y",
               }


# code
en_convert = G2p()
def vi_convert(graph: str) -> list:
    """Tone location: Location of tone in phonemes of word input form: {inside, last, both}
    Two type of phonemes:
    - Tone at end of syllable: C1wVC2T | _consonant, _medial, _vowel, _coda, tone
    - Tone after vowel: C1wVTC2
    - Tone present both: C1wVTC2T
    """
    
    # connectred phonemes
    if len(graph) == 1 and graph in g2p_consonants:

        return [g2p_consonants[graph]]

    # initilize tone 
    tone = "0"
    graph = list(graph)
    for i, w in enumerate(graph):
        if w in g2p_tone:
            tone = "{}".format(g2p_tone[w])
            graph[i] = remove_tone[w]
            break

    graph = "".join(graph)
    # initilize phonemes
    phone = [graph[0]]
    for i in range(1, len(list(graph))):
        if (unidecode(graph[i]) in vowels and unidecode(graph[i - 1]) not in vowels) \
                or (unidecode(graph[i]) not in vowels and unidecode(graph[i - 1]) in vowels):
            phone.append(" | " + graph[i])
        else:
            phone.append(graph[i])

    phone = [x.strip() for x in "".join(phone).split("|")]
    if unidecode(phone[0][0]) in vowels:
        phone = [""] + phone
    phone.extend(["" for x in range(3 - len(phone))])

    # get consonants
    uni_phone = [unidecode(x) for x in phone]
    # get medial and semi-vowels
    if phone[1]:
        if uni_phone[0] == "g" and uni_phone[1][0] == "i":
            phone[0] = "d"
            phone[1] = phone[1] if uni_phone[1] in ["i", "ieu"] or (phone[1] == "iê" and phone[2]) else phone[1][1:]
        elif uni_phone[0] == "q" and uni_phone[1][0] == "u":
            phone[0] = "qu" if phone[1] != "u" else "c"
            phone[1] = phone[1][1:] if uni_phone[1] != "u" else phone[1]

        if len(phone[1]) > 1:
            if phone[1][-1] in ["u", "o", "i", "y"] and phone[1] not in g2p_diphthongs and not phone[2]:
                phone[2] = phone[1][-1]
                phone[1] = phone[1][:-1]
            if phone[1][0] in ["u", "o"] and phone[1] not in g2p_diphthongs and phone[1] != "oo":
                phone[1] = phone[1][0] + " " + phone[1][1:]

    # re-get consonants
    _consonant = g2p_consonants[phone[0]] if phone[0] in g2p_consonants else ""
    if phone[1]:
        phone[1] = phone[1].split()
        # special phonemes o (this must try)
        phone[1][-1] = "oo" if len(phone[1]) == 1 and phone[1][-1] == "o" and phone[2] in ["n", "t", "i"] \
            else phone[1][-1]
        _medial = g2p_medial[phone[1][0]] if len(phone[1]) == 2 else ""
        _vowel = g2p_diphthongs[phone[1][-1]] if len(phone[1][-1]) == 2 and phone[1][-1] != "oo"\
            else g2p_monophthongs[phone[1][-1]]
    else:
        _medial = _vowel = ""

    # get conda
    _coda = g2p_coda[phone[2]] if phone[2] in g2p_coda else ""

    # format: C1 w_V_T C2 
    phone = [_consonant, _medial, f"{_vowel}_{tone}", _coda]

    return [x for x in phone if x]


def normalize_phonemes(text: Union[list, str], foreign_dict: dict=None, is_training: bool=True) -> list:
    if foreign_dict is None:
        foreign_dict = en_words
    sequences = text.split() if isinstance(text, str) else text
    if sequences[-1] not in list(string.punctuation):
        sequences.append(".")

    # initilize phonemes depend on sequence words
    for i, word in enumerate(sequences):
        if foreign_dict is not None and word in foreign_dict:
            if foreign_dict[word]["phonemes"] is not None:
                if "|" in foreign_dict[word]["phonemes"]:
                    sequences[i] = [[f"@{ph[:-1] if ph[-1].isdigit() else ph}" for ph in x.strip().split()] for x in foreign_dict[word]["phonemes"].split("|")]
                else:
                    sequences[i] = [f"@{ph[:-1] if ph[-1].isdigit() else ph}" for ph in foreign_dict[word]["phonemes"].split()]
            else:
                sequences[i] = [vi_convert(x) for x in foreign_dict[word]["subtitle"].split("-")]
            print(f"[*] word(condition 0): {word} => sequences: {sequences[i]}")
        else:
            if "-" in word:
                sequences[i] = [vi_convert(x) for x in word.split("-") if len(x) > 0]
            else:
                if word in list(string.punctuation):
                    sequences[i] = ["</s>"] if i == len(sequences) - 1 else ["<silent>"]
                else:
                    sequences[i] = vi_convert(word)

    # initilize phonemes with boundaries
    expand_sequences, boundaries = [], []
    for seq in sequences:
        if isinstance(seq[0], list):
            expand_sequences.extend([ph for w in seq for ph in w])
            if is_training is True:
                boundaries.extend([len(w) for w in seq])
            else:
                boundaries.append([len(w) for w in seq])
        else:
            expand_sequences.extend(seq)
            boundaries.append(len(seq))
    expand_sequences = [x.upper() for x in expand_sequences]

    return expand_sequences, boundaries
