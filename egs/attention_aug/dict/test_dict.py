from g2p_en import G2p
import string
from phonemizer import phonemize
# from g2p import make_g2p

# Use a pipeline as a high-level helper
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pipe = pipeline("text2text-generation", model="charsiu/g2p_multilingual_byT5_small_100")
# tokenizer = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
# model = AutoModelForSeq2SeqLM.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")

ipa_symbols = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
           "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
           "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
           "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

ipa_symbols_from_wiki = {
    "AA" : 'a',
    'AE' : 'æ',
    'AH0' : 'ə',
    'AH'  : 'ʌ',
    'AO' : 'ɔ',
    'AW' : 'aʊ',
    'AY' : 'aɪ',
    'EH' : 'ɛ',
    'ER' : 'ɜ',
    'EY' : 'eɪ',
    'IH' : 'ɪ',
    'IY' : 'i',
    'OW' : 'oʊ',
    'OY' : 'ɔɪ',
    'UH' : 'ʊ',
    'UW' : 'u',
    'B' : 'b',
    'CH' : 'tʃ',
    'D' : 'd',
    'DH' : 'ð',
    'F' : 'f',
    'G' : 'g',
    'HH' : 'h',
    'JH' : 'dʒ',
    'K' : 'k',
    'L' : 'l',
    'M' : 'm',
    'N' : 'n',
    'NG' : 'ŋ',
    'P' : 'p',
    'R' : 'r',
    'S' : 's',
    'SH' : 'ʃ',
    'T' : 't',
    'TH' : 'θ',
    'V' : 'v',
    'W' : 'w',
    'Y' : 'j',
    'Z' : 'z',
    'ZH' : 'ʒ'
}

cmu_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

cmudict_ipa = {}
with open('./cmudict-0.7b-ipa.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        parts = l.split('\t')
        parts = [p.strip() for p in parts]
        word = parts[0].lower()
        phonetic = parts[1].split(',')
        phonetic = [p.strip(' ˈˌ') for p in phonetic]

        cmudict_ipa[word] = phonetic

# print(cmudict_ipa)
# print(len(cmudict_ipa))
# print(cmu_phones)
# print(len(cmu_phones))
# print(len(ipa_symbols))
# print(len(ipa_symbols_from_wiki))

for phone in cmu_phones:
    if phone not in ipa_symbols_from_wiki:
        print(phone)

    p1 = ipa_symbols.get(phone.lower(), phone.lower())
    p2 = ipa_symbols_from_wiki.get(phone, phone.lower())

    if p1 != p2:
        print(phone, p1, p2)

g2p = G2p()
transducer = make_g2p('dan', 'eng-arpabet')

def look_up_ipa_dict(word):
    l = cmudict_ipa.get(word, None)
    if l:
        return l[0]
    else:
        return word

def use_g2p_en(word):
    phones = g2p(word)
    phones = [p.rstrip(string.digits)  if p != 'AH0' else 'AH0' for p in phones]
    phones = [ipa_symbols_from_wiki.get(p, p) for p in phones]

    return "".join(phones)

def use_phonemizer(word):
    return word

def look_up_cmu_dict(word):
    return word

# word
words = ["about", "through", "rough", "cough", "ought", "magazine", "hurt", "but", "accept", "2", "talked", "bananas", "wishes", "OPPO"]

for word in words:
    print(word, look_up_ipa_dict(word), use_g2p_en(word))

exit()

# sentence
texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist."] # newly coined word
words = ["hurt", 'bananas', 'unquote', "the apple of my eye at 7 am"]
for word in words:
    print(g2p(word))
    # print(pipe(word))

    # print(transducer(word).output_string)

    # out = phonemize(word, backend='espeak', language= 'en-us')
    # print(out)
# >>> ['AY1', ' ', 'HH', 'AE1', 'V', ' ', 'T', 'UW1', ' ', 'HH', 'AH1', 'N', 'D', 'R', 'AH0', 'D', ' ', 'F', 'IH1', 'F', 'T', 'IY0', ' ', 'D', 'AA1', 'L', 'ER0', 'Z', ' ', 'IH0', 'N', ' ', 'M', 'AY1', ' ', 'P', 'AA1', 'K', 'AH0', 'T', ' ', '.']
# >>> ['P', 'AA1', 'P', 'Y', 'AH0', 'L', 'ER0', ' ', 'P', 'EH1', 'T', 'S', ' ', ',', ' ', 'F', 'AO1', 'R', ' ', 'IH0', 'G', 'Z', 'AE1', 'M', 'P', 'AH0', 'L', ' ', 'K', 'AE1', 'T', 'S', ' ', 'AH0', 'N', 'D', ' ', 'D', 'AA1', 'G', 'Z']
# >>> ['AY1', ' ', 'R', 'IH0', 'F', 'Y', 'UW1', 'Z', ' ', 'T', 'UW1', ' ', 'K', 'AH0', 'L', 'EH1', 'K', 'T', ' ', 'DH', 'AH0', ' ', 'R', 'EH1', 'F', 'Y', 'UW2', 'Z', ' ', 'ER0', 'AW1', 'N', 'D', ' ', 'HH', 'IY1', 'R', ' ', '.']
# >>> ['AY1', ' ', 'AH0', 'M', ' ', 'AE1', 'N', ' ', 'AE2', 'K', 'T', 'IH0', 'V', 'EY1', 'SH', 'AH0', 'N', 'IH0', 'S', 'T', ' ', '.']