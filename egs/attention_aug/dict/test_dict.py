from g2p_en import G2p
import string
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend

# from g2p import make_g2p
# transducer = make_g2p('dan', 'eng-arpabet')

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
    # TODO: Suprasegmentals in wiki: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet#Pitch_and_tone
    'ER' : 'ɜː',
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

cmu_vowel_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']

cmu_consonant_phones = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH']

# with open('./cmudict.phones', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         parts = line.split('\t')
#         phone = parts[0]
#         p_type = parts[1].strip()
#         if p_type == 'vowel':
#             cmu_vowel_phones.append(phone)
#         else:
#             cmu_consonant_phones.append(phone)

# print(cmu_vowel_phones, len(cmu_vowel_phones))
# print(cmu_consonant_phones, len(cmu_consonant_phones))

# init cmu ipa dict
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

cmudict_plain = {}
with open('./cmudict.dict', 'r') as f:
    lines = f.readlines()
    for l in lines:
        parts = l.split(' ')
        parts = [p.strip() for p in parts]
        word = parts[0].lower()
        phonetic = parts[1:]
        cmudict_plain[word] = phonetic

# exit()
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
backend = EspeakBackend('en-us')

def look_up_ipa_dict(word):
    l = cmudict_ipa.get(word, None)
    if l:
        return l[0]
    else:
        return word

def look_up_plain_dict(word, stress = False):
    phones = cmudict_plain.get(word, None)
    if phones:
        if stress:
            vowels = [(i, p) for i, p in enumerate(phones) if p not in cmu_consonant_phones]

            if not vowels:
                pass
            else:
                i = 0
                while i < len(vowels):
                    if vowels[i][1][2] == '1':
                        break
                    i += 1
            
                if i == 0:
                    pass
                else:
                    index = vowels[i][0] - 1
                    # add stress
                    phones.insert(index, 'ˈ')
            
        phones = [p.rstrip(string.digits)  if p != 'AH0' else p for p in phones]
        phones = [ipa_symbols_from_wiki.get(p, p) for p in phones]
        print(phones)
        return "".join(phones)
    else:
        return word

def use_g2p_en(word, stress = False):
    phones = g2p(word)
    if stress:
        vowels = [(i, p) for i, p in enumerate(phones) if p not in cmu_consonant_phones]

        if not vowels:
            pass
        else:
            i = 0
            while i < len(vowels):
                if vowels[i][1][2] == '1':
                    break
                i += 1
            
            if i == 0:
                pass
            else:
                index = vowels[i][0] - 1
                # add stress
                phones.insert(index, 'ˈ')
            
    phones = [p.rstrip(string.digits)  if p != 'AH0' else p for p in phones]
    phones = [ipa_symbols_from_wiki.get(p, p) for p in phones]

    return "".join(phones)

def use_phonemizer(word):
    return backend.phonemize([word])[0]

# word
# words = ["about", "through", "rough", "cough", "content", "ought", "magazine", "hurt", "but", "accept", "2", "talked", "bananas", "wishes", "OPPO"]
words = ['suburban', 'kit']
for word in words:
    # print(word, look_up_ipa_dict(word), use_g2p_en(word), use_phonemizer(word))
    # print(word, look_up_ipa_dict(word), use_g2p_en(word, True), look_up_plain_dict(word, True))
    print(word, look_up_ipa_dict(word), use_g2p_en(word, True), look_up_plain_dict(word, True), use_phonemizer(word))

exit()

# sentence
texts = ["I have $250 in my pocket.", # number -> spell-out
         "popular pets, e.g. cats and dogs", # e.g. -> for example
         "I refuse to collect the refuse around here.", # homograph
         "I'm an activationist.",
         "The apple of my eye at 7 am run off."
         ] # newly coined word
for sentence in texts:
    print(sentence)
    print(use_g2p_en(sentence))
    # print(use_phonemizer(sentence))
