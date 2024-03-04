from g2p_en import G2p
import string
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from typing import List

# from g2p import make_g2p
# transducer = make_g2p('dan', 'eng-arpabet')

# Use a pipeline as a high-level helper
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pipe = pipeline("text2text-generation", model="charsiu/g2p_multilingual_byT5_small_100")
# tokenizer = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
# model = AutoModelForSeq2SeqLM.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")

cmu_to_ipa = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
           "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
           "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
           "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}

cmu_to_ipa_wiki = {
    "AA" : 'a',
    'AE' : 'æ',
    'AH0' : 'ə',
    'AH'  : 'ʌ',
    'AO' : 'ɔ',
    'AW' : 'aʊ',
    'AY' : 'aɪ',
    'EH' : 'ɛ',
    # TODO: Suprasegmentals in wiki: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet#Pitch_and_tone
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

ipa_to_cmu_wiki = {}

for k, v in cmu_to_ipa_wiki.items():
    ipa_to_cmu_wiki[v] = k

cmu_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 
              'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 
              'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 
              'W', 'Y', 'Z', 'ZH']

cmu_vowel_phones = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 
                    'IY', 'OW', 'OY', 'UH', 'UW']

cmu_consonant_phones = ['B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 
                        'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 
                        'Z', 'ZH']

ipa_vowel_phones = []

ipa_consonant_phones = []

for p in cmu_vowel_phones:
    ipa_vowel_phones.append(cmu_to_ipa_wiki[p])
ipa_vowel_phones.append(cmu_to_ipa_wiki['AH0'])

for p in cmu_consonant_phones:
    ipa_consonant_phones.append(cmu_to_ipa_wiki[p])

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

for phone in cmu_phones:
    if phone not in cmu_to_ipa_wiki:
        print(phone)

    p1 = cmu_to_ipa.get(phone.lower(), phone.lower())
    p2 = cmu_to_ipa_wiki.get(phone, phone.lower())

    if p1 != p2:
        print(phone, p1, p2)

g2p = G2p()
backend = EspeakBackend('en-us', with_stress = True)

def look_up_ipa_dict(word, normalized = True):
    phonetics = cmudict_ipa.get(word, None)
    if phonetics:
        phonetic = phonetics[0]
        if normalized:
            phonetic = phonetic.replace('ɝ', 'ɜ')
            phonetic = phonetic.replace('ˌ', '')
        
        return phonetic
    else:
        return word

def phones39_to_ipa(phones: List[str], stress: bool = True) -> List[str]:
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
                if phones[index] == 'T' and index - 1 >= 0 and phones[index-1] == 'S':
                    phones.insert(index-1, 'ˈ')
                else:
                    phones.insert(index, 'ˈ')
       
    phones = [p.rstrip(string.digits)  if p != 'AH0' else p for p in phones]
    phones = [cmu_to_ipa_wiki.get(p, p) for p in phones]
    
    return phones

def look_up_plain_dict(word, to_ipa = True, stress = False):
    phones = cmudict_plain.get(word, None)
    if phones:
        if to_ipa:
            phones = phones39_to_ipa(phones, stress)
            return "".join(phones)
        else:
            return " ".join(phones)
    else:
        return word

def use_g2p_en(word, to_ipa = True, stress = False):
    phones = g2p(word)
    if to_ipa:
        phones = phones39_to_ipa(phones)
        return "".join(phones)
    else:
        return " ".join(phones)

def ipa_phonemizer_normalized(phonetic : str) -> str:
    phonetic = phonetic.replace('ɹ', 'r')
    phonetic = phonetic.replace('ɐ', 'ə')
    phonetic = phonetic.replace('ᵻ', 'ɪ')
    phonetic = phonetic.replace('ɡ', 'g')
    phonetic = phonetic.replace('ɑ', 'a')
    
    # remove 2nd stress ˌ
    phonetic = phonetic.replace('ˌ', '')

    # change 1st stress position
    index = phonetic.find('ˈ')
    if index == -1:
        pass
    elif index == 0:
        phonetic = phonetic[1:]
    else:
        is_first_vowel = True
        for p in ipa_vowel_phones:
            if p in phonetic[:index]:
                is_first_vowel = False
                break
            
        if is_first_vowel:
            phonetic = phonetic[:index] + phonetic[index+1:]
        else:
            if phonetic[index-1] in ipa_vowel_phones:
                pass
            elif index-2 >= 0 and phonetic[index-2:index] in ipa_vowel_phones:
                pass
            elif index-2 >= 0 and phonetic[index-2:index] == 'st':
                phonetic = phonetic[:index-2] + 'ˈ'  + 'st' + phonetic[index+1:]
            else:
                phonetic = phonetic[:index-1] + 'ˈ'  + phonetic[index-1] + phonetic[index+1:]
    
    return phonetic

def ipa_to_phones39(phonetic : str) -> List[str]:
    phonetic = phonetic.replace('ɹ', 'r')
    phonetic = phonetic.replace('ɐ', 'ə')
    phonetic = phonetic.replace('ᵻ', 'ɪ')
    phonetic = phonetic.replace('ɡ', 'g')
    phonetic = phonetic.replace('ɑ', 'a')
    phonetic = phonetic.replace('ː', '')

    phones = []
    i = 0

    stress_map = {
        'ˌ' : '2',
        'ˈ' : '1'
    }

    while i < len(phonetic):
        # print(i, phonetic[i], phonetic)
        if phonetic[i] == 'ˌ' or phonetic[i] == 'ˈ':
            if i + 3 <= len(phonetic) and phonetic[i+1:i+3] in ipa_to_cmu_wiki.keys():
                phones.append(ipa_to_cmu_wiki[phonetic[i+1:i+3]] + stress_map[phonetic[i]])
                i += 3
            elif i + 2 <= len(phonetic) and phonetic[i+1] in ipa_to_cmu_wiki.keys():
                phones.append(ipa_to_cmu_wiki[phonetic[i+1]] + stress_map[phonetic[i]])
                i += 2
        else:
            if i + 2 <= len(phonetic) and phonetic[i:i+2] in ipa_to_cmu_wiki.keys():
                p = ipa_to_cmu_wiki[phonetic[i:i+2]]
                if p in cmu_vowel_phones:
                    phones.append(p+'0')
                else:
                    phones.append(p)
                
                i += 2
            else:
                p = ipa_to_cmu_wiki[phonetic[i]]
                if p == 'AH0':
                    phones.append(p)
                elif p in cmu_vowel_phones:
                    phones.append(p+'0')
                else:
                    phones.append(p)
                i += 1

    return phones

def use_phonemizer(word, to_phones = False, normalized = True):
    phonetic = backend.phonemize([word])[0]
    phonetic = phonetic.strip()

    if to_phones:
        phones = ipa_to_phones39(phonetic)
        return " ".join(phones)

    if normalized:
        return ipa_phonemizer_normalized(phonetic)

# word
words1 = ["about", "through", "rough", "cough", "content", "ought", "magazine", "hurt", "but", "accept", "2", "talked", "bananas", "wishes", "OPPO"]
words2 = ['suburban', 'kit', 'odd', 'outstanding', 'geology', 'ZZ', 'dashing', "good", 'longtimenosee']

words = words1 + words2
for word in words:
    s1 = look_up_ipa_dict(word)
    s2 = use_phonemizer(word)
    s2_1 = use_phonemizer(word, True, True)
    s3 = use_g2p_en(word, True, True)
    s3_1 = use_g2p_en(word, False)
    s4 = look_up_plain_dict(word, False)
    
    print(word, s2, s3)
    print(s2_1)
    print(s3_1)
    # print(word, s1, s2, s3, s4, s1 == s2)
    

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
    print(use_phonemizer(sentence))
