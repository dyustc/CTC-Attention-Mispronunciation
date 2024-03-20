from g2p_en import G2p
import string
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from typing import List
import sys
import csv
import os
import warnings

ECDICT_PATH = os.path.abspath('/'.join([os.path.dirname(__file__), '..', '..', '..', 'ECDICT']))
sys.path.append(ECDICT_PATH)
from stardict import DictCsv

class Phonetic(object):
    def __init__(self) -> None:
        self.cmu_to_ipa_wiki = {
            # TODO: ɒ vs ɑ ɑːd dɒɡ hɑːt a hɑːrt
            "AA" : 'a',
            'AE' : 'æ',
            'AH0' : 'ə',
            'AH'  : 'ʌ',
            'AO' : 'ɔ',
            'AW' : 'aʊ',
            'AY' : 'aɪ',
            'EH' : 'e',
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

        self.cmu_to_ipa = {
            "a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
            "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
            "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
            "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"
        }
        
        self.ipa_to_cmu_wiki = {}

        for k, v in self.cmu_to_ipa_wiki.items():
            self.ipa_to_cmu_wiki[v] = k
        self.ipa_to_cmu_wiki['ɜ'] = 'ER'

        self.cmu_phones = list(self.cmu_to_ipa_wiki.keys())
        self.cmu_phones.remove('AH0')

        self.cmu_vowel_phones = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 
            'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']

        self.cmu_consonant_phones = [
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 
            'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 
            'Z', 'ZH']

        self.ipa_vowel_phones = []

        self.ipa_consonant_phones = []

        for p in self.cmu_vowel_phones:
            self.ipa_vowel_phones.append(self.cmu_to_ipa_wiki[p])
        self.ipa_vowel_phones.append(self.cmu_to_ipa_wiki['AH0'])

        for p in self.cmu_consonant_phones:
            self.ipa_consonant_phones.append(self.cmu_to_ipa_wiki[p])
        
        self.cmudict_ipa = {}
        self.cmudict_plain = {}
        self.letter_ipa_dict = {}
        self.dc = {}

        self.g2p = G2p()
        self.backend = EspeakBackend('en-us', with_stress = True)

    def load_ecdict(self, reload = False) -> None:
        if self.dc.__len__() == 0 or reload:
            csvname = os.path.join(ECDICT_PATH, 'ecdict.csv')
            self.dc = DictCsv(csvname)
        
        return
   
    def load_cmudict_ipa(self, reload = False) -> None:
        if self.cmudict_ipa.__len__() == 0 or reload:
            # init cmu ipa dict
            with open('./cmudict-0.7b-ipa.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.split('\t')
                    parts = [p.strip() for p in parts]
                    word = parts[0].lower()
                    phonetic = parts[1].split(',')
                    phonetic = [p.strip(' ˈˌ') for p in phonetic]

                    self.cmudict_ipa[word] = phonetic
    
        return 

    def load_cmudict_plain(self, reload = False) -> None:
        if self.cmudict_plain.__len__() == 0 or reload:
            with open('./cmudict.dict', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    parts = l.split(' ')
                    parts = [p.strip() for p in parts]
                    word = parts[0].lower()
                    phonetic = parts[1:]
                    self.cmudict_plain[word] = phonetic
        
        return

    def load_letter_ipa_dict(self, reload = False) -> None:
        if self.letter_ipa_dict.__len__() == 0 or reload:
            with open('./phonics_engine.csv', newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
                for row in spamreader:
                    word = row[0]
                    phonetic = row[2].replace('.', '')
                    mapping_string = row[3]
                    parts = mapping_string.split(',')
                    mapping = []
                    for p in parts:
                        subparts = p.split('-')
                        mapping.append(tuple(subparts))
                
                    self.letter_ipa_dict[word] = {
                        'phonetic' : phonetic,
                        'mapping' : mapping
                    }
        return
                
    def check_phone_map(self) -> None:
        for phone in self.cmu_phones:
            if phone not in self.cmu_to_ipa_wiki:
                print(phone)

            p1 = self.cmu_to_ipa.get(phone.lower(), phone.lower())
            p2 = self.cmu_to_ipa_wiki.get(phone, phone.lower())

            if p1 != p2:
                print(phone, p1, p2)
    
    def dc_dict(self, word) -> dict:
        self.load_ecdict()

        result = self.dc.query(word.lower())
        # fields = [
        #     'word', 
        #     'translation', 
        #     'phonetic', 
        #     'exchange', 
        #     'audio'
        #     ]

        if result is None or not result.get('translation', None):
            message = f"Word {word} not found in dictionary."
            warnings.warn(message)
            return {}
        else:
            return result

    def word_phrase_translation(self, word) -> str:
        result = self.dc_dict(word)
        ret = result.get('translation', None)
        if ret:
            return ret
        else:
            # phrase inference use translation model
            return
    
    def word_phrase_dc_phonetic(self, word) -> str:
        result = self.dc_dict(word)
        ret = result.get('phonetic', None)
        if ret:
            return ret
        else:
            # phrase inference use g2p or phonemizer
            return

    def ipa_dict(self, word, normalized = True, to_phones = False, take_first = True) -> str:
        self.load_cmudict_ipa()
        phonetics = self.cmudict_ipa.get(word, None)
        
        if take_first:
            index = 0
        
        if phonetics:
            phonetic = phonetics[index]
            
            phonetic = phonetic.replace('ɝ', 'ɜ')
            phonetic = phonetic.replace('ˌ', '')

            if to_phones:
                # FIXME: stress position if different from phonemizer
                phones = self._ipa_to_phones39(phonetic)
                return " ".join(phones)

            if normalized:
                return phonetic
        else:
            return self.phonemizer(word, to_phones, normalized)

    def plain_dict(self, word, to_ipa = True, stress = True, take_first = True) -> str:
        self.load_cmudict_plain()
        phones = self.cmudict_plain.get(word, None)
        
        if phones:
            if to_ipa:
                phones = self._phones39_to_ipa(phones, stress)
                return "".join(phones)
            else:
                return " ".join(phones)
        else:
            return self.g2p_ex(word, to_ipa, stress)

    def _phones39_to_ipa(self, phones: List[str], stress: bool = True) -> List[str]:
        if stress:
            vowels = [(i, p) for i, p in enumerate(phones) if p not in self.cmu_consonant_phones]

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
                    elif phones[index].rstrip(string.digits) in self.cmu_vowel_phones:
                        phones.insert(index+1, 'ˈ')
                    else:
                        phones.insert(index, 'ˈ')
       
        phones = [p.rstrip(string.digits)  if p != 'AH0' else p for p in phones]
        phones = [self.cmu_to_ipa_wiki.get(p, p) for p in phones]
    
        return phones

    def _ipa_phonemizer_normalized(self, phonetic : str) -> str:
        phonetic = phonetic.replace('ɹ', 'r')
        phonetic = phonetic.replace('ɚr', 'ər')
        phonetic = phonetic.replace('ɚˈr', 'ər')
        phonetic = phonetic.replace('ɚ', 'ər')
        phonetic = phonetic.replace('ɛ', 'e')
        phonetic = phonetic.replace('ɐ', 'ə')
        phonetic = phonetic.replace('ᵻ', 'ɪ')
        phonetic = phonetic.replace('ɡ', 'g')
        phonetic = phonetic.replace('ɑ', 'a')
        phonetic = phonetic.replace('ɾ', 't')

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
            for p in self.ipa_vowel_phones:
                if p in phonetic[:index]:
                    is_first_vowel = False
                    break
            
            if is_first_vowel:
                phonetic = phonetic[:index] + phonetic[index+1:]
            else:
                if phonetic[index-1] in self.ipa_vowel_phones:
                    pass
                elif index-2 >= 0 and phonetic[index-2:index] in self.ipa_vowel_phones:
                    pass
                elif index-2 >= 0 and phonetic[index-2:index] == 'st':
                    phonetic = phonetic[:index-2] + 'ˈ'  + 'st' + phonetic[index+1:]
                else:
                    phonetic = phonetic[:index-1] + 'ˈ'  + phonetic[index-1] + phonetic[index+1:]
    
        return phonetic

    def _ipa_to_phones39(self, phonetic : str) -> List[str]:
        phonetic = phonetic.replace('ɹ', 'r')
        phonetic = phonetic.replace('ɚr', 'ər')
        phonetic = phonetic.replace('ɚˈr', 'ər')
        phonetic = phonetic.replace('ɚ', 'ər')
        phonetic = phonetic.replace('ɛ', 'e')
        phonetic = phonetic.replace('ɐ', 'ə')
        phonetic = phonetic.replace('ᵻ', 'ɪ')
        phonetic = phonetic.replace('ɡ', 'g')
        phonetic = phonetic.replace('ɑ', 'a')
        phonetic = phonetic.replace('ɾ', 't')
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
                if i + 3 <= len(phonetic) and phonetic[i+1:i+3] in self.ipa_to_cmu_wiki.keys():
                    phones.append(self.ipa_to_cmu_wiki[phonetic[i+1:i+3]] + stress_map[phonetic[i]])
                    i += 3
                elif i + 2 <= len(phonetic) and phonetic[i+1] in self.ipa_to_cmu_wiki.keys():
                    phones.append(self.ipa_to_cmu_wiki[phonetic[i+1]] + stress_map[phonetic[i]])
                    i += 2
            else:
                if i + 2 <= len(phonetic) and phonetic[i:i+2] in self.ipa_to_cmu_wiki.keys():
                    p = self.ipa_to_cmu_wiki[phonetic[i:i+2]]
                    if p in self.cmu_vowel_phones:
                        phones.append(p+'0')
                    else:
                        phones.append(p)
                
                    i += 2
                else:
                    p = self.ipa_to_cmu_wiki[phonetic[i]]
                    if p == 'AH0':
                        phones.append(p)
                    elif p in self.cmu_vowel_phones:
                        phones.append(p+'0')
                    else:
                        phones.append(p)
                    i += 1

        return phones

    def phonemizer(self, word, to_phones = False, normalized = True) -> str:
        phonetic = self.backend.phonemize([word])[0]
        phonetic = phonetic.strip()

        if to_phones:
            phones = self._ipa_to_phones39(phonetic)
            return " ".join(phones)

        if normalized:
            return self._ipa_phonemizer_normalized(phonetic)

    def g2p_ex(self, word, to_ipa = True, stress = True) -> str:
        phones = self.g2p(word)
        # print(phones)
        
        if to_ipa:
            phones = self._phones39_to_ipa(phones, stress)
            return "".join(phones)
        else:
            return " ".join(phones)

    def phonemizer_sentence(self, text, to_phones = False, normalized = True) -> str:
        phonetic = self.backend.phonemize([text])[0]
        phonetic = phonetic.strip()
        
        if to_phones:
            phonetics = phonetic.split(' ')
            phones = [self._ipa_to_phones39(p) for p in phonetics]
            full_phones = list()
            for p in phones:
                # TODO: better formatting
                full_phones = full_phones + p + [" "]
            return " ".join(full_phones), phones

        if normalized:
            phonetics = [self._ipa_phonemizer_normalized(p) for p in phonetic.split(' ')]
            return ' '.join(phonetics)

    def g2p_ex_sentence(self, text, to_phones = False, normalized = True) -> str:
        to_ipa = True if not to_phones else False
        stress = True if normalized else False
        
        if to_ipa:
            return self.g2p_ex(text, to_ipa, stress)
        else:
            ret = self.g2p_ex(text, to_ipa, stress)
            return ret, ret.split(' ')

def main():
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
    phonetic = Phonetic()
    phonetic.load_letter_ipa_dict()

    # word
    words0 = ["2"]
    words1 = ["about", "through", "rough", "cough", "content", "ought", "magazine", "hurt", "but", "accept", "talked", "bananas", "wishes", "OPPO"]
    words2 = ['suburban', 'kit', 'odd', 'outstanding', 'geology', 'ZZ', 'dashing', "good", 'longtimenosee', 'phoneme']
    words3 = ['vocabulary', 'algorithm', 'thorough', 'gathering', 'metal', 'pull', 'Toronto', 'hot', 'heart']
    words4 = ['rear', 'bear', 'tour', 'cats', 'tree', 'dog', 'dream', 'beds', 'brother', 'oat']
    words = words1 + words2 + words3
    # ɑːd dɒɡ hɑːt a hɑːrt
    # words = words4
    # words = ['about']
    for word in words:
        s0 = phonetic.word_phrase_translation(word)
        s0_0 = phonetic.word_phrase_dc_phonetic(word)
        s1 = phonetic.ipa_dict(word)
        s1_1 = phonetic.ipa_dict(word, True, True)
        s4 = phonetic.plain_dict(word)
        s2 = phonetic.phonemizer(word)
        s2_1 = phonetic.phonemizer(word, True)
        s3 = phonetic.g2p_ex(word)
        s3_1 = phonetic.g2p_ex(word, False)
        
        print(word, s2, s3, s0_0)
        print(s0)
        print()
        # print(s2_1)
        # print(s3_1)
        # print(word, s2, s3, s1, s4)
        # print(s1_1)
        # print(s2_1)
        # print(s3_1)
        # print(word, s1, s2, s3, s4, s1 == s2)
    exit()
    #sentence
    texts = [
        "I refuse to collect the refuse around here.", # homograph
        "I'm an activationist.",
        "The apple of my eye at 7 am run off.",
        # "I have $250 in my pocket.", # number -> spell-out
        # "popular pets, e.g. cats and dogs", # e.g. -> for example
        # "the",
        # "am"
    ] # newly coined word
    for sentence in texts:
        print(sentence)
        print(phonetic.phonemizer_sentence(sentence))
        print(phonetic.phonemizer_sentence(sentence, True, False))
        print(phonetic.g2p_ex(sentence, False))
        print("")
        # print(phonetic.g2p_ex(sentence, False))
    
if __name__ == '__main__':
    sys.exit(main())


# from g2p import make_g2p
# transducer = make_g2p('dan', 'eng-arpabet')

# Use a pipeline as a high-level helper
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pipe = pipeline("text2text-generation", model="charsiu/g2p_multilingual_byT5_small_100")
# tokenizer = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
# model = AutoModelForSeq2SeqLM.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
