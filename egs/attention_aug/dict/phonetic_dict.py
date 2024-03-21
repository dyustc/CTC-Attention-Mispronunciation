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

        self.ipa_to_cmu_wiki = {}

        for k, v in self.cmu_to_ipa_wiki.items():
            self.ipa_to_cmu_wiki[v] = k
        # self.ipa_to_cmu_wiki['ɜ'] = 'ER'

        self.cmu_phones = list(self.cmu_to_ipa_wiki.keys())
        self.cmu_phones.remove('AH0')

        self.cmu_vowel_phones = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 
            'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']

        self.cmu_consonant_phones = [
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 
            'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 
            'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
            ]

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

        self.g2p_backend = G2p()
        self.backend_us = EspeakBackend('en-us', with_stress = True)
        self.backend_br = EspeakBackend('en', with_stress = True)

    def load_ecdict(self, reload = False) -> None:
        if self.dc.__len__() == 0 or reload:
            csvname = os.path.join(ECDICT_PATH, 'ecdict.csv')
            self.dc = DictCsv(csvname)
        
        return
   
    def load_ipadict(self, reload = False) -> None:
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

    def load_cmudict(self, reload = False) -> None:
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

    def dc_dict_word_translation(self, word) -> str:
        result = self.dc_dict(word)
        texts = result.get('translation', '')

        if not texts:
            return ''
        
        text = texts.split('\n')
        
        first_classes = [
            'n.', 'v.', 'vt.', 'vi.', 'a.', 'adj.', 
            'adv.', 'prep.', 'conj.', 'pron.',  'num.'
        ]
        
        second_classes = [
            'art.', 'int.', 'abbr.', 'aux.', 'modal.', 
            'phr.', 'idiom.'
        ]
        
        classes = first_classes + second_classes
        filtered_texts = []
        for t in text:
            c = t.split(' ')[0]
            if c in first_classes:
                if c == 'a.':
                    filtered_texts.append('adj.' + t[2:])
                else:
                    filtered_texts.append(t)
        
        if not filtered_texts:
            extra_info = result.get('exchange', None)
            if extra_info:
                infos = extra_info.split('/')
                parent = None
                for info in infos:
                    if info[:2] == '0:':
                        parent = info[2:]
                        break
                if not parent:
                    pass
                else:
                    return self.dc_dict_word_translation(parent)

        texts = '\n'.join(filtered_texts)
        
        return texts
    
    def dc_dict_phrase_translation(self, phrase) -> str:
        result = self.dc_dict(phrase)
        return result.get('translation', '')
    
    def dc_dict_phonetic(self, word) -> str:
        result = self.dc_dict(word)
        return result.get('phonetic', None)

    def api_word_translation(self, word) -> str:
        translation = self.dc_dict_word_translation(word)
        return translation
    
    def api_phrase_sentence_translation(self, text) -> str:
        translation = self.dc_dict_phrase_translation(text)
        return translation
    
    # TODO: update ecdict to fix or update meaning of words or phrases

    def _stress_normalize(self, phonetic : str) -> str:
        # change 1st stress position
        index = phonetic.find('ˈ')
        
        if index == -1:
            pass
        elif index == 0:
            phonetic = phonetic[1:]
        elif index == len(phonetic) - 1: # last stress
            phonetic = phonetic[:index]
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

    def _character_normalize(self, phonetic : str, style = 'us') -> str:
        phonetic = phonetic.replace('ɹ', 'r')
        phonetic = phonetic.replace('ɚr', 'ər')
        phonetic = phonetic.replace('ɚˈr', 'ər')
        phonetic = phonetic.replace('ɚ', 'ər')
        phonetic = phonetic.replace('ɛ', 'e')
        phonetic = phonetic.replace('ɐ', 'ə')
        phonetic = phonetic.replace('ᵻ', 'ɪ')
        phonetic = phonetic.replace('ɡ', 'g')
        
        if style == 'us':
            phonetic = phonetic.replace('ɑ', 'a')
        elif style == 'br':
            phonetic = phonetic.replace('a', 'æ')
            phonetic = phonetic.replace('æʊ', 'aʊ')
            phonetic = phonetic.replace('æɪ', 'aɪ')
            phonetic = phonetic.replace('ɑ', 'a')
    
        phonetic = phonetic.replace('ɾ', 't')
        phonetic = phonetic.replace('ɝ', 'ɜ')
        phonetic = phonetic.replace('iə', 'ɪə')

        return phonetic

    def _ipa_phonemizer_normalized(self, phonetic : str, style = 'us') -> str:
        phonetic = self._character_normalize(phonetic, style)

        # remove 2nd stress ˌ
        phonetic = phonetic.replace('ˌ', '')

        # move 1st stress ˈ
        phonetic = self._stress_normalize(phonetic)
        
        return phonetic

    def _ipa_to_phones39(self, phonetic : str) -> List[str]:
        phonetic = self._character_normalize(phonetic)
        
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
    
    def ipa_dict(self, word, index = 0) -> str:
        self.load_ipadict()
        phonetics = self.cmudict_ipa.get(word, None)
                
        if phonetics:
            if index >= len(phonetics):
                message = f"Word {word} only found {len(phonetics)} in dictionary."
                warnings.warn(message)
                index = 0
        
            phonetic = phonetics[index]

            return phonetic
        else:
            return None

    def phonemizer(self, word, style = 'us', to_phones = False) -> str:
        assert style in ['us', 'br']
        if style == 'us':
            backend = self.backend_us
        else:
            backend = self.backend_br
        
        phonetic = backend.phonemize([word])[0]
        phonetic = phonetic.strip()

        if to_phones:
            phones = self._ipa_to_phones39(phonetic)
            return " ".join(phones)
        else:
            return self._ipa_phonemizer_normalized(phonetic, style)

    def api_word_phonetic(self, word) -> str:
        phonetic_us = self.phonemizer(word, 'us')
        phonetic_br = self.phonemizer(word, 'br')
        ret = '英: /' + phonetic_br + '/ ' + '美: /' + phonetic_us + '/ '
        return ret

    def phonemizer_phrase_sentence(self, text, style='us') -> str:
        phonetic = self.phonemizer(text, style, False)
        
        parts = phonetic.split(' ')
        phonetic = ' '.join([self._stress_normalize(p) for p in parts])

        return phonetic

    def api_phrase_sentence_phonetic(self, text) -> str:
        phonetic_us = self.phonemizer_phrase_sentence(text, 'us')
        phonetic_br = self.phonemizer_phrase_sentence(text, 'br')

        ret = '英: /' + phonetic_br + '/ ' + '美: /' + phonetic_us + '/ '
        return ret

    def cmu_dict(self, word, to_ipa = False) -> str:
        self.load_cmudict()
        phones = self.cmudict_plain.get(word, None)
        
        if phones:
            if not to_ipa:
                return " ".join(phones)
            else:
                phones = self._phones39_to_ipa(phones, True)
                return "".join(phones)   
        else:
            return None

    def g2p(self, word, to_ipa = False) -> str:
        phones = self.g2p_backend(word)
        
        if to_ipa:
            phones = self._phones39_to_ipa(phones, True)
            return "".join(phones)
        else:
            return " ".join(phones)

    def g2p_sentence(self, text, to_ipa = False, normalized = True) -> str:
        stress = True if normalized else False
        
        if to_ipa:
            return self.g2p(text, to_ipa, stress)
        else:
            ret = self.g2p(text, to_ipa, stress)
            return ret, ret.split(' ')

def main():
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
    phonetic = Phonetic()
    phonetic.load_letter_ipa_dict()

    # word
    words0 = ["2"]
    words1 = ["apple", "about", "through", "rough", "cough", "content", "ought", "magazine", "hurt", "but", "accept", "talked", "bananas", "wishes", "OPPO"]
    words2 = ['suburban', 'kit', 'odd', 'outstanding', 'geology', 'ZZ', 'dashing', "good", 'longtimenosee', 'phoneme']
    words3 = ['vocabulary', 'algorithm', 'thorough', 'gathering', 'metal', 'pull', 'Toronto', 'hot', 'heart', 'mark', 'astronaut', 'ideal']
    words4 = ['rear', 'bear', 'tour', 'cat', 'tree', 'dog', 'dream', 'beds', 'brother', 'oat']
    words5 = ['jffsosejfi']

    words6 = ['cat', 'cats', 'kit', 'kits']
    words7 = ['talk', 'talks', 'talked', 'talking']
    words8 = ['good', 'goods', 'better', 'best']
    words9 = ['in', 'outstanding', 'odd', 'odds']
    words10 = ['two']
    words11 = ['tjjdjfporflker']

    # words = words0 + words1 + words2 + words3 + words4 + words5
    words = words6 + words7 + words8 + words9 + words10 + words11
    # words = ['cat']
    phrases = ["make up", "pass for something", "the apple of my eye", "get away with"]

    # exit()
    for word in words:
        s0 = phonetic.api_word_translation(word)
        # s1 = phonetic.dc_dict_phonetic(word)
        # s2 = phonetic.ipa_dict(word)
        # s4_1 = phonetic.phonemizer(word, 'us')
        # s4_2 = phonetic.phonemizer(word, 'br')
        # s3 = phonetic.cmu_dict(word)
        # s5 = phonetic.g2p(word)
        
        syllables = phonetic.api_word_phonetic(word)
        
        # print(word, s1, s2, s3, s4_2, s4_1, s5)
        # print(word, s4_2, s4_1)
        print(word, syllables)
        print(s0)
        print()
        # print(s2_1)
        # print(s3_1)
        # print(word, s2, s3, s1, s4)
        # print(s1_1)
        # print(s2_1)
        # print(s3_1)
        # print(word, s1, s2, s3, s4, s1 == s2)
    # exit()
    for phrase in phrases:
        # s1 = phonetic.phonemizer_phrase_sentence(phrase, 'us')
        # s2 = phonetic.phonemizer_phrase_sentence(phrase, 'br')
        s = phonetic.api_phrase_sentence_phonetic(phrase)
        s0 = phonetic.api_phrase_sentence_translation(phrase)
        # print(phrase, s1, s2)
        print(phrase, s)
        print(s0)
        print()
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
        print(phonetic.api_phrase_sentence_phonetic(sentence))
        # print(phonetic.phonemizer_sentence(sentence, True, False))
        # print(phonetic.g2p(sentence, False))
        print("")
        # print(phonetic.g2p(sentence, False))
    
if __name__ == '__main__':
    sys.exit(main())


# from g2p_backend import make_g2p
# transducer = make_g2p('dan', 'eng-arpabet')

# Use a pipeline as a high-level helper
# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# pipe = pipeline("text2text-generation", model="charsiu/g2p_multilingual_byT5_small_100")
# tokenizer = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
# model = AutoModelForSeq2SeqLM.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")
