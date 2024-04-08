from g2p_en import G2p
import string
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from typing import List
import sys
import csv
import os
import warnings
import platform
import time
# from melo.api import TTS

ECDICT_PATH = os.path.abspath('/'.join([os.path.dirname(__file__), '..', '..', '..', 'ECDICT']))
sys.path.append(ECDICT_PATH)
from stardict import DictCsv

class Phonetic(object):
    def __init__(self) -> None:
        # This ipa mapping is done as the us syllable way.
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
            'ER0' : 'ər',
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

        self.cmu_vowel_phones = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 
            'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
            'AH0', 'ER0']

        self.cmu_consonant_phones = [
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 
            'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 
            'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
            ]

        self.ipa_vowel_phones = []

        self.ipa_consonant_phones = []

        for p in self.cmu_vowel_phones:
            self.ipa_vowel_phones.append(self.cmu_to_ipa_wiki[p])
        self.ipa_vowel_phones.remove(self.cmu_to_ipa_wiki['ER0'])

        for p in self.cmu_consonant_phones:
            self.ipa_consonant_phones.append(self.cmu_to_ipa_wiki[p])
        
        self.cmudict_ipa = {}
        self.cmudict_plain = {}
        self.letter_ipa_dict = {}
        self.dc = {}

        self.g2p_backend = G2p()
        self.backend_us = EspeakBackend('en-us', with_stress = True)
        self.backend_br = EspeakBackend('en', with_stress = True)

        self.wav_dir = os.path.join(os.path.dirname(__file__), 'wav')
        if not os.path.exists(self.wav_dir):
            os.makedirs(self.wav_dir)

        self.speed = 0.7
        self.accent ='Default'
        # self.tts_model = TTS(language='EN', device='cpu')

    def load_ecdict(self, reload = False) -> None:
        if self.dc.__len__() == 0 or reload:           
            csvname = os.path.join(ECDICT_PATH, 'ecdict.csv')
            self.dc = DictCsv(csvname)
        
        return
   
    def load_ipadict(self, reload = False) -> None:
        if self.cmudict_ipa.__len__() == 0 or reload:
            IPADICT_PATH = os.path.abspath('/'.join([os.path.dirname(__file__), 'cmudict-0.7b-ipa.txt']))
            with open(IPADICT_PATH, 'r') as f:
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
            CMUDICT_PATH = os.path.abspath('/'.join([os.path.dirname(__file__), 'cmudict.dict']))
            with open(CMUDICT_PATH, 'r') as f:
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
            LETTERIPADICT_PATH = os.path.abspath('/'.join([os.path.dirname(__file__), 'phonics_engine.csv']))
            with open(LETTERIPADICT_PATH, newline='') as csvfile:
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
    
    def dc_dict(self, text) -> dict:
        self.load_ecdict()
        text = text.strip()
        result = self.dc.query(text.lower())
        # fields = [
        #     'word', 
        #     'translation', 
        #     'phonetic', 
        #     'exchange', 
        #     'audio'
        #     ]

        if result is None or not result.get('translation', None):
            parts = text.split(' ')
            text_type = 'WORD'
            if len(parts) > 1:
                text_type = 'PHRASE'
            message = f"{text_type} '{text}' not found in dictionary."
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
        
        if filtered_texts:
            texts = '\n'.join(filtered_texts)
        
        texts = texts.strip()

        return texts
    
    def dc_dict_phrase_translation(self, phrase) -> str:
        result = self.dc_dict(phrase)
        return result.get('translation', '')
    
    def dc_dict_phonetic(self, word) -> str:
        result = self.dc_dict(word)
        return result.get('phonetic', None)
   
    # TODO: update ecdict to fix or update meaning of words or phrases

    def _stress_normalize(self, phonetic : str) -> str:
        # change 1st stress position
        index = phonetic.find('ˈ')
        
        if index == -1:
            pass
        elif index == 0:
            phonetic = phonetic[1:]
        elif index == len(phonetic) - 1: # last stress only happens in sentence or phrase phonetic annotation
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

    def _ipa_phonemizer_normalize(self, phonetic : str, style = 'us') -> str:
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
                    i += 1 
            else:
                if i + 2 <= len(phonetic) and phonetic[i:i+2] in self.ipa_to_cmu_wiki.keys():
                    p = self.ipa_to_cmu_wiki[phonetic[i:i+2]]
                    if p in self.cmu_vowel_phones:
                        if p != 'ER0':
                            phones.append(p+'0')
                        else:
                            phones.append(p)
                    else:
                        phones.append(p)
                
                    i += 2
                else:
                    p = self.ipa_to_cmu_wiki.get(phonetic[i], None)
                    if not p:
                        pass
                    elif p == 'AH0':
                        phones.append(p)
                    elif p in self.cmu_vowel_phones:
                        phones.append(p+'0')
                    else:
                        phones.append(p)
                    i += 1

        return phones

    # TODO: not verified, not called in normal API call
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
        
        if to_phones and style == 'br':
            warnings.warn("Britain phonemizer does not support to_phones option.")
            style = 'us'
        
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
            return self._ipa_phonemizer_normalize(phonetic, style)

    def phonemizer_phrase_sentence(self, text, style='us') -> str:
        phonetic = self.phonemizer(text, style, False)
        
        parts = phonetic.split(' ')
        phonetic = ' '.join([self._stress_normalize(p) for p in parts])

        return phonetic

    def cmu_dict(self, word, to_ipa = False) -> str:
        self.load_cmudict()
        phones = self.cmudict_plain.get(word.lower(), None)

        if not phones:
            return None
        
        if not to_ipa:
            return " ".join(phones)
        else:
            phones = self._phones39_to_ipa(phones, True)
            return "".join(phones)   

    def g2p(self, word, to_ipa = False) -> str:
        phones = self.g2p_backend(word)
        
        if not to_ipa:
            return " ".join(phones)
        else:
            phones = self._phones39_to_ipa(phones, True)
            return "".join(phones)

    def g2p_sentence(self, text, to_ipa = False) -> str:        
        return self.g2p(text, to_ipa)

    def api_word_phonetic(self, word) -> str:
        phonetic_us = self.phonemizer(word, 'us')
        phonetic_br = self.phonemizer(word, 'br')
        ret = '英: /' + phonetic_br + '/ ' + '美: /' + phonetic_us + '/ '
        return ret

    def api_phrase_sentence_phonetic(self, text) -> str:
        phonetic_us = self.phonemizer_phrase_sentence(text, 'us')
        phonetic_br = self.phonemizer_phrase_sentence(text, 'br')

        ret = '英: /' + phonetic_br + '/ ' + '美: /' + phonetic_us + '/ '
        return ret

    def api_word_phones_cmu(self, word) -> str:
        # TODO: Won't be calling all 3 calls in the future, and there is test to all APIs to all words
        p1 = self.cmu_dict(word)
        p2 = self.g2p(word)
        p3 = self.phonemizer(word, 'us', True)

        if p1 and p1 != p2:
            warnings.warn(f"CMU Dict and G2P phonetic not match for word {word}.")
            print(p1)
            print(p2)
        
        if p2 != p3:
            warnings.warn(f"G2P and Phonemizer phonetic not match for word {word}.")
            print(p2)
            print(p3)
        
        # TODO: return p2 or p3, return p3 after the dataset is updated according to phonemizer 
        return p3

    def api_phrase_sentence_phones_cmu(self, text) -> str:
        p2 = self.g2p_sentence(text)
        return p2

    def api_word_phrase_tts(self, text, accent = 'Default', speed = None) -> str:
        assert accent in ['Default', 'US', 'BR', 'AU', 'IN']
        if not speed:
            speed = self.speed

        model = self.tts_model
        speaker_ids = model.hps.data.spk2id
        # print(speaker_ids)
        # exit()
        
        text = text.strip()
        
        parts = text.split(' ')
        naming = ''
        
        if len(parts) == 1:
            naming = parts[0].lower()
        else:
            naming = '_'.join([p.lower() for p in parts])

        output_path = os.path.join(self.wav_dir, f'{naming.lower()}.wav')
        model.tts_to_file(text, speaker_ids[f'EN-{accent}'], output_path, speed = speed)

        return output_path
    
    def api_sentence_tts(self, text, accent = 'Default', speed = None) -> str:
        assert accent in ['Default', 'US', 'BR', 'AU', 'IN']
        if not speed:
            speed = self.speed + 0.1

        model = self.tts_model
        speaker_ids = model.hps.data.spk2id

        text = text.strip()
        
        parts = text.split(' ')
        naming = ''
        if len(parts) == 1:
            naming = parts[0].lower()    
        elif len(parts) <= 3:
            naming = '_'.join([p.lower() for p in parts])
        else:
            naming = '_'.join([p.lower() for p in parts[:3]])

        naming = 'sentence_' + naming + '_' + str(len(parts))
        output_path = os.path.join(self.wav_dir, f'{naming.lower()}.wav')
        model.tts_to_file(text, speaker_ids[f'EN-{accent}'], output_path, speed = speed)

        return output_path

    def api_word_translation(self, word) -> str:
        translation = self.dc_dict_word_translation(word)
        if not translation:
            # use translation model
            pass
        return translation
    
    def api_phrase_translation(self, phrase) -> str:
        translation = self.dc_dict_phrase_translation(phrase)
        if not translation:
            # use translation model
            pass
        return translation
    
    def api_sentence_translation(self, sentence) -> str:
        # use translation model
        return 
 
def main():
    t0 = time.time()
    system = platform.system()
    if system == 'Darwin':
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

    words6 = ['cat', 'cats', 'kit', 'kits', 'ate', 'eat']
    words7 = ['talk', 'talks', 'talked', 'talking']
    words8 = ['good', 'goods', 'better', 'best']
    words9 = ['in', 'outstanding', 'odd', 'odds']
    words10 = ['two']
    words11 = ['Summer', 'hurt', 'skirt', 'giver', 'Toronto', 'tomorrow', 'chat', 'mark', 
               'hot', 'astronaut', 'caught', 'not', 'australia', 'obstacle', 'montage']
    words12 = ['garage', 'tomato']
    words13 = ['tjjdjfporflker','epple', 'Kung']

    words = words0 + words1 + words2 + words3 + words4 + words5
    words = words + words4 + words6 + words7 + words8 + words9 + words10 + words11 + words12 + words13
    words = words + ['class','caught', 'hurt', 'heart']
    phrases = ["makes up", "pass for", "about time", "get away with", 'long time no see', 'take chance', 'shake it off']
    
    texts = [
        "I refuse to collect the refuse around here.", # homograph
        "I'm an activationist.",
        "The apple of my eye at 7 am run off.",
    ] 

    texts = [
        "I catched the suspect in a supermarket.",
        "He isn't reasonable enough to suspect anyone of such a crime.",
    ]

    words = ['vocabulary', 'gather', 'about', 'through', 'rough', 'content', 'magazine', 'accept', 'talked', 'bananas',
             'wishes', 'OPPO', 'suburban', 'outstanding', 'geology', 'dashing', 'longtimenosee', 'phoneme', 'thorough', 'Toronto']

    start = time.time()
    print(start - t0)
    for word in words:
        
        # s1 = phonetic.dc_dict_phonetic(word)
        # s2 = phonetic.ipa_dict(word)
        # s4_1 = phonetic.phonemizer(word, 'us')
        # s4_2 = phonetic.phonemizer(word, 'br')
        # s3 = phonetic.cmu_dict(word)x
        # s5 = phonetic.g2p(word)
        
        syllables = phonetic.api_word_phonetic(word)
        phones = phonetic.api_word_phones_cmu(word)
        text = phonetic.api_word_translation(word)
        # phonetic.api_word_phrase_tts(word, accent='Default', speed=0.7)
        
        # print(word, s1, s2, s3, s4_2, s4_1, s5)
        # print(word, s4_2, s4_1)
        print(word, syllables)
        print(phones)
        # print(s3)
        # print(s5)
        # print(s4)
        print(text)
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
    for sentence in texts:
        # print(sentence)
        phonetic.api_sentence_tts(sentence)
        syllables = phonetic.api_phrase_sentence_phonetic(sentence)
        phones = phonetic.api_phrase_sentence_phones_cmu(sentence)
        # print(phonetic.phonemizer_sentence(sentence, True, False))
        # print(phonetic.g2p(sentence, False))
        print(text)
        print(syllables)
        print(phones)
        print("")
        # print(phonetic.g2p(sentence, False))
    # exit()
    for phrase in phrases:
        # s1 = phonetic.phonemizer_phrase_sentence(phrase, 'us')
        # s2 = phonetic.phonemizer_phrase_sentence(phrase, 'br')
        syllables = phonetic.api_phrase_sentence_phonetic(phrase)
        phones = phonetic.api_phrase_sentence_phones_cmu(phrase)
        text = phonetic.api_phrase_translation(phrase)
        phonetic.api_word_phrase_tts(phrase, 'BR', speed=1.0)
        # print(phrase, s1, s2)
        print(phrase, syllables)
        print(phones)
        print(text)
        print()
    # exit()

    
    print(time.time() - start)
    return 0

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
