import time
import platform
import os
import sys
import yaml

from scoring.pronunciation_assessment import PronunciationAssessment
from dict.phonetic_dict import Phonetic

def main():
    t0 = time.time()
    system = platform.system()
    if system == 'Darwin':
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
    
    try:
        conf = yaml.safe_load(open('conf.yaml','r'))
        engine_id = conf['id']
        engine_key = conf['key']
    except:
        print("Config file not exist!")
        engine_id = None
        engine_key = None    

    phonetic = Phonetic(engine_id=engine_id, engine_key=engine_key)
    assessment = PronunciationAssessment()

    txts = ["vocabulary", "magnets can be found on a can opener."]
    wav_files = [
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/vocabulary/default_0.7/1.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/sentence/single/my.wav'
    ]
    mixed = dict(zip(wav_files, txts))
    
    
    word_cnt = 0
    character_cnt = 0
    start = time.time()
    print(start - t0)
    for wav, txt in mixed.items():
        json_file = wav.replace('.wav', '.json')
        if os.path.exists(json_file):
            os.remove(json_file)
        print(phonetic.api_all_in_one(txt, to_json=True, json_file=json_file))
        print(assessment.assess_text(txt, wav, to_json=True, json_file=json_file))

        word_cnt += len(txt.split(' '))
        character_cnt += len(txt)

    print(word_cnt, character_cnt)
    print(time.time() - start)

    return 0

if __name__ == '__main__':
    sys.exit(main())
