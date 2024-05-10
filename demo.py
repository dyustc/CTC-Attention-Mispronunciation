import time
import platform
import os
import sys
import yaml
import soundfile as sf

from scoring.pronunciation_assessment import PronunciationAssessment
from dict.phonetic_dict import Phonetic

def main():
    start = time.time()
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

    txts = [
        "vocabulary", 
        "magnets can be found on a can opener.", 
        'get away with', 
        'on the clock',
        "magnets can't be found on a can opener.",
        'dashing'
        ]
    wav_files = [
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/vocabulary/default_0.7/1.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/sentence/single/my.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/phrase/default_0.7/get_away_with.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/phrase/default_0.7/on_the_clock.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/sentence/single/男1a.wav',
        '/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/vocabulary/default_0.7/16.wav'
    ]
    mixed = dict(zip(wav_files, txts))
    
    
    word_cnt = 0
    character_cnt = 0
    wav_length = 0
    for wav, txt in mixed.items():
        word_cnt += len(txt.split(' '))
        character_cnt += len(txt)
        data, sr = sf.read(wav)
        wav_length += len(data) / sr

    t0 = time.time()
    for wav, txt in mixed.items():
        json_file = wav.replace('.wav', '.json')
        if os.path.exists(json_file):
            os.remove(json_file)
        print(phonetic.api_all_in_one(txt, to_json=True, json_file=json_file))
    
    t1 = time.time()

    for wav, txt in mixed.items():
        json_file = wav.replace('.wav', '.json')
        print(assessment.assess_text(txt, wav, to_json=True, json_file=json_file))

    t2 = time.time()
    print('word number: ',  word_cnt)
    print('character number: ', character_cnt)
    print('wav length: ', wav_length)
    print('init time:', t0 - start)
    print('dict time:', t1 - t0)
    print('assessment time:', t2 - t1)

    return 0

if __name__ == '__main__':
    sys.exit(main())
