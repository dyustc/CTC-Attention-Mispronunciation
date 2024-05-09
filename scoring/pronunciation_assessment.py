import os
import sys
import subprocess
import json

class PronunciationAssessment:
    def __init__(self):
        # Initialize any necessary variables or resources
        self.gop_path = './rt_gop'
    
    def assess_text(self, text = None, wav_file = None):
        # Assess the pronunciation of a single word
        # Implement your logic here
        input_text = text.replace("\"", "'")
        cmd = ' '.join([self.gop_path, wav_file, '"'+input_text+'"'])
        ret_dict = {
            'output_type' : None,
            'sentence_accuracy' : None,
            'sentence_content' : None,
            'sentence_fluency' : None,
            'sentence_integrity' : None,
            'word_global_num' : None,
            'words' : None
        }

        err_msg = None

        try:
            # TODO: Bugfix catch 
            ret = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            err_msg = e.output
            ret = b''

        endpoint_dict = dict()
        final_dict = dict()
        
        parts = ret.split(b'[SEP]\n')
        for part in parts:
            dict_str = part
            try:
                d = json.loads(dict_str)
                if d.get('output_type', None) == 'endpoint':
                    endpoint_dict = d
                
                if d.get('output_type', None) == 'final':
                    final_dict = d
            except json.JSONDecodeError:
                continue

        if endpoint_dict:
            ret_dict = endpoint_dict
            ret_dict['output_type'] = 'final'
        else:
            if final_dict:
                ret_dict = final_dict

        ret_dict['total_accuracy'] = final_dict.get('sentence_accuracy', None)
        ret_dict['total_fluency'] = final_dict.get('sentence_fluency', None)
        ret_dict['total_integrity'] = final_dict.get('sentence_integrity', None)
          
        ret_dict['ref_content'] = input_text
        ret_dict['raw_content'] = text
        ret_dict['wav_file'] = wav_file
        ret_dict['err_msg'] = err_msg

        return ret_dict

# Usage example
assessment = PronunciationAssessment()

word = "vocabulary's"
wav_file = '/Users/daiyi/work/ramp/pronunciation/mispronunciation_diagnosis/egs/vocabulary/default_0.7/1.wav'
word_result = assessment.assess_text(word, wav_file)
# print(word_result)
# print(word_result.keys())


phrase = "\"about time"
wav_file = '/Users/daiyi/work/ramp/pronunciation/mispronunciation_diagnosis/egs/phrase/default_0.7/about_time.wav'
phrase_result = assessment.assess_text(phrase, wav_file)
print(phrase_result)
# print(phrase_result.keys())

sentence = "magnets can't be found on a can opener."
wav_file = "/Users/daiyi/work/ramp/CTC-Attention-Mispronunciation/egs/sentence/single/男1a.wav"
sentence_result = assessment.assess_text(sentence, wav_file)
# print(sentence_result)
# print(sentence_result.keys())

# TODO: Bugfix
# John'ss vocabulary's
