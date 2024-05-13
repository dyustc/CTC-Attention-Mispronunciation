import os
import sys
import subprocess
import json
import time
import soundfile as sf

class PronunciationAssessment:
    def __init__(self):
        # Initialize any necessary variables or resources
        self.gop_dir = os.path.normpath(os.path.dirname(__file__))
        self.gop_path = './rt_gop'
    
    # TODO: renaming and add is_word param
    def assess_text(self, text, wav_file, to_json = False, json_file = None):
        # Assess the pronunciation of a single word
        # Implement your logic here
        t0 = time.time()
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

        t1 = time.time()
        try:
            # TODO: Bugfix catch 
            ret = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, cwd = self.gop_dir)
        except subprocess.CalledProcessError as e:
            # TODO: e.output is too long for a error message
            err_msg = str(e.output)
            ret = b''
        t2 = time.time()

        endpoint_dicts = []
        final_dict = dict()
        # TODO: warning search
        parts = ret.split(b'[SEP]\n')
        for part in parts:
            dict_str = part
            try:
                d = json.loads(dict_str)
                if d.get('output_type', None) == 'endpoint':
                    endpoint_dict = d
                    endpoint_dicts.append(endpoint_dict)
                
                if d.get('output_type', None) == 'final':
                    final_dict = d
            except json.JSONDecodeError:
                continue

        if final_dict:
            ret_dict = final_dict
        else:
            err_msg = 'no final dict'
        
        sentence_content = ''
        words = []
        for endpoint_dict in endpoint_dicts:
            sentence_content += endpoint_dict['sentence_content'] + ' '
            words += endpoint_dict['words']
        
        if sentence_content:
            ret_dict['sentence_content'] = sentence_content + ret_dict['sentence_content']
        
        if words:
            ret_dict['words'] = words + ret_dict['words']

        ret_dict['total_accuracy'] = final_dict.get('sentence_accuracy', None)
        ret_dict['total_fluency'] = final_dict.get('sentence_fluency', None)
        ret_dict['total_integrity'] = final_dict.get('sentence_integrity', None)

        # TODO: fix for accuracy is on debate
        if ret_dict['total_accuracy'] and ret_dict['total_integrity']:
            ret_dict['total_accuracy'] = ret_dict['total_accuracy'] * 100 / ret_dict['total_integrity']
            ret_dict['total_accuracy'] = min(100, ret_dict['total_accuracy'])
        
        # TODO: for word, total score is total accuracy
        if ret_dict['total_accuracy'] and ret_dict['total_fluency'] and ret_dict['total_integrity']:
            ret_dict['total_score'] = 0.5 * ret_dict['total_accuracy'] + 0.3 * ret_dict['total_fluency'] + 0.2 * ret_dict['total_integrity']
        else:
            ret_dict['total_score'] = None
        
        ret_dict['endpoint_num'] = len(endpoint_dicts)
        ret_dict['ref_content'] = input_text
        ret_dict['raw_content'] = text
        ret_dict['wav_file'] = wav_file
        ret_dict['err_msg'] = err_msg
        ret_dict['time'] = round((time.time() - t0) * 1000)
        ret_dict['time_of_cmd'] = round((t2 - t1) * 1000)
        data, sr = sf.read(wav_file)
        wav_length = len(data) / sr
        ret_dict['time_of_wav'] = round(wav_length * 1000)

        if to_json and json_file:
            with open(json_file, 'a') as f:
                json.dump(ret_dict, f, indent=2, ensure_ascii=False)

        return ret_dict
