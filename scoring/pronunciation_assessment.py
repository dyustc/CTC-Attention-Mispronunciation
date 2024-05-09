import os
import sys
import subprocess
import json

class PronunciationAssessment:
    def __init__(self):
        # Initialize any necessary variables or resources
        self.gop_dir = os.path.normpath(os.path.dirname(__file__))
        self.gop_path = './rt_gop'
    
    def assess_text(self, text, wav_file, to_json = False, json_file = None):
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
            ret = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, cwd = self.gop_dir)
        except subprocess.CalledProcessError as e:
            err_msg = e.output
            ret = b''

        endpoint_dict = dict()
        final_dict = dict()
        # TODO: warning search
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

        if ret_dict['total_accuracy'] and ret_dict['total_fluency'] and ret_dict['total_integrity']:
            ret_dict['total_score'] = 0.7 * ret_dict['total_accuracy'] + 0.3 * ret_dict['total_fluency']
        else:
            ret_dict['total_score'] = None
          
        ret_dict['ref_content'] = input_text
        ret_dict['raw_content'] = text
        ret_dict['wav_file'] = wav_file
        ret_dict['err_msg'] = err_msg

        if to_json and json_file:
            with open(json_file, 'a') as f:
                json.dump(ret_dict, f, indent=2, ensure_ascii=False)

        return ret_dict
