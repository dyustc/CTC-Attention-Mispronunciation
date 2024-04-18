import glob
import os
import string
import textgrid
import re
import argparse
import sys
import subprocess
import shutil
import platform

import time
import torch
import yaml
import argparse
import torch.nn as nn
import re
from g2p_en import G2p
import soundfile as sf
from termcolor import colored, cprint
import librosa
import warnings
import json
import logging

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config
from dict.phonetic_dict import Phonetic

parser = argparse.ArgumentParser(description="infer with only wav and transcript")

parser.add_argument('--conf', 
                    default="conf/ctc_config.0329.yaml", 
                    help='configure file for train and infer')

parser.add_argument("--wav_transcript_path",
                    required=True,
                    help="path of input wav files")

parser.add_argument("--output_dir",
                    dest="output_dir",
                    required=True,
                    help="path of output log files")

# could be from other sources
parser.add_argument("-p", "--phonetic", 
                    dest="phonetic", 
                    default="phonemizer", 
                    choices=['g2p', 'phonemizer', 'transcript'], 
                    help="the phonetics generated should come one of the above list, default phonemizer"
                    )

parser.add_argument("-f", "--format",
                    dest="phonetic_format", 
                    default="ipa",
                    choices=['cmu', 'ipa'],
                    help="phonetic format, default ipa"
                    )

parser.add_argument("--textgrid_path", 
                    help="input textgrid path, needed if phonetic value is 'transcript'")

args = parser.parse_args()

def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1,len(phn_lst)):
        if(phn_lst[i] == phn_lst[i-1] and phn_lst[i]=="sil"):
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp

g_pairs = {
    # according to english/american phonetic symbol annotation
    '0' : {
        'aa r' : 'aa',
        'ae' : 'aa',
        'ah0' : 'er0',
        'er0' : ['ah0', 'ah0 r'],
        'ao' : 'aa',
        'aa' : 'ao',
        'er r' : 'er',
        'ih r' : 'ih ah0',
        'eh r' : 'eh ah0',
        'uh r' : 'uh ah0',
    },
    
    # according to long/short vowel
    '1' : {
        'ih' : 'iy',
        'iy' : 'ih',
        'uh' : 'uw',
        'uw' : 'uh',
        'ah0' : 'er',
        'er' : 'ah0'
    },

    # according to actual false postive cases
    '2' : {
        'er0' : 'er0 r',
    },

    # according to american accent
    '3' : {
        'uh' : 'ah0',
        'ih' : 'ah0',
        'ae' : 'ah0',
        'ah' : 'ah0',
    },

    # phones can be easily mixed up
    '4' : {
        'v'  : 'w',
        'w'  : 'v',
        'ng' : 'n',
        'n'  : 'ng',
        'z'  : 'dh',
        'th' : 's',
    },

    '5' : {
        'ae' : 'eh',
        'eh' : 'ae',
        'ih' : 'iy',
        'iy' : 'ih',
        'er' : ['ah', 'aa'],
        'v'  : 'w',
        'w'  : 'v',
        'ay' : 'ae',
        'uw' : ['ah', 'uh'],
        'aa' : ['ah', 'ao'],
        'ah' : ['ao', 'ow'],
        'th' : 's',
        'ng' : 'n',
        'dh' : ['z', 'd'],
        'aw' : 'ah',
        'ey' : 'eh',
        'uh' : 'ow',
        'ao' : 'aa',
        'm'  : 'n',
        'n'  : 'm',
        'n'  : 'ng',
        'z'  : 'dh',
    },

    '6' : {
        'z'  : 's',
    }
}

def ipa_pair_rule_generate(mapping, level = 0):
    cmu_pairs = dict()
    for i in range(level+1):
        for k, v in g_pairs[str(i)].items():
            if k not in cmu_pairs:
                cmu_pairs[k] = v
            else:
                if isinstance(cmu_pairs[k], list):
                    if isinstance(v, list):
                        for e in v:
                            if e not in cmu_pairs[k]:
                                cmu_pairs[k].append(e)
                    else:
                        if v not in cmu_pairs[k]:
                            cmu_pairs[k].append(v)
                else:
                    if isinstance(v, list):
                        cmu_pairs[k] = [cmu_pairs[k]]
                        for e in v:
                            if e not in cmu_pairs[k]:
                                cmu_pairs[k].append(e)
                    else:
                        cmu_pairs[k] = [cmu_pairs[k]]
                        if v not in cmu_pairs[k]:
                            cmu_pairs[k].append(v)

    # print(cmu_pairs)
    ipa_pairs = dict()
    for k in cmu_pairs.keys():
        if k.upper() in mapping:
            new_g = mapping[k.upper()]
        else:
            parts = k.split(' ')
            new_g = ' '.join([mapping[p.upper()] for p in parts])
        
        if isinstance(cmu_pairs[k], list):
            ipa_pairs[new_g] = list()
            for e in cmu_pairs[k]:
                if e.upper() in mapping:
                    ipa_pairs[new_g].append(mapping[e.upper()])
                else:
                    parts = e.split(' ')
                    ipa_pairs[new_g].append(' '.join([mapping[p.upper()] for p in parts]))
        else:
            if cmu_pairs[k].upper() in mapping:
                ipa_pairs[new_g] = mapping[cmu_pairs[k].upper()]
            else:
                parts = cmu_pairs[k].split(' ')
                ipa_pairs[new_g] = ' '.join([mapping[p.upper()] for p in parts])
    # print(ipa_pairs)
    return ipa_pairs

def post_revise_to_rule(cannoicals, decodes, dc, rules):
    assert len(cannoicals) == len(decodes) == len(dc)

    i = 0
    altered = False
    while i < len(dc):
        if i + 2 <= len(dc):
            if dc[i] == dc[i+1] == '-':
                pass
            else:
                can_tmp = " ".join(cannoicals[i:i+2])
                dec_tmp = " ".join(decodes[i:i+2])
                if 'I' not in can_tmp:
                    if can_tmp not in rules:
                        pass
                    else:
                        alternatives = rules[can_tmp]
                        if dec_tmp in alternatives:
                            dc[i] = '-'
                            dc[i+1] = '-'
                            decodes[i] = cannoicals[i]
                            decodes[i+1] = cannoicals[i+1]
                            altered = True
                            i += 2
                            continue
                        else:
                            if decodes[i] == 'D':
                                left = decodes[i+1]
                            elif decodes[i+1] == 'D':
                                left = decodes[i]
                            else:
                                left = None
                            
                            if left and left in alternatives:
                                dc[i] = '-'
                                dc[i+1] = '-'
                                decodes[i] = cannoicals[i]
                                decodes[i+1] = cannoicals[i+1]
                                altered = True
                                i += 2
                                continue
                elif can_tmp == "I I":
                    pass
                else:
                    if cannoicals[i] == 'I':
                        left = cannoicals[i+1]
                    else:
                        left = cannoicals[i]
                    
                    if left not in rules:
                        pass
                    else:
                        alternatives = rules[left]
                        if dec_tmp in alternatives:
                            dc[i] = '-'
                            decodes[i] = left
                            cannoicals[i] = left

                            dc.pop(i+1)
                            decodes.pop(i+1)
                            cannoicals.pop(i+1)

                            altered = True
                            i += 1
                            continue
        
        if dc[i] in ['-', 'D', 'I']:
            pass
        else:
            if cannoicals[i] not in rules:
                pass
            else:
                alternatives = rules[cannoicals[i]]
                if decodes[i] in alternatives:
                    dc[i] = '-'
                    decodes[i] = cannoicals[i]
                    altered = True

        i += 1
    
    assert len(cannoicals) == len(decodes) == len(dc)

    return cannoicals, decodes, dc, altered

def post_del_repeat(decodes):
    parts = decodes.split(' ')
    ans = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i] == parts[i+1]:
            i += 1
            continue
        else:
            ans.append(parts[i])
        i += 1
    
    return " ".join(ans)

def print_aligned_string(s1, s2, l):
    s1 = [p+" " if len(p) == 1 else p for p in s1]
    s2 = [p+" " if len(p) == 1 else p for p in s2]
    l = [s+" " for s in l]

    return " ".join(s1), " ".join(s2), " ".join(l)

def align_canonical_decoded(s1, s2, l):
    d = {} # log for each phoneme in canonicals
    for i in range(len(s2)):
        d[i] = ""
    d['I'] = []
    
    i, j = 0, 0
    while i < len(l):
        if l[i] == '-' or l[i] == 'S':
            d[j] = l[i]
            if l[i] == 'S':
                d[j] += s1[i]
            i += 1
            j += 1
            continue

        if l[i] == 'D':
            d[j] = 'D'
            j += 1
            s1.insert(i, 'D')
        else:
            d['I'] += [i]
            s2.insert(i, 'I')
        i += 1
    
    # TODO: remove insertion at the beginning, this is the model inference optimization
    if d['I']:
        i = 0
        while i == d['I'][i]:
            i += 1
            if i == len(d['I']):
                break
        
        if i == 0:
            pass
        else:
            s1 = s1[i-1:]
            s2 = s2[i-1:]
            l = l[i-1:]
            d['I'] = d['I'][i-1:]

    # TODO: remove the repeating phoneme at start, this is the model inference optimization
    if l[0] == s2[0] == 'I' and len(s1) >= 2 and s1[0] == s1[1]:
        l = l[1:]
        s2 = s2[1:]
        s1 = s1[1:]
    
    return s1, s2, l

def infer_init():
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)    
    
    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)

    use_cuda = opts.use_gpu and torch.cuda.is_available()
    # use_cuda = False
    print('mdd use cuda: ', use_cuda)
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    
    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    package = torch.load(model_path, map_location=device)
    
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    drop_out = package['_drop_out']

    beam_width = opts.beam_width
    decoder_type =  opts.decode_type
    
    vocab_file = opts.vocab_file
    vocab = Vocab(vocab_file)
    # ipa verify
    # for p in vocab.word2index:
    #     if p not in ipa_symbols.keys():
    #         print(p, p)
    # print('-------')
    # for p in vocab.word2index:
    #     if p in ipa_symbols.keys():
    #         print(p, ipa_symbols[p])


    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha) 

    return opts, device, model, decoder, vocab

def infer_data_init(opts, vocab):
    test_wrd_file = args.wav_transcript_path + "/wrd.txt"
    test_transcipt_dict = {}
    with open(test_wrd_file) as f:
        for l in f.readlines():
            # TODO: only support one-line and no repeating words in a sequence
            l = l.strip("\n")
            i = l.find(' ')
            test_transcipt_dict[l[:i]] = l[i+1:]    
    
    test_scp_path = args.wav_transcript_path + "/fbank.scp"
    test_trans_path = args.wav_transcript_path + "/transcript_phn.txt"
    # print(vocab.word2index)
    # exit()
    # Note: no label dataset format, so use test_trans_path as label, won't be used anyway
    test_dataset = SpeechDataset(vocab, test_scp_path, test_trans_path, test_trans_path, opts)

    # batch_size = opts.batch_size
    batch_size = 1
    test_loader = SpeechDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    
    return test_loader, test_transcipt_dict
    
def infer(phonetic, word_dict, test_loader, device, model, decoder, vocab, test_transcipt_dict, use_ipa):    
    total_correct_cnt = 0
    total_cnt = 0
    total_insertion_cnt = 0
    w1 = open(args.wav_transcript_path + "/decode_seq.txt",'w+')
    ipa_mapping = ipa_pair_rule_generate(phonetic.cmu_to_ipa_wiki, level=3)
    
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, _, _, trans, trans_sizes, utt_list = data
            inputs = inputs.to(device)
            trans = trans.to(device)
            
            # TODO: padding to remove much insertion err
            p1d = (0, 1)
            trans = torch.nn.functional.pad(trans, p1d, "constant", 2)
            # p1d = (0, 1)
            # trans = torch.nn.functional.pad(trans, p1d, "constant", 0)
            
            probs = model(inputs,trans)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
        
            trans, trans_sizes = trans.cpu().numpy(), trans_sizes.numpy()

            canonicals = []
            for i in range(len(trans)):
                canonical = [vocab.index2word[num] for num in trans[i][:trans_sizes[i]]]
                canonicals.append(' '.join(canonical))
                
            ## compute with out sil     
            decoded_nosil = []
            canonicals_nosil = []
            for i in range(len(canonicals)):
                hyp = decoded[i].split(" ")
                c_ref = canonicals[i].split(" ")

                hyp_precess = [ i   for i in hyp if(i != "sil")  ]
                c_ref_precess = [ i   for i in c_ref if(i != "sil")]

                decoded_nosil.append(' '.join(hyp_precess))
                canonicals_nosil.append(' '.join(c_ref_precess))

            for x in range(len(decoded_nosil)):
                utterance = test_transcipt_dict[utt_list[x]]
                decoded_nosil[x] = decoded_nosil[x].replace('err', '')
                decoded_nosil[x] = decoded_nosil[x].replace('  ', ' ')
                # TODO:
                decoded_nosil[x] = post_del_repeat(decoded_nosil[x])
                
                if not decoded_nosil[x]:
                    # warnings.warn("Empty decoded sequence, SKIP FOR: " + utterance)
                    logging.info("Empty decoded sequence, SKIP FOR: " + utterance)
                    phones_canonicals = [c for c in canonicals_nosil[x].split(' ') if c]
                    if use_ipa:
                        phones_canonicals = [phonetic.cmu_to_ipa_wiki.get(c.upper(), c) for c in phones_canonicals]
                    phones_decoded = ['D'] * len(phones_canonicals)
                    dc_path = ['D'] * len(phones_canonicals)
                else:
                    _, dc_path = decoder.wer(decoded_nosil[x], canonicals_nosil[x])

                    phones_decoded = [c for c in decoded_nosil[x].split(' ') if c]
                    phones_canonicals = [c for c in canonicals_nosil[x].split(' ') if c]
                
                    if use_ipa:
                        phones_decoded = [phonetic.cmu_to_ipa_wiki.get(c.upper(), c) for c in phones_decoded]
                        phones_canonicals = [phonetic.cmu_to_ipa_wiki.get(c.upper(), c) for c in phones_canonicals]

                    phones_decoded, phones_canonicals, dc_path = align_canonical_decoded(phones_decoded, phones_canonicals, dc_path)
                    phones_canonicals, phones_decoded, dc_path, altered = post_revise_to_rule(phones_canonicals, phones_decoded, dc_path, ipa_mapping)
                    if altered:
                        # warnings.warn("Some phonemes are revised according to the rules, for this utterance: " + utterance)
                        logging.warning("Some phonemes are revised according to the rules. FOR UTTERANCE: " + utterance)
                        

                tmp1, tmp2, tmp3 = print_aligned_string(phones_decoded, phones_canonicals, dc_path) 
                
                del_sub_cnt = sum([1 if c == 'D' or c == 'S' else 0 for c in dc_path])
                correct_cnt = sum([1 if c == '-' else 0 for c in dc_path])
                insertion_fault, substution_fault, deletion_fault = stastics(dc_path, phones_canonicals, phones_decoded)
                tmp_score = min(len(insertion_fault) / 4, 0.1 * (correct_cnt + del_sub_cnt))
                score = math.ceil((1 - (del_sub_cnt + tmp_score)/(del_sub_cnt + correct_cnt)) * 100)

                # if utt_list[x] == '18':
                #     print(dc_path)
                #     print(phones_canonicals)
                #     print(phones_decoded)
                #     print(insertion_fault)
                #     print(substution_fault)
                #     print(deletion_fault)
                annotation = word_dict[utt_list[x]]['ipa']
                translation = phonetic.api_word_translation(utterance)
                print("id     : " + utt_list[x])
                print(utt_list[x] + ": " + utterance)
                print(annotation)
                print(translation)
                # phonetic.api_word_phrase_tts(utterance, accent='Default', speed=0.7)
                print(tmp2) 
                print(tmp3)
                print(tmp1)
                print("ins err: " + " ".join(insertion_fault))
                print("sub err: " + " ".join(substution_fault))
                print("del err: " + " ".join(deletion_fault))
                print('Comp.  : ' + str(correct_cnt) + '/'+ str(correct_cnt + del_sub_cnt))
                print('score  : ' + str(score))
                print("")
                
                with open(args.output_dir + "/" + utt_list[x] + '.json','a') as w2:
                    result = {
                        'id' : utt_list[x],
                        'utterance' : utterance,
                        'annotation' : annotation,
                        'translation' : translation,
                        'alignment_1' : tmp2,
                        'alignment_2' : tmp3,
                        'alignment_3' : tmp1,
                        'insertion_fault' : " ".join(insertion_fault),
                        'substution_fault' : " ".join(substution_fault),
                        'deletion_fault' : " ".join(deletion_fault),
                        'completeness' : str(correct_cnt) + '/'+ str(correct_cnt + del_sub_cnt),
                        'score' : str(score)
                    }
                    json.dump(result, w2, indent=2, ensure_ascii=False)
                
                total_correct_cnt += correct_cnt
                total_cnt += correct_cnt + del_sub_cnt
                total_insertion_cnt += len(insertion_fault)
                w1.write(utt_list[x] + " " + " ".join(phones_decoded) + "\n")    
    w1.close()
    return total_correct_cnt, total_cnt, total_insertion_cnt

def read_phonemes_from_transcript(can_phn_path):
    can_transcript_phns = []
    can_tg = textgrid.TextGrid.fromFile(can_phn_path)

    for i in can_tg[1]:
        if i.mark == "" or i.mark == None:
            can_transcript_phns.append("sil")
            
            trans_phn = i.mark
            trans_phn = trans_phn.strip(" ")
            trans_phn = trans_phn.rstrip(string.digits)

            ## trans phn 
            if(trans_phn == "sp" or trans_phn == "SIL" or trans_phn == "" or trans_phn == "spn" ):
                can_transcript_phns.append("sil")
            else:
                if(trans_phn == "ERR" or trans_phn == "err"):
                    can_transcript_phns.append("err")
                elif(trans_phn == "ER)"):
                    can_transcript_phns.append("er")
                elif(trans_phn == "AX" or trans_phn == "ax" or trans_phn == "AH)"):
                    can_transcript_phns.append("ah")
                elif(trans_phn == "V``"):
                    can_transcript_phns.append("v")
                elif(trans_phn == "W`"):
                    can_transcript_phns.append("w")    
                else:
                    can_transcript_phns.append(trans_phn.lower())
    
    return can_transcript_phns

def stastics(dc_path, phones_canonicals, phones_decoded):
    insertion_fault = []
    substution_fault = []
    deletion_fault = []
    
    i, j, k = 0, 0, 0
    while i < len(dc_path):
        if dc_path[i] == '-':
            i += 1
            j += 1
            k += 1
            continue
        elif dc_path[i] == 'S':
            substution_fault.append(phones_canonicals[j])
            i += 1
            j += 1
            k += 1
        elif dc_path[i] == 'I':
            insertion_fault.append(phones_decoded[k])
            i += 1
            j += 1
            k += 1
        else:
            deletion_fault.append(phones_canonicals[j])
            i += 1
            j += 1
            k += 1
    
    return insertion_fault, substution_fault, deletion_fault          

def main():
    system = platform.system()
    subfolder = ''
    
    if system == 'Darwin':
        subfolder = 'mac'
    else:
        subfolder = 'linux'
    
    t0 = time.time()
    tmp_path = args.wav_transcript_path
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = glob.glob(output_dir + '/*')
        for f in files:
            os.remove(f)
    
    # dict_log  = output_dir + "/dict.log"
    # with open(dict_log, 'w+') as f:
    #     pass
    
    rule_log = output_dir + "/rule.log"
    with open(rule_log, 'w+') as f:
        pass
    logging.basicConfig(filename=rule_log, level=logging.DEBUG, format='')

    use_ipa = args.phonetic_format == 'ipa'
    phoneme_frd = args.phonetic
    
    w = open(tmp_path+"/wrd.txt",'w+')
    w1 = open(tmp_path+"/wav.scp",'w+')
    w4 = open(tmp_path+"/transcript_phn.txt",'w+')
    
    if args.phonetic == 'transcript':
        if not os.path.exists(args.textgrid_path):
            print(args.textgrid_path + ' , not a valid textGrid source')
    else:
        args.textgrid_path = None # bypass

    print(args.wav_transcript_path, use_ipa, phoneme_frd, rule_log, output_dir)

    total_wav_time = 0
    cnt = 0
    t1 = time.time()
    if system == 'Darwin':
        try:
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
            phonetic = Phonetic()
        except:
            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = '/usr/local/Cellar/espeak/1.48.04_1/lib/libespeak.dylib'
            phonetic = Phonetic()
    else:
        phonetic = Phonetic()

    can_transcript_words_dict = dict()

    opts, device, model, decoder, vocab = infer_init()

    silence_wav_path = './silence.wav'
    denoised_dir = os.path.normpath('/'.join([args.wav_transcript_path, 'denoised']))
    os.makedirs(denoised_dir, exist_ok=True)
    far_data, fs = sf.read(silence_wav_path)
    data_limit = len(far_data)
    
    # denoise process
    for p in os.listdir(args.wav_transcript_path):
        if os.path.isdir('/'.join([args.wav_transcript_path, p])):
            continue

        ext = p.split('.')[1]
        utt_id = p.split('.')[0]
        if ext != 'wav':
            continue
        
        cnt += 1
        wav_path = os.path.normpath('/'.join([args.wav_transcript_path, p]))
        data, fs = sf.read(wav_path)
        if len(data.shape) != 1:
            data = data[:, 0]

        if fs != 16000:
            data = librosa.resample(data, orig_sr=fs, target_sr=16000)
            sf.write(wav_path, data, 16000)

        denoised_wav_path = os.path.normpath('/'.join([denoised_dir, p])) 
        cmd1 = ' '.join(['./bin/{}/eeo_apm_test'.format(subfolder), wav_path, silence_wav_path, denoised_wav_path, '4', '0'])
        subprocess.check_output(cmd1, shell=True, stderr=subprocess.STDOUT)

        data, fs = sf.read(denoised_wav_path)
        if len(data) > data_limit:
            print('{} skipped, currently wav length should be no more than 3 minutes!'.format(wav_path))
            continue

        total_wav_time += len(data) / fs
        w1.write(utt_id + " " + denoised_wav_path + "\n" )
    w1.close()
    
    t2 = time.time()
    for p in os.listdir(args.wav_transcript_path):
        if os.path.isdir('/'.join([args.wav_transcript_path, p])):
            continue

        ext = p.split('.')[1]
        utt_id = p.split('.')[0]
        if ext != 'wav':
            continue
            
        can_transcript_phns = []
        
        if args.phonetic == 'transcript':
            tmp1 = re.sub('wav', 'TextGrid', p)
            can_phn_path = os.path.normpath('/'.join([args.textgrid_path, tmp1]))
            if not os.path.exists(can_phn_path):
                continue

            can_transcript_phns = read_phonemes_from_transcript(can_phn_path)
        else:
            tmp2 = re.sub('wav', 'txt', p)
            text_path = os.path.normpath('/'.join([args.wav_transcript_path, tmp2]))
            if not os.path.exists(text_path):
                continue
            
            with open(text_path,'r') as f:
                for utterance in f.readlines():
                    utterance = utterance.strip()
                    if not utterance:
                        continue

                    w.write(utt_id + " " + utterance + "\n")
                    
                    is_word = True
                    if ' ' in utterance:
                        is_word = False
                    
                    if is_word:
                        can_transcript_phns_ipa = phonetic.api_word_phonetic(utterance)
                        can_transcript_phns = phonetic.api_word_phones_cmu(utterance)
                    else:
                        can_transcript_phns_ipa = phonetic.api_phrase_sentence_phonetic(utterance)
                        can_transcript_phns = phonetic.api_phrase_sentence_phones_cmu(utterance)
                    
                    parts_ = can_transcript_phns.split(' ')
                    parts_ = [p for p in parts_ if p.strip()]
                    parts_ = [p.rstrip(string.digits) if p not in ['ER0', 'AH0'] else p for p in parts_]
                    
                    parts = [p.lower() for p in parts_]
                    can_transcript_phns_for_models = ' '.join(parts)
                    
                    parts = [phonetic.cmu_to_ipa_wiki[p] for p in parts_]

                can_transcript_words_dict[utt_id] = {
                    'ipa': can_transcript_phns_ipa,
                    'cmu_phns' : can_transcript_phns,
                    'ipa_phns' : parts
                }
        
        w4.write(utt_id + " " + can_transcript_phns_for_models + "\n" )
    
    w.close()
    w4.close()
    t3 = time.time()
    cmd1 = './bin/{}/compute-fbank-feats --config=conf/fbank.conf scp,p:{}/wav.scp ark:- | '.format(subfolder, tmp_path)
    cmd2 = './bin/{}/apply-cmvn --norm-vars=true {}/global_fbank_cmvn.txt ark:- ark:- | '.format(subfolder, 'data')
    cmd3 = './bin/{}/copy-feats --compress={} ark:- ark,scp:{}/fbank.ark,{}/fbank.scp'.format(subfolder, 'false', tmp_path, tmp_path)
    cmd4 = '>/dev/null 2>&1'
    cmd = cmd1 + cmd2 + cmd3
    # print(cmd)
    # os.system(cmd)
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    
    test_loader, test_transcipt_dict = infer_data_init(opts, vocab)
    c1, c2, c3 = infer(phonetic, can_transcript_words_dict, test_loader, device, model, decoder, vocab, test_transcipt_dict, use_ipa)
    print(c1, c2, c3)
    # remove denoise dir
    shutil.rmtree(denoised_dir)
    os.remove(tmp_path+"/wrd.txt")
    os.remove(tmp_path+"/transcript_phn.txt")
    os.remove(tmp_path+"/decode_seq.txt")
    os.remove(tmp_path+"/wav.scp")
    os.remove(tmp_path+"/fbank.scp")
    os.remove(tmp_path+"/fbank.ark")

    end = time.time()
    time_used = (end - t0)
    rtf = time_used / total_wav_time
    rtf1 = (t1-t0) / total_wav_time
    rtf2 = (t2-t1) / total_wav_time
    rtf3 = (t3-t2) / total_wav_time
    rtf4 = (end-t3) / total_wav_time
    print("RTF: %.4f, time used for decode %d sentences: %.4f seconds, total wav length: %.4f seconds" % (rtf, cnt, time_used, total_wav_time))
    print("init model time: %.4f, init phone time: %.4f, denoise time: %.4f, mdd+tts infer time: %.4f" %(rtf1, rtf3, rtf2, rtf4))
    print("process time: %.4f" % (rtf2 + rtf4))
    return 0

if __name__ == '__main__':
    sys.exit(main())
