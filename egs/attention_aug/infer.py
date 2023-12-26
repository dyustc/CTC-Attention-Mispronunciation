import glob
import os
import string
import textgrid
import re
import argparse
import sys
import subprocess

import time
import torch
import yaml
import argparse
import torch.nn as nn
import re
from g2p_en import G2p
import soundfile as sf
from termcolor import colored, cprint

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config


parser = argparse.ArgumentParser(description="infer with only wav and transcript")
parser.add_argument('--conf', default="conf/ctc_config.yaml", help='conf file for train and infer')
parser.add_argument("--wav_transcript_path",default="/data2/daiyi/dataset/TXHC_EXTRA/wav",help="input path")

# could be from other sources
parser.add_argument("--no_g2p_en", action="store_true", help="use phoneme from g2p_en, must have a source of phones, e.g, textgrid")
parser.add_argument("--textgrid_path",default="/data2/daiyi/dataset/TXHC_EXTRA/cmu_aligned",help="input textgrid path")

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
    '0' : {
        'ah' : 'ae',
        },
    '1' : {
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
        'ey' : 'eh'
        # 'n'  : 'ng',
        # 'z'  : 'dh',
    },
    '2' : {
        'z'  : 's',
    }
}

def mild1(s1, s2, s3, level = 1):
    pairs = dict()
    for i in range(level+1):
        d = g_pairs[str(i)]
        for k in d:
            if k not in pairs:
                if type(d[k]) == list:
                    pairs[k] = d[k]
                else:
                    pairs[k] = [d[k]]
            else:
                if type(d[k]) == list:
                    pairs[k] += d[k]
                else:
                    pairs[k].append(d[k])

    l1 = s1.split(' ')
    l2 = s2.split(' ')
    l3 = s3.split(' ')

    i, j, k = 0, 0, 0
    while i < len(l1) and j < len(l2) and k < len(l3):
        while i < len(l1) and not l1[i]:
            i += 1
        while j < len(l2) and not l2[j]:
            j += 1
        while k < len(l3) and not l3[k]:
            k += 1
        
        if i < len(l1) and j < len(l2) and k < len(l3):
            if l1[i] in pairs and l2[j] == 'S' and l3[k] in pairs[l1[i]]:
                l2[j] = '-'
                if len(l3[k]) == len(l1[i]):
                    l3[k] = l1[i]
                else:
                    if len(l3[k]) < len(l1[i]):
                        if k+1 < len(l3):
                            l3[k] = l1[i]
                            l3.pop(k+1)
                    else:
                        if i+1 < len(l1):
                            l1.pop(i+1)
                            l3[k] = l1[i]
        i += 1
        j += 1
        k += 1

    s1 = ' '.join(l1)
    s2 = ' '.join(l2)
    s3 = ' '.join(l3)

    return s1, s2, s3

def mild2(s1, s2, s3):
    l1 = s1.split(' ')
    l2 = s2.split(' ')
    l3 = s3.split(' ')

    i, j, k = 0, 0, 0
    while l1[i] in ['I', ''] and l2[j] == l1[i]:
        i += 1
        j += 1
        if l1[i]:
            k += 1
    
    if k > 1:
        k1 = 0
        k2 = 0
        while k1 < k - 1:
            if l3[k2] != '':
                k1 += 1
            k2 += 1

        l1 = l1[2 * (k-1):]
        l2 = l2[2 * (k-1):]
        l3 = l3[k2:]

    l1 = l1[::-1]
    l2 = l2[::-1]
    l3 = l3[::-1]

    i, j, k = 0, 0, 0
    while l1[i] in ['I', ''] and l2[j] == l1[i]:
        i += 1
        j += 1
        if l1[i]:
            k += 1

    if k > 2:
        k1 = 0
        k2 = 0
        while k1 < k - 2:
            if l3[k2] != '':
                k1 += 1
            k2 += 1

        l1 = l1[2 * (k-2):]
        l2 = l2[2 * (k-2):]
        l3 = l3[k2:]
    
    l1 = l1[::-1]
    l2 = l2[::-1]
    l3 = l3[::-1]

    s1 = ' '.join(l1)
    s2 = ' '.join(l2)
    s3 = ' '.join(l3)
    
    return s1, s2, s3

def print_align_space_canonical_origin(s1, s2, l):
    p_s1 = s1.split(' ')
    p_s2 = s2.split(' ')

    d = {}
    for i in range(len(p_s2)):
        d[i] = ""
    d['I'] = []
    
    i = 0
    j = 0
    while i < len(l):
        if l[i] == '-' or l[i] == 'S':
            d[j] = l[i]
            if l[i] == 'S':
                d[j] += p_s1[i]
            i += 1
            j += 1
            continue

        if l[i] == 'D':
            d[j] = 'D'
            j += 1
            p_s1.insert(i, 'D')
        else:
            d['I'] += [str(j-1) + str(j)]
            p_s2.insert(i, 'I')
        i += 1
    
    p_s1 = [p+" " if len(p) == 1 else p for p in p_s1]
    p_s2 = [p+" " if len(p) == 1 else p for p in p_s2]
    l = [s+" " for s in l]

    return ' '.join(p_s2), ' '.join(p_s1), ' '.join(l), len(p_s2), d

def infer(word_dict):
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)    
    
    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)

    use_cuda = opts.use_gpu
    use_cuda = False
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

    test_wrd_file = args.wav_transcript_path + "/wrd.txt"
    test_wrd_dict = {}
    with open(test_wrd_file) as f:
        for l in f.readlines():
            # TODO: only support one-line and no repeating words in a sequence
            l = l.strip("\n")
            i = l.find(' ')
            test_wrd_dict[l[:i]] = l[i+1:]    
    
    vocab = Vocab(vocab_file)

    test_scp_path = args.wav_transcript_path + "/fbank.scp"
    test_trans_path = args.wav_transcript_path + "/transcript_phn.txt"

    # Note: no label dataset format, so use test_trans_path as label, won't be used anyway
    test_dataset = SpeechDataset(vocab, test_scp_path, test_trans_path, test_trans_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha) 
        
    w1 = open(args.wav_transcript_path + "/decode_seq.txt",'w+')
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, _, _, trans, trans_sizes, utt_list = data
            inputs = inputs.to(device)
            trans = trans.to(device)
            
            probs = model(inputs,trans)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
        
            trans, trans_sizes = trans.cpu().numpy(), trans_sizes.numpy()

            canonicals = []
            for i in range(len(trans)):
                canonical = [ vocab.index2word[num] for num in trans[i][:trans_sizes[i]]]
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
                w1.write(utt_list[x] + " " + decoded_nosil[x] + "\n")    
            
            for x in range(len(decoded_nosil)):
                utterance = test_wrd_dict[utt_list[x]]
                decoded_nosil[x] = decoded_nosil[x].replace('err', '')
                decoded_nosil[x] = decoded_nosil[x].replace('  ', ' ')
                _, dc_path = decoder.wer(decoded_nosil[x], canonicals_nosil[x])
                # print(canonicals_nosil[x], len(canonicals_nosil[x].split(' ')))
                # print(decoded_nosil[x], len(decoded_nosil[x].split(' ')))
                # complete_score1 = sum([1 if c == 'D' or c == 'S' else 0 for c in dc_path])
                tmp1, tmp2, tmp3, canonical_len2, d2 = print_align_space_canonical_origin(decoded_nosil[x], canonicals_nosil[x], dc_path)
                # print(tmp1, len([c for c in tmp1.split(' ') if c]))
                # print(tmp3, len([c for c in tmp3.split(' ') if c]))
                # print(tmp2, len([c for c in tmp2.split(' ') if c]))
                tmp1, tmp3, tmp2 = mild2(tmp1, tmp3, tmp2)
                tmp1, tmp3, tmp2 = mild1(tmp1, tmp3, tmp2, level = 1)

                c_path = [c for c in canonicals_nosil[x].split(' ') if c]
                dc_path = [c for c in tmp3.split(' ') if c]
                complete_score = sum([1 if c == 'D' or c == 'S' else 0 for c in dc_path])
                
                print(utt_list[x])
                print(len(c_path), complete_score)
                print("text      : " + utterance)
                repeatted_words_list = word_dict.get(utt_list[x], None)
                # print(repeatted_words_list)
                # print(dc_path)

                formatted_text = []
                insertion_cnt = 0
                # print(len(dc_path), repeatted_words_list[-1][-1]+1, len(canonicals_nosil[x].split(" ")))
                # TODO: Due to mismatch problem
                insertion_cnt_preview = sum([e == 'I' for e in dc_path])
                if len(dc_path)-insertion_cnt_preview < repeatted_words_list[-1][-1]+1:
                    continue

                for l in repeatted_words_list:
                    w, start, end = l
                    k = start + insertion_cnt
                    is_word_added = False
                    while k <= end + insertion_cnt:
                        if dc_path[k] == 'S' or dc_path[k] == 'D':
                            if not is_word_added:
                                formatted_text.append(colored(w, 'red', attrs=['bold']))
                                is_word_added = True
                            k += 1
                        elif dc_path[k] == '-':
                            k += 1
                        else:
                            if not is_word_added:
                                formatted_text.append(colored(w, 'red', attrs=['bold']))
                                is_word_added = True
                            k += 1
                            insertion_cnt += 1
                    if not is_word_added:
                        formatted_text.append(colored(w, 'white', attrs=['bold']))
                
                cprint("c-text    : " + " ".join(formatted_text))    
                print("canonical : " + tmp1) 
                print("            " + tmp3)
                print("decode    : " + tmp2)
                print("\n")

    w1.close()

def main():
    start = time.time()
    tmp_path = args.wav_transcript_path
    w = open(tmp_path+"/wrd.txt",'w+')
    w1 = open(tmp_path+"/wav.scp",'w+')
    w4 = open(tmp_path+"/transcript_phn.txt",'w+')
    print(args.wav_transcript_path)
    if args.no_g2p_en:
        if not os.path.exists(args.textgrid_path):
            print(args.textgrid_path + ' , not a valid textGrid source')
    else:
        args.textgrid_path = None # bypass

    total_wav_time = 0
    cnt = 0
    g2p = G2p()
    can_transcript_words_dict = dict()
    for p in os.listdir(args.wav_transcript_path):
        ext = p.split('.')[1]
        utt_id = p.split('.')[0]
        if ext != 'wav':
            continue
        
        cnt += 1
        wav_path = os.path.normpath('/'.join([args.wav_transcript_path, p]))
        data, fs = sf.read(wav_path)
        total_wav_time += len(data) / fs

        tmp2 = re.sub('wav', 'txt', p)
        text_path = os.path.normpath('/'.join([args.wav_transcript_path, tmp2]))

        can_transcript_phns = []
        can_transcript_words = []

        with open(text_path,'r') as f:
            index = 0
            for utterance in f.readlines():
                w.write(utt_id + " " + utterance + "\n")
                words = utterance.split(" ")
                l3_g2pen = []
                l3_word = []
                for word in words:
                    p_w = g2p(word)
                    p_w = [p.lower() for p in p_w]
                    p_w = [p.rstrip(string.digits) for p in p_w]
                    l3_g2pen += p_w
                    l3_word.append([word, index, index + len(p_w) - 1])
                    index += len(p_w)

                can_transcript_phns += l3_g2pen
                can_transcript_words += l3_word
        
        can_transcript_words_dict[utt_id] =  can_transcript_words
        
        if args.no_g2p_en:
            can_transcript_phns = []
            tmp1 = re.sub('wav', 'TextGrid', p)
            can_phn_path = os.path.normpath('/'.join([args.textgrid_path, tmp1]))
            
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
                
        w1.write(utt_id + " " + wav_path + "\n" )
        w4.write(utt_id + " " + " ".join(del_repeat_sil(can_transcript_phns)) + "\n" )
    
    w.close()
    w1.close()
    w4.close()

    cmd1 = 'compute-fbank-feats --config=conf/fbank.conf scp,p:{}/wav.scp ark:- | '.format(tmp_path)
    cmd2 = 'apply-cmvn --norm-vars=true {}/global_fbank_cmvn.txt ark:- ark:- | '.format('data')
    cmd3 = 'copy-feats --compress={} ark:- ark,scp:{}/fbank.ark,{}/fbank.scp'.format('false', tmp_path, tmp_path)
    cmd4 = '>/dev/null 2>&1'
    cmd = cmd1 + cmd2 + cmd3
    # print(cmd)
    # os.system(cmd)
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    infer(can_transcript_words_dict)
    
    end = time.time()
    time_used = (end - start)
    rtf = time_used / total_wav_time
    print("RTF: %.4f, time used for decode %d sentences: %.4f seconds, total wav length: %.4f seconds" % (rtf, cnt, time_used, total_wav_time))
    return 0

if __name__ == '__main__':
    sys.exit(main())
