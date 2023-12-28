#!/usr/bin/python
#encoding=utf-8

import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn
import re
from g2p_en import G2p
from tqdm import tqdm

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')

def print_align_space_string(s):
    phonemes = s.split(' ')
    phonemes = [p+" " if len(p) == 1 else p for p in phonemes]
    return ' '.join(phonemes)

def print_align_space_list(l):
    l = [s+" " for s in l]
    return ' '.join(l)

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

    return ' '.join(p_s2), ' '.join(p_s1), ' '.join(l), d

def test():
    args = parser.parse_args()
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)    
    
    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)
        # print('{:50}:{}'.format(k, v))

    use_cuda = opts.use_gpu
    # use_cuda = False
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    
    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    package = torch.load(model_path, map_location=device)
    
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type =  opts.decode_type
    vocab_file = opts.vocab_file
    test_wrd_file = opts.test_wrd_path
    test_wrd_dict = {}
    with open(test_wrd_file) as f:
        for l in f.readlines():
            l = l.strip("\n")
            i = l.find(' ')
            test_wrd_dict[l[:i]] = l[i+1:]    
    # g2p = G2p()

    vocab = Vocab(vocab_file)
    test_dataset = SpeechDataset(vocab, opts.test_scp_path, opts.test_lab_path, opts.test_trans_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha) 
        
    w1 = open("decode_seq.log",'w+')
    w2 = open("human_seq.log",'w+')
    w3 = open("transcribe.log", "w+")
    total_wer = 0

    true_accept = 0
    false_rejection = 0
    false_accept = 0
    true_rejection_correct_diagnose = 0
    true_rejection_wrong_diagnose = 0
    total_phonemes_in_canonical = 0

    # mandarin
    m_true_accept = 0
    m_false_rejection = 0
    m_false_accept = 0
    m_true_rejection_correct_diagnose = 0
    m_true_rejection_wrong_diagnose = 0
    m_total_phonemes_in_canonical = 0

    m_wer = 0
    m_decoder_num = 0

    start = time.time()
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, input_sizes, targets, target_sizes, trans, trans_sizes, utt_list = data
            inputs = inputs.to(device)
            trans = trans.to(device)
            
            probs = model(inputs,trans)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
            
            targets, target_sizes = targets.numpy(), target_sizes.numpy()
            trans, trans_sizes = trans.cpu().numpy(), trans_sizes.numpy()

            labels = []
            canonicals = []
            assert (len(targets) == len(trans))

            for i in range(len(targets)):
                label = [ vocab.index2word[num] for num in targets[i][:target_sizes[i]]]
                labels.append(' '.join(label))

                canonical = [ vocab.index2word[num] for num in trans[i][:trans_sizes[i]]]
                canonicals.append(' '.join(canonical))
                
            ## compute with out sil     
            decoded_nosil = []
            labels_nosil = []
            canonicals_nosil = []
            for i in range(len(labels)):
                hyp = decoded[i].split(" ")
                ref = labels[i].split(" ")
                c_ref = canonicals[i].split(" ")

                ref_precess = [ i   for i in ref if(i != "sil")  ]
                hyp_precess = [ i   for i in hyp if(i != "sil")  ]
                c_ref_precess = [ i   for i in c_ref if(i != "sil")]

                labels_nosil.append(' '.join(ref_precess))
                decoded_nosil.append(' '.join(hyp_precess))
                canonicals_nosil.append(' '.join(c_ref_precess))
    
            for x in range(len(targets)):
                w3.write(utt_list[x] + " " + canonicals_nosil[x] + "\n")
                w2.write(utt_list[x] + " " + labels_nosil[x] + "\n")
                w1.write(utt_list[x] + " " + decoded_nosil[x] + "\n")    
            
            assert (len(targets) == len(labels_nosil))

            for x in range(len(labels_nosil)):
                _, lc_path = decoder.wer(labels_nosil[x], canonicals_nosil[x])
                tmp, _ = decoder.wer(decoded_nosil[x], labels_nosil[x])
                _, dc_path = decoder.wer(decoded_nosil[x], canonicals_nosil[x])

                tmp1, tmp2, tmp3, d1 = print_align_space_canonical_origin(labels_nosil[x], canonicals_nosil[x], lc_path)
                # if utt_list[x][:4] == 'TXHC':
                #     print(utt_list[x])
                #     utterance = test_wrd_dict[utt_list[x]]
                #     print("text      : " + utterance)
                #     print("origin    : " + tmp2)
                #     print("            " + tmp3)
                #     print("canonical : " + tmp1)
                tmp1, tmp2, tmp3, d2 = print_align_space_canonical_origin(decoded_nosil[x], canonicals_nosil[x], dc_path)
                # if utt_list[x][:4] == 'TXHC':
                #     print("canonical : " + tmp1) 
                #     print("            " + tmp3)
                #     print("decode    : " + tmp2)

                total_phonemes_in_canonical += len(d1.keys()) - 1
                if utt_list[x][:4] == 'TXHC':
                    m_total_phonemes_in_canonical += len(d1.keys()) - 1
                
                for k in d1.keys():
                    if k != 'I':
                        if d1[k] == '-' and d2[k] == '-':
                            true_accept += 1
                            if utt_list[x][:4] == 'TXHC':
                                m_true_accept += 1
                        elif d1[k] == '-' and d2[k] != '-':
                            false_rejection += 1
                            if utt_list[x][:4] == 'TXHC':
                                m_false_rejection += 1
                        elif d1[k] != '-' and d2[k] == '-':
                            false_accept += 1
                            if utt_list[x][:4] == 'TXHC':
                                m_false_accept += 1
                        else:
                            if d1[k] == d2[k]:
                                true_rejection_correct_diagnose += 1
                                if utt_list[x][:4] == 'TXHC':
                                    m_true_rejection_correct_diagnose += 1
                            else:
                                true_rejection_wrong_diagnose += 1
                                if utt_list[x][:4] == 'TXHC':
                                    m_true_rejection_wrong_diagnose += 1
                    else:
                        # pass
                        if d1['I'] == [] and d2['I'] == []:
                            pass
                        elif d1['I'] != [] and d2['I'] == []:
                            false_accept += len(d1['I'])
                            if utt_list[x][:4] == 'TXHC':
                                m_false_accept += 1
                        elif d1['I'] == [] and d2['I'] != []:
                            false_rejection += len(d2['I'])
                            if utt_list[x][:4] == 'TXHC':
                                m_false_rejection += 1
                        else:
                            for e in d1['I']:
                                if e in d2['I']:
                                    d1['I'].remove(e)
                                    d2['I'].remove(e)
                                    true_rejection_correct_diagnose += 1
                                    if utt_list[x][:4] == 'TXHC':
                                        m_true_rejection_correct_diagnose += 1
                            false_accept += len(d1['I'])
                            false_rejection += len(d2['I'])
                            if utt_list[x][:4] == 'TXHC':
                                m_false_accept += len(d1['I'])
                                m_false_rejection += len(d2['I'])          
                
                total_wer += tmp
                decoder.num_word += len(labels_nosil[x].split(" "))

                if utt_list[x][:4] == 'TXHC':
                    m_wer += tmp
                    m_decoder_num += len(labels_nosil[x].split(" "))
    
    print("-" * 13 + ' all languages ' + "-" * 13)
    print("total_error:",total_wer)
    print("total_phoneme:",decoder.num_word)
    WER = (float(total_wer) / decoder.num_word)*100
    print("Phoneme error rate on test set: %.4f" % WER)

    true_rejection = true_rejection_correct_diagnose + true_rejection_wrong_diagnose
    print("total : ", total_phonemes_in_canonical, true_accept + false_rejection + false_accept + true_rejection)
    print("TA : ", true_accept)
    print("FR : ", false_rejection)
    print("FA : ", false_accept)
    print("TR : ", true_rejection)
    print("TR correct : ", true_rejection_correct_diagnose)
    print("TR wrong :  ", true_rejection_wrong_diagnose)
    p = (float)(true_rejection) / (true_rejection + false_rejection)
    r = (float)(true_rejection) / (true_rejection + false_accept)
    print('Precision : %.4f' % (p * 100))
    print('Recall : %.4f' % (r * 100))
    print('F1 score : %.4f' % (2 * p * r/(p+r) * 100))

    print("-" * 15 + ' mandarin ' + "-" * 15)
    print("total_error:",m_wer)
    print("total_phoneme:",m_decoder_num)
    WER = (float(m_wer) / m_decoder_num)*100
    print("Phoneme error rate on test set: %.4f" % WER)

    m_true_rejection = m_true_rejection_correct_diagnose + m_true_rejection_wrong_diagnose
    print("total : ", m_total_phonemes_in_canonical, m_true_accept + m_false_rejection + m_false_accept + m_true_rejection)
    print("TA : ", m_true_accept)
    print("FR : ", m_false_rejection)
    print("FA : ", m_false_accept)
    print("TR : ", m_true_rejection)
    print("TR correct : ", m_true_rejection_correct_diagnose)
    print("TR wrong :  ", m_true_rejection_wrong_diagnose)
    p = (float)(m_true_rejection) / (m_true_rejection + m_false_rejection)
    r = (float)(m_true_rejection) / (m_true_rejection + m_false_accept)
    print('Precision : %.4f' % (p * 100))
    print('Recall : %.4f' % (r * 100))
    print('F1 score : %.4f' % (2 * p * r/(p+r) * 100))

    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))
    w1.close()
    w2.close()
    w3.close()

if __name__ == "__main__":
    test()
