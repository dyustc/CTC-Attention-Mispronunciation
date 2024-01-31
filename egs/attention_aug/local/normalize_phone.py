#encoding=utf-8

import os
import sys
import argparse
import random
import soundfile as sf

parser = argparse.ArgumentParser(description="Normalize the phoneme on TIMIT")
parser.add_argument("--map", default="./decode_map_48-39/phones.60-48-39.map", help="The map file")
parser.add_argument("--to", default=48, help="Determine how many phonemes to map")
parser.add_argument("--src", default='./data_prepare/train/phn_text', help="The source file to mapping")
parser.add_argument("--tgt", default='./data_prepare/train/48_text' ,help="The target file after mapping")

def main():
    args = parser.parse_args()
    if not os.path.exists(args.map) or not os.path.exists(args.src):
        print("Map file or source file not exist !")
        sys.exit(1)
    
    map_dict = {}
    with open(args.map) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if args.to == "60-48":
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[1]
            elif args.to == "60-39": 
                if len(line) == 1:
                    map_dict[line[0]] = ""
                else:
                    map_dict[line[0]] = line[2]
            elif args.to == "48-39":
                if len(line) == 3:
                    map_dict[line[1]] = line[2]
            else:
                print("%s phonemes are not supported" % args.to)
                sys.exit(1)
    
    print(args.tgt)
    # if 'transcript' not in args.tgt:
    if 1:
        with open(args.src, 'r') as rf, open(args.tgt, 'w') as wf:
            for line in rf.readlines():
                line = line.strip().split(' ')
                uttid, utt = line[0], line[1:]
                map_utt = [ map_dict[phone] for phone in utt if map_dict[phone] != "" ]
                wf.writelines(uttid + ' ' + ' '.join(map_utt) + '\n')
        return
    
    print(args.tgt)
    # data augmentation
    # PS, VS, CP
    def cp_dict():
        vowel_s = {
            'aa' : 'ah',
            'ah' : 'ih',
            'ae' : 'eh',
            'ih' : 'iy',
            'iy' : 'ih'
        }

        consonant_s = {
            'd' : 'dh',
            'dh' : 'd',
            't' : 'd',
            'sh' : 's',
            's' : 'z',
            'z' : 's'
        }

        mandarin_s = {
            'z' : 's',
            'dh' : 'd',
            'ih' : 'iy',
            'n' : 'ng',
            'v' : 'f',
        }

        mandarin_d = ['d', 't', 'r', 'l', 'n']
        mandarin_i = ['ah', 'ax', 'ih', 'n', 'r']

        mixed_dict = {}
        for k, v in vowel_s.items():
            if v not in mixed_dict:
                mixed_dict[v] = [k]
            else:
                mixed_dict[v].append(k)
            
        for k, v in consonant_s.items():
            if v not in mixed_dict:
                mixed_dict[v] = [k]
            else:
                mixed_dict[v].append(k)

        for k, v in mandarin_s.items():
            if v not in mixed_dict:
                mixed_dict[v] = [k]
            else:
                mixed_dict[v].append(k)
            
        # insert error
        # for k in mandarin_i:
        #     if k in mixed_dict:
        #         mixed_dict[k].append('')
        #     else:
        #         mixed_dict[k] = ['']
            
        # # delete error
        # for k in mixed_dict.keys():
        #     temp = []
        #     for k1 in mandarin_d:
        #         temp.append(' '.join([k, k1]))
        #         temp.append(' '.join([k1, k]))
        #     mixed_dict[k].append(temp)

        return mixed_dict
    
    def data_augment(p, mixed_dict, aug_type = None):
        if not aug_type:
            return p
        
        if aug_type == 'CP':
            if p in mixed_dict:
                p1 = random.choice(mixed_dict[p])
                if type(p1) == list:
                    return random.choice(p1)
                else:
                    return p1
            else:
                return p

    with open(args.src, 'r') as rf, open(args.tgt, 'w') as wf:
        lines = rf.readlines()
        l = len(lines)
        ratio = 0.4
        n = int(l * ratio)
        print(l, n)
        indexs = list(range(l))
        random.shuffle(indexs)

        mixed_dict = cp_dict()
        print(mixed_dict)

        for i in range(l):
            line = lines[i]
            line = line.strip().split(' ')
            uttid, utt = line[0], line[1:]
            if i not in indexs[:n]:
                map_utt = [ map_dict[phone] for phone in utt if map_dict[phone] != "" ]
                wf.writelines(uttid + ' ' + ' '.join(map_utt) + '\n')
            else:
                ratio_1 = 0.5
                l1 = len(utt)
                n1 = int(ratio_1 * l1)
                indexs1 = list(range(l1))
                random.shuffle(indexs1)

                map_utt = []
                for j in range(l1):
                    phone = utt[j]
                    if map_dict[phone] != "":
                        if j not in indexs1[:n1]:
                            map_utt.append(map_dict[phone])
                        else:
                            p = data_augment(map_dict[phone], mixed_dict, 'CP')
                            if p != '':
                                map_utt.append(p)

                wf.writelines(uttid + ' ' + ' '.join(map_utt) + '\n')


if __name__ == "__main__":
    main()
