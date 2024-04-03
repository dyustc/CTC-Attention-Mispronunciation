import glob
import os
import string
import textgrid
import re
import argparse
import soundfile as sf

parser = argparse.ArgumentParser(description="Prepare L2 data")
parser.add_argument("--l2_path",default="/data2/daiyi/dataset/l2arctic",help="l2-Arctic path")
parser.add_argument("--save_path",default="./data/l2_train",help="l2-Arctic path")

args = parser.parse_args()

path = args.l2_path+"/*/annotation/*.TextGrid"
#   Spanish\Vietnamese\Hindi\Mandarin\Korean\Arabic
train_spk = ["EBVS","ERMS","HQTV","PNV","ASI","RRBI","BWC","LXC","HJK","HKK","ABA","SKA"]
dev_spk = ["MBMPS","THV","SVBI","NCC","YDCK","YBAA"]
test_spk = ["NJS","TLV","TNI","TXHC","YKWK","ZHAA"] 
load_error_file = ["YDCK/annotation/arctic_a0209.TextGrid",
                  "YDCK/annotation/arctic_a0272.TextGrid"]
wav_lst = glob.glob(path)
tmp_path = args.save_path
os.system("mkdir %s" % tmp_path)
type_ = args.save_path.split("_")
w = open(tmp_path+"/wrd_text",'w+')
w1 = open(tmp_path+"/wav.scp",'w+')
w2 = open(tmp_path+"/wav_sph.scp",'w+')
w3 = open(tmp_path+"/phn_text",'w+')
w4 = open(tmp_path+"/transcript_phn_text",'w+')
w5 = open(tmp_path+"/a_transcript_phn_text",'w+' )

def del_repeat_sil(phn_lst):
    tmp = [phn_lst[0]]
    for i in range(1,len(phn_lst)):
        if(phn_lst[i] == phn_lst[i-1] and phn_lst[i]=="sil"):
            continue
        else:
            tmp.append(phn_lst[i])
    return tmp

for phn_path in wav_lst:
    if(  "/".join(phn_path.split("/")[-3:]) in load_error_file  ):
        continue
    spk_id = phn_path.split("/")[-3]
    utt_id = spk_id + "_" + phn_path.split("/")[-1][:-9]
    tmp = re.sub("annotation","wav",phn_path)
    wav_path = re.sub("TextGrid","wav",tmp)
    tmp = re.sub("annotation","transcript",phn_path)
    text_path = re.sub("TextGrid","txt",tmp)
    can_phn_path = re.sub("annotation", "arpa_textgrid", phn_path)
    if(spk_id in eval(type_[-1]+"_spk") ):
        cur_phns = []
        transcript_phns = []
        tg = textgrid.TextGrid.fromFile(phn_path)

        can_transcript_phns = []
        can_tg = textgrid.TextGrid.fromFile(can_phn_path)
        for i in tg[1]:            
            if(i.mark == ''):
                transcript_phns.append(("sil"))
                cur_phns.append("sil")
            else:
                trans_human_type = i.mark.split(",")
                trans_phn = trans_human_type[0]
                if trans_phn == 'AH0' or trans_phn == 'ER0':
                    pass
                else:
                    trans_phn = trans_phn.rstrip(string.digits)
                
                if( len(trans_human_type) == 1 ):
                    phn = trans_human_type[0]
                else:
                    phn = trans_human_type[1]
    
                ## phn
                if phn == 'AH0' or phn == 'ER0':
                    pass
                else:
                    phn = phn.rstrip(string.digits+'*_')
                
                if(phn == "sp" or phn == "SIL" or phn == " " or phn == "spn" ):
                    cur_phns.append("sil")
                else:
                    phn = phn.strip(" ")
                    if(phn == "ERR" or phn == "err"):
                        cur_phns.append("err")
                    elif(phn == "ER)" or phn == 'ER0'):
                        cur_phns.append("er0")
                    elif(phn == "AX" or phn == "ax" or phn == "AH)" or phn == 'AH0'):
                        cur_phns.append("ah0")
                    elif(phn == "V``"):
                        cur_phns.append("v")
                    elif(phn == "W`"):
                        cur_phns.append("w")    
                    else:
                        cur_phns.append(phn.lower())
                        
                ## trans phn 
                if(trans_phn == "sp" or trans_phn == "SIL" or trans_phn == " " or trans_phn == "spn" ):
                    transcript_phns.append(("sil"))
                else:
                    trans_phn = trans_phn.strip(" ")
                    if(trans_phn == "ERR" or trans_phn == "err"):
                        transcript_phns.append("err")
                    elif(trans_phn == "ER)" or trans_phn == 'ER0'):
                        transcript_phns.append("er0")
                    elif(trans_phn == "AX" or trans_phn == "ax" or trans_phn == "AH)" or trans_phn == 'AH0'):
                        transcript_phns.append("ah0")
                    elif(trans_phn == "V``"):
                        transcript_phns.append("v")
                    elif(trans_phn == "W`"):
                        transcript_phns.append("w")    
                    else:
                        transcript_phns.append(trans_phn.lower())

        for i in can_tg[1]:
            if i.mark == "" or i.mark == None:
                can_transcript_phns.append("sil")
            
            trans_phn = i.mark
            if trans_phn == 'AH0' or trans_phn == 'ER0':
                pass
            else:
                rans_phn = trans_phn.rstrip(string.digits)

            ## trans phn 
            if(trans_phn == "sp" or trans_phn == "SIL" or trans_phn == " " or trans_phn == "spn" ):
                can_transcript_phns.append(("sil"))
            else:
                trans_phn = trans_phn.strip(" ")
                if(trans_phn == "ERR" or trans_phn == "err"):
                    can_transcript_phns.append("err")
                elif(trans_phn == "ER)" or trans_phn == 'ER0'):
                    can_transcript_phns.append("er0")
                elif(trans_phn == "AX" or trans_phn == "ax" or trans_phn == "AH)" or trans_phn == 'AH0'):
                    can_transcript_phns.append("ah0")
                elif(trans_phn == "V``"):
                    can_transcript_phns.append("v")
                elif(trans_phn == "W`"):
                    can_transcript_phns.append("w")    
                else:
                    can_transcript_phns.append(trans_phn.lower())
                
        f = open(text_path,'r')
        for line in f:
            w.write(utt_id + " " + line.lower() + "\n")
        w1.write(utt_id + " " + wav_path + "\n" )
        w2.write(utt_id + " " + wav_path + "\n" )
        w3.write(utt_id + " " + " ".join(del_repeat_sil(cur_phns)) + "\n" )
        w4.write(utt_id + " " + " ".join(del_repeat_sil(can_transcript_phns)) + "\n" )
        w5.write(utt_id + " " + " ".join(del_repeat_sil(transcript_phns)) + "\n" )

w.close()
w1.close()
w2.close()
w3.close()
w4.close()
w5.close()


