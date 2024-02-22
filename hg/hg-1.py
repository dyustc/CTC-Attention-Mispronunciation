from transformers import pipeline
from g2p_en import G2p


from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 
from datasets import load_dataset
import torch
import soundfile as sf
import string

# Process raw audio
# transcript: magnets can be found on a can opener
# canonical: m ae g  n  ah t  s  k  ae n  b  iy  f  aw n  d  aa n  ah k  ae n  ow p  ah  n er

transcript = 'magnets can be found on a can opener'
file1 = "男1a.wav"
file2 = "男1.wav"

g2p = G2p()
words = transcript.split(" ")
l3_g2pen = []
for word in words:
    p_w = g2p(word)
    p_w = [p.lower() for p in p_w]
    p_w = [p.rstrip(string.digits) for p in p_w]
    l3_g2pen += p_w
canonical = ' '.join(l3_g2pen)
print(canonical)

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
model = Wav2Vec2ForCTC.from_pretrained("vitouphy/wav2vec2-xls-r-300m-timit-phoneme")

# Read and process the input
audio_input, sample_rate = sf.read(file1)
inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

# Decode id into string
predicted_ids = torch.argmax(logits, axis=-1)      
predicted_sentences = processor.batch_decode(predicted_ids)
print(predicted_sentences)

# pipeline
pipe = pipeline(model="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
output1 = pipe(file1, chunk_length_s=10, stride_length_s=(4, 2))
output2 = pipe(file2, chunk_length_s=10, stride_length_s=(4, 2))
print(output1['text'])
print(output2['text'])

pipe = pipeline(model="mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme")
output1 = pipe(file1)
output2 = pipe(file2)
print(output1['text'])
print(output2['text'])

pipe = pipeline("audio-classification", model="junbro1016/pronunciation-scoring-prosodic")
output1 = pipe(file1)
output2 = pipe(file2)
print(output1)
print(output2)

pipe = pipeline("audio-classification", model="junbro1016/pronunciation-scoring-fluency")
output1 = pipe(file1)
output2 = pipe(file2)
print(output1)
print(output2)

pipe = pipeline("audio-classification", model="junbro1016/pronunciation-scoring-completeness")
output1 = pipe(file1)
output2 = pipe(file2)
print(output1)
print(output2)

pipe = pipeline("audio-classification", model="junbro1016/pronunciation-scoring-accuracy")
output1 = pipe(file1)
output2 = pipe(file2)
print(output1)
print(output2)
