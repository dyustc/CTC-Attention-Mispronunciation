# How to run

# 目前误读检测和诊断，根据竞品用户场景，当前demo支持单词，后续上线前会支持词组和句子。

# 这次的更新主要是优化了字典模块，单词的发音音标，以及相应的模型训练, 同时在 CMU39 音素的基础上，增添了 AH0， ER0 两个音素，而且本次针对单词没有根据易混肴的音素，做模型推理后的美化，遵循非常严格的纠错准则。同时，此次更新提供了单词的音标标注，和词语释义。

# 针对单词，后续会跑一个全量单词（常见词， 派生词， 字典外词，特殊符号词，缩写词，常见词组）的测试。此外, 如果有任何单词导致程序无法运行（非检测诊断效果不佳），欢迎提供具体单词；而针对诊断效果不佳的， 会根据模型持续迭代优化。

# 另 tts，已经支持，但碍于环境配置较为复杂，此次先不提供。

# 词组和句子的翻译暂不支持，后续提供。

# 跑之前，需要进行以下命令行安装

# for mac M1/M2
` $ pip install phonemizer `
` $ arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`
` $ arch -x86_64 /usr/local/bin/brew install espeak`

# 运行，目前单词放在 vocabulary/single 文件夹下， 也可以自行按照格式添加其他路径单词
` $ cd egs/attention_aug`
` $ python infer.py --wav_transcript_path ../vocabulary/single`
