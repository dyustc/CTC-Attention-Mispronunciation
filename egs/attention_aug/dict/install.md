# espeak install on Mac M1/M2

- arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
- arch -x86_64 /usr/local/bin/brew install espeak
- export PHONEMIZER_ESPEAK_LIBRARY=/usr/local/Cellar/espeak/1.48.04_1/lib/libespeak.dylib
or
- export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.dylib
