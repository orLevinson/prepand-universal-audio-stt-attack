
# prepand universal audio stt attack


This project aims to create an attacked aimed to obfuscate audio from Whisper-base STT model.

*I hope you will understand the code, because I don't.*

## Install dependencies

This code runs on python. To run the code and generate the universal audio download the following dependencies:

```bash
pip install torch torchaudio datasets git+https://github.com/openai/whisper.git
```
and also install the latest ffmpeg (this is for ubuntu-based systems):

```bash
sudo apt-get install ffmpeg
```

## Run the code

Just run it regulary, you've got 2 working versions:
```bash
english-multilingual
english-targeted
```
those are for different configs of the model (during transcribe, choose whether the model will be multilingual or only english).

## Use the audio

To use the adversary audio you've got, you have to prepand it to an already existing audio file, and then feed it to whisper to transcribe. If all works good you should get gibberish after running whisper every time.

There are already a working sample inside every folder called ```sample.wav``` and ```universal_endoftext_prefix.wav``` that holds and already trained adversary audio.

### Have fun! 


## Authors

- [@orLevinson](https://www.github.com/orLevinson)

