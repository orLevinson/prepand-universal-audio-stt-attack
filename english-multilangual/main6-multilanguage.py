import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
import whisper
from whisper.tokenizer import get_tokenizer
# --- Settings ---
SAMPLE_RATE = 16000
PREFIX_SECONDS = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from whisper.audio import log_mel_spectrogram, pad_or_trim

VOCAB_SIZE = 51864  # Whisper vocab size

# --- Load Whisper ---
model = whisper.load_model("base").to(DEVICE)
tokenizer = get_tokenizer(multilingual=model.is_multilingual)
eot_token_id = tokenizer.eot

# --- Learnable Prefix ---
prefix_len = int(SAMPLE_RATE * PREFIX_SECONDS)
universal_prefix = torch.randn(prefix_len, requires_grad=True, device=DEVICE)

# --- Dataset (3 silent clips, replace with real samples) ---
dataset = load_dataset("google/fleurs", "en_us", split="train[:500]")

def preprocess_audio(sample):
    waveform = torch.tensor(sample["audio"]["array"])
    waveform = torchaudio.functional.resample(waveform, sample["audio"]["sampling_rate"], 16000)
    if waveform.shape[-1] < 32000:
        waveform = torch.nn.functional.pad(waveform, (0, 32000 - waveform.shape[-1]))
    return waveform[:32000]  # 2 seconds max

train_audios = [preprocess_audio(s) for s in dataset]


# --- Mel Converter ---
def audio_to_mel(audio):
    return log_mel_spectrogram(audio.float())


# --- Optimization ---
optimizer = torch.optim.Adam([universal_prefix], lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss()

# --- Train Loop ---
for epoch in range(30):
    total_loss = 0
    for audio in train_audios:
        optimizer.zero_grad()
        x = torch.cat([universal_prefix, audio.to(universal_prefix.device)], dim=0).unsqueeze(0)
        x = pad_or_trim(x)  # Pads/trims to 48000 samples (30 sec)
        mel = log_mel_spectrogram(x.float()).to(DEVICE)  # (80, 3000)

        # Forward through encoder + decoder
        enc_out = model.encoder(mel)
        dec_input = torch.tensor([[tokenizer.sot]], device=DEVICE)
        logits = model.decoder(dec_input, enc_out)

        loss = loss_fn(logits[0, 0], torch.tensor(eot_token_id).to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: loss={total_loss:.4f}")

# --- Save Result ---
final_prefix = universal_prefix.detach().cpu().unsqueeze(0)
torchaudio.save("universal_endoftext_prefix.wav", final_prefix, SAMPLE_RATE)
