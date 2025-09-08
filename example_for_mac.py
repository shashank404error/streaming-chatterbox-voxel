import torch
import torchaudio as ta
from src.chatterbox.tts import ChatterboxTTS
from multilingual_app import generate_tts_audio

# Detect device (Mac with M1/M2/M3/M4)
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = "I know what you're thinking.. 'How the hell does this work???' 'Believe me, even I don't know...' Looks like one day you wake up and just start speaking. And you're like.... 'Oh, I know what you're thinking. How the hell does this work?'"

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "./resources/audio5.mp3"
# wav = model.generate(
#     text, 
#     audio_prompt_path=AUDIO_PROMPT_PATH,
#     exaggeration=0.0,
#     cfg_weight=0.5
#     )
# ta.save("test-sohel_talking_head1.wav", wav, model.sr)

wav = generate_tts_audio(
    text,
    "hi", 
    audio_prompt_path_input=AUDIO_PROMPT_PATH,
    exaggeration_input=0.0,
    temperature_input=0,
    cfgw_input=0.5
    )
ta.save("test-hindi.wav", wav, model.sr)