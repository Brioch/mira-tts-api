import utils
import config
from mira.model import MiraTTS
import numpy as np
from typing import Optional

mira_tts = MiraTTS("YatharthS/MiraTTS")  ## downloads model from huggingface


def generate_audio(
    text: str,
    voice: str,
    output_format: str,
    speed: float = 1.0,
    chunk_size: int = 250,
    seed: int = 0,
) -> Optional[bytes]:
    if seed != 0:
        utils.set_seed(seed)  # For reproducibility

    voice_file = config.VOICES_DIR + f"{voice}.wav"

    all_audio_data = []

    chunks = utils.chunk_text_by_sentences(text, chunk_size)
    sample_rate = 48000

    # split in chunks
    for chunk in chunks:
        print(f"Generating audio for chunk: {chunk}")

        # Generate the waveform
        context_tokens = mira_tts.encode_audio(voice_file)
        audio_tensor = mira_tts.generate(chunk, context_tokens)

        # adjust speed
        tensor_tuple = utils.apply_speed_factor(audio_tensor, sample_rate, speed)
        audio_tensor = tensor_tuple[0]

        audio_data = audio_tensor.squeeze(0).numpy()
        audio_data = np.clip(audio_data, -1.0, 1.0)  # Clip to prevent saturation
        audio_data = (audio_data * 32767).astype(np.int16)
        all_audio_data.append(audio_data)

    all_audio_data = np.concatenate(all_audio_data)
    bytes_object = utils.encode_audio(all_audio_data, sample_rate, output_format)

    return bytes_object
