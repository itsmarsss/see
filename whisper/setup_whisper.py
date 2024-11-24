from faster_whisper import WhisperModel
import torch

# Constants for Configuration
MODEL_SIZE = "large-v3"
SAMPLE_RATE = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "int8"


def setup_whisper() -> tuple:
    """
    Set up the Whisper model and configuration.

    Returns:
        tuple: A tuple containing the Whisper model, model size, sample rate, and mode.
    """
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    print(f"Whisper model '{MODEL_SIZE}' set up successfully.")

    return model, MODEL_SIZE, SAMPLE_RATE


#prompt, language = transcriber.run()