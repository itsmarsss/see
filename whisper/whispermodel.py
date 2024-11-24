import numpy as np
from scipy.io.wavfile import write
from .setup_whisper import setup_whisper

class WhisperTranscriber:
    def __init__(self):
        """
        Initialize the WhisperTranscriber with the model and configuration.
        """
        self.model, self.model_size, self.sample_rate = setup_whisper()

        print(f"Initialized WhisperTranscriber with model {self.model_size}.")

    def transcribe_audio(self, file_path: str) -> tuple:
        """
        Transcribe the recorded audio using the Whisper model.

        Args:
            file_path (str): The path to the temporary audio file.

        Returns:
            tuple: The transcription text and the detected language.
        """
        segments, info = self.model.transcribe(file_path, beam_size=5)
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        full_transcription = " ".join([segment.text for segment in segments])
        if info.language_probability <= 0.7:
            print("Low language probability detected. Transcription may be inaccurate.")
            full_transcription=""
        print(f"Transcription: {full_transcription}")
        return full_transcription, info.language

    # def on_press(self, key: keyboard.Key) -> None:
    #     """
    #     Handle the key press event for starting the recording.

    #     Args:
    #         key (keyboard.Key): The key that was pressed.
    #     """
    #     if key == keyboard.Key.space:
    #         if not self.is_recording:
    #             self.is_recording = True
    #             print("Recording started.")

    # def on_release(self, key: keyboard.Key) -> bool:
    #     """
    #     Handle the key release event for stopping the recording.

    #     Args:
    #         key (keyboard.Key): The key that was released.

    #     Returns:
    #         bool: False to stop the listener if the spacebar is released.
    #     """
    #     if key == keyboard.Key.space:
    #         if self.is_recording:
    #             self.is_recording = False
    #             print("Recording stopped.")
    #             return False

    # def on_toggle(self, key: keyboard.Key) -> bool:
    #     """
    #     Handle the key press event for toggling the recording.

    #     Args:
    #         key (keyboard.Key): The key that was pressed.

    #     Returns:
    #         bool: False to stop the listener if the spacebar is released.
    #     """
    #     if key == keyboard.Key.space:
    #         self.is_recording = not self.is_recording
    #         print("Recording started." if self.is_recording else "Recording stopped.")
    #         return False if not self.is_recording else None

    # def record_audio(self) -> np.ndarray:
    #     """
    #     Record audio using the microphone.

    #     Returns:
    #         np.ndarray: The recorded audio data.
    #     """
    #     recording = np.array([], dtype="float64").reshape(0, 2)
    #     frames_per_buffer = int(self.sample_rate * 0.1)
    #     p = pyaudio.PyAudio()
    #     stream = p.open(format=pyaudio.paFloat32,
    #                     channels=2,
    #                     rate=self.sample_rate,
    #                     input=True,
    #                     frames_per_buffer=frames_per_buffer)

    #     # Manually manage the stream's lifecycle because the Stream object from pyaudio
    #     # does not support the context manager protocol (with statement).
        
    #     if self.mode == "toggle":
    #         listener = keyboard.Listener(on_press=self.on_toggle)
    #     else:
    #         listener = keyboard.Listener(on_press=self.on_press)

    #     listener.start()

    #     try:
    #         while True:
    #             if self.is_recording:
    #                 data = stream.read(frames_per_buffer)
    #                 chunk = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
    #                 recording = np.vstack([recording, chunk])
    #             if not self.is_recording and len(recording) > 0:
    #                 break
    #     finally:
    #         listener.stop()
    #         listener.join()
    #         stream.stop_stream()
    #         stream.close()
    #         p.terminate()

    #     print(f"Recorded audio shape: {recording.shape}")
    #     print(f"Recorded audio sample: {recording[:5]}")  # Log the first 5 samples

    #     print("Audio recording completed.")
    #     return recording

    # def save_temp_audio(self, recording: np.ndarray) -> str:
    #     """
    #     Save the recorded audio to a temporary file.

    #     Args:
    #         recording (np.ndarray): The recorded audio data.

    #     Returns:
    #         str: The path to the temporary file.
    #     """
    #     temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    #     write(temp_file.name, self.sample_rate, recording)
    #     print(f"Audio saved to temporary file {temp_file.name}")
    #     return temp_file.name

    

    # def run(self) -> tuple:
    #     """
    #     Run the WhisperTranscriber to record and transcribe audio.

    #     Returns:
    #         tuple: The transcription text and the detected language.
    #     """
    #     if self.mode == "toggle":
    #         print("Press the spacebar to start recording. Press again to stop it.")
    #     else:
    #         print("Hold the spacebar to start recording...")

    #     recording = self.record_audio()
    #     file_path = self.save_temp_audio(recording)

    #     start_time = time.time()
    #     prompt, language = self.transcribe_audio(file_path)
    #     end_time = time.time()

    #     print(f"Execution time for Transcription block: {end_time - start_time:.2f} seconds")
    #     return prompt, language
