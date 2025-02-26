import sys
import requests
from loguru import logger
import soundfile
import datetime

sys.path.append("e2e_tts/models")
sys.path.append("e2e_tts")
from models import *
from src.api.utils import *

class Synthesizer:
    def __init__(self, acoustic_path: str, vocoder_path: str, output_dir: str = "outputs") -> None:
        self.api_text_normalize = "https://demo.corenlp.admicro.vn/tts-normalization"
        self.model = TTS(
            acoustic_path=acoustic_path,
            vocoder_path=vocoder_path,
        )
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def tts_to_file(self, text: str, file_path: str, speed: float=1):
        return self.synthesis(text, file_path, speed)

    def synthesis(self, text: str, save_filepath: str=None, speed: float=1, speaker_id: str="hn_minhphuong", sr: int=22050):
        assert len(text) > 0

        try:
            r = requests.post(self.api_text_normalize, json={"text": text})
            text = r.json()["result"]
            logger.info(f"normlized text: {text}")
        except:
            logger.info("requests api text normalize failed")

        if not save_filepath:
            save_filepath = os.path.join(self.output_dir, datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + ".wav")

        silent_distance =  0.5
        audio = self.model.inference(
            texts=text,
            speaker_id=speaker_id,
            pitch_control=1.0, 
            energy_control=1.0, 
            duration_control=1.0,
            silence_distance=silent_distance,
        )
        soundfile.write(save_filepath, audio, samplerate=sr)
        if speed != 1:
            save_filepath = audio_speed_change(save_filepath, speed_rate=speed)
        return save_filepath
