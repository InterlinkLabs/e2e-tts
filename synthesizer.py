import os
from TTS.api import TTS
from datetime import datetime
from loguru import logger
from e2e_tts.src.api.inference import Synthesizer as SynthesizerVN

def gen_filename():
    return datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ".wav"

class Synthesizer:
    def __init__(self, output_dir: str="outputs") -> None:
        eng_model = TTS(model_name="tts_models/en/ljspeech/vits")
        myanmar_model = TTS(model_name="tts_models/mya/fairseq/vits")

        vie_model = SynthesizerVN(
            acoustic_path="e2e_tts/exps/acoustic/statedict.pt",
            vocoder_path="e2e_tts/exps/vocoder/statedict.pt",
            output_dir=output_dir
        )
        
        self.model_dict = {
            "eng": eng_model,
            "mya": myanmar_model,
            "vie": vie_model
        }
        voice_conversion_model = eng_model
        voice_conversion_model.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        self.voice_conversion_model = voice_conversion_model
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def synthesis(
        self,
        text: str,
        language: str,
        target_filepath: str = None,
        speed: float=1.0
    ) -> str:
        if not isinstance(speed, float):
            speed = float(speed)
        logger.info(f"text: {text}\nlanguage: {language}\nspeed: {speed}\n target: {target_filepath}")
        
        language = language.split()[0]

        tts_output_filepath = os.path.join(self.output_dir, gen_filename())
        self.model_dict[language].tts_to_file(text, file_path=tts_output_filepath, speed=speed)
        if target_filepath:
            vc_output_filepath = os.path.join(self.output_dir, gen_filename())
            self.voice_conversion_model.voice_conversion_to_file(
                source_wav=tts_output_filepath,
                target_wav=target_filepath,
                file_path=vc_output_filepath
            )
        else:
            vc_output_filepath = None
        return tts_output_filepath, vc_output_filepath
    
    def voice_conversion(self, src_filepath: str, target_filepath: str):
        save_filepath = os.path.join(self.output_dir, gen_filename())
        self.voice_conversion_model.voice_conversion_to_file(
            source_wav=src_filepath,
            target_wav=target_filepath,
            file_path=save_filepath
        )
        return save_filepath