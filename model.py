#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
############################################################################

from anonymization.pipelines.sttts_pipeline import STTTSPipeline
import torch
import yaml
from pathlib import Path
import soundfile as sf
import numpy as np


PIPELINES = {
    'sttts': STTTSPipeline
}


def anonymize(input_audio_path):
    config_path = Path("./parameters/anon_ims_sttts_pc_whisper.yaml")
    if config_path.exists():
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
    else:
        raise FileNotFoundError(f"Config file not found at {config_path}")

    devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]

    with torch.no_grad():
        pipeline = PIPELINES[config['pipeline']](
            config=config, 
            force_compute=True, 
            devices=devices, 
            config_name="anon_ims_sttts_pc_whisper"
        )

        anonymized_audio_path = pipeline.run_single_audio(input_audio_path)
        
        # Read the generated audio file
        audio, sr = sf.read(anonymized_audio_path)
        
        # Convert to mono and ensure float32
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
    
    return audio, sr