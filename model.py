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
import requests
import os
from tqdm import tqdm

PIPELINES = {
    'sttts': STTTSPipeline
}

MODEL_FILES = {
    "embedding_function.pt": "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt",
    "embedding_gan.pt": "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_gan.pt",
    "aligner.pt": "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/aligner.pt",
    "ToucanTTS_Meta.pt": "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/ToucanTTS_Meta.pt",
    "Avocodo.pt": "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/Avocodo.pt"
}

PARAMETERS_DIR = Path("./parameters")
PARAMETERS_DIR.mkdir(parents=True, exist_ok=True)

def download_model(url: str, filename: str, chunk_size: int = 1024):
    """
    Download a model file with progress bar and error handling
    """
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        
        total_size = int(resp.headers.get('content-length', 0))
        with open(PARAMETERS_DIR/filename, "wb") as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = f.write(data)
                bar.update(size)
                
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename}: {str(e)}")

def verify_models():
    """
    Check if all required models exist and download missing ones
    """
    for filename, url in MODEL_FILES.items():
        file_path = PARAMETERS_DIR/filename
        if not file_path.exists():
            print(f"Model file {filename} not found, downloading...")
            download_model(url, filename)
            
            if not file_path.exists() or os.path.getsize(file_path) == 0:
                raise FileNotFoundError(f"Failed to download valid {filename}")


def anonymize(input_audio_path):
    verify_models()

    config_path = PARAMETERS_DIR/"anon_ims_sttts_pc_whisper.yaml"
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
        
        audio, sr = sf.read(anonymized_audio_path)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
    
    return audio, sr