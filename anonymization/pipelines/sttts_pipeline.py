from pathlib import Path
import time
import tempfile
import os

from anonymization.modules import (
    SpeechRecognition,
    SpeechSynthesis,
    ProsodyExtraction,
    ProsodyAnonymization,
    SpeakerExtraction,
    SpeakerAnonymization,
)
import typing
from utils import setup_logger

logger = setup_logger(__name__)

class STTTSPipeline:
    def __init__(self, config: dict, force_compute: bool, devices: list, config_name: str):
        """
        Instantiates a STTTSPipeline with the complete feature extraction,
        modification and resynthesis.

        This pipeline consists of:
              - ASR -> phone sequence                    -
        input - (prosody extr. -> prosody anon.)         - TTS -> output
              - speaker embedding extr. -> speaker anon. -

        Args:
            config (dict): a configuration dictionary.
            force_compute (bool): if True, forces re-computation of all steps.
            devices (list): a list of torch-interpretable devices.
            config_name (str): a name identifier for the configuration.
        """
        self.total_start_time = time.time()
        self.config = config
        self.config_name = config_name
        model_dir = Path(config.get("models_dir", "models"))
        vectors_dir = Path(config.get("vectors_dir", "original_speaker_embeddings"))
        self.results_dir = Path(config.get("results_dir", "results"))
        self.data_dir = Path(config["data_dir"]) if "data_dir" in config else None
        save_intermediate = config.get("save_intermediate", True)

        modules_config = config["modules"]

        # ASR component
        self.speech_recognition = SpeechRecognition(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["asr"],
            force_compute=force_compute,
        )

        # Speaker component
        self.speaker_extraction = SpeakerExtraction(
            devices=devices,
            save_intermediate=save_intermediate,
            settings=modules_config["speaker_embeddings"],
            force_compute=force_compute,
        )
        if 'anonymizer' in modules_config["speaker_embeddings"]:
            self.speaker_anonymization = SpeakerAnonymization(
                vectors_dir=vectors_dir,
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["speaker_embeddings"],
                force_compute=force_compute,
            )
        else:
            self.speaker_anonymization = None

        # Prosody component
        if "prosody" in modules_config:
            self.prosody_extraction = ProsodyExtraction(
                device=devices[0],
                save_intermediate=save_intermediate,
                settings=modules_config["prosody"],
                force_compute=force_compute,
            )
            if "anonymizer" in modules_config["prosody"]:
                self.prosody_anonymization = ProsodyAnonymization(
                    save_intermediate=save_intermediate,
                    settings=modules_config["prosody"],
                    force_compute=force_compute,
                )
            else:
                self.prosody_anonymization = None
        else:
            self.prosody_extraction = None

        # TTS component
        self.speech_synthesis = SpeechSynthesis(
            devices=devices,
            settings=modules_config["tts"],
            save_output=self.config.get("save_output", True),
            force_compute=force_compute,
        )

    def run_single_audio(self, audio_file: str) -> str:
        """
        Runs the anonymization pipeline on a single audio file.
        Accepts a direct audio file path and returns the path to the resulting anonymized audio.
        
        Args:
            audio_file (str): Path to the input audio file.
        
        Returns:
            str: Path to the anonymized output audio file.
        """
        logger.info("Processing single audio file: %s", audio_file)
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file {audio_file} does not exist.")

        # Create a temporary directory containing the minimal files expected by the pipeline.
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            # Use the stem of the audio file as an utterance ID.
            utt_id = audio_path.stem

            # Create minimal Kaldi-like files:
            # 1. wav.scp: mapping from utterance ID to audio file path.
            wav_scp_file = tmpdir / "wav.scp"
            with open(wav_scp_file, "w", encoding="utf8") as f:
                f.write(f"{utt_id} {str(audio_path.resolve())}\n")
            
            # 2. utt2spk: assign a dummy speaker (e.g., "spk1").
            utt2spk_file = tmpdir / "utt2spk"
            with open(utt2spk_file, "w", encoding="utf8") as f:
                f.write(f"{utt_id} spk1\n")
            
            # 3. spk2utt: map the dummy speaker to the utterance.
            spk2utt_file = tmpdir / "spk2utt"
            with open(spk2utt_file, "w", encoding="utf8") as f:
                f.write(f"spk1 {utt_id}\n")
            
            # 4. text: create a dummy transcription.
            text_file = tmpdir / "text"
            with open(text_file, "w", encoding="utf8") as f:
                f.write(f"{utt_id} dummy_transcription\n")
            
            # 5. utt2dur: compute the audio duration.
            try:
                import torchaudio
                info = torchaudio.info(str(audio_path))
                duration = info.num_frames / info.sample_rate
            except Exception:
                duration = 0.0
            utt2dur_file = tmpdir / "utt2dur"
            with open(utt2dur_file, "w", encoding="utf8") as f:
                f.write(f"{utt_id} {duration:.2f}\n")
            
            # 6. spk2gender: assign a dummy gender.
            spk2gender_file = tmpdir / "spk2gender"
            with open(spk2gender_file, "w", encoding="utf8") as f:
                f.write("spk1 m\n")
            
            # Define a dataset name for this run.
            dataset_name = "single_audio"

            # --- Pipeline steps ---

            # Step 1: Speech Recognition
            start_time = time.time()
            texts = self.speech_recognition.recognize_speech(dataset_path=tmpdir, dataset_name=dataset_name)
            logger.info("Speech recognition completed in %.2f seconds", time.time() - start_time)

            # Step 2: Speaker Extraction
            start_time = time.time()
            spk_embeddings = self.speaker_extraction.extract_speakers(dataset_path=tmpdir, dataset_name=dataset_name)
            logger.info("Speaker extraction completed in %.2f seconds", time.time() - start_time)

            # Step 3: Prosody Extraction (if available)
            if self.prosody_extraction:
                start_time = time.time()
                prosody = self.prosody_extraction.extract_prosody(
                    dataset_path=tmpdir, dataset_name=dataset_name, texts=texts
                )
                logger.info("Prosody extraction completed in %.2f seconds", time.time() - start_time)
            else:
                prosody = None

            # Step 4: Speaker Anonymization (if available)
            if self.speaker_anonymization:
                start_time = time.time()
                anon_embeddings = self.speaker_anonymization.anonymize_embeddings(
                    speaker_embeddings=spk_embeddings, dataset_name=dataset_name
                )
                logger.info("Speaker anonymization completed in %.2f seconds", time.time() - start_time)
            else:
                anon_embeddings = spk_embeddings

            # Step 5: Prosody Anonymization (if available)
            if self.prosody_anonymization:
                start_time = time.time()
                anon_prosody = self.prosody_anonymization.anonymize_prosody(prosody=prosody)
                logger.info("Prosody anonymization completed in %.2f seconds", time.time() - start_time)
            else:
                anon_prosody = prosody

            # Step 6: Speech Synthesis
            start_time = time.time()
            wav_scp = self.speech_synthesis.synthesize_speech(
                dataset_name=dataset_name,
                texts=texts,
                speaker_embeddings=anon_embeddings,
                prosody=anon_prosody,
                emb_level=anon_embeddings.emb_level,
            )
            logger.info("Speech synthesis completed in %.2f seconds", time.time() - start_time)

            # Retrieve the output audio path from the synthesis result.
            # We assume that wav_scp is a dictionary mapping utterance IDs to output file paths.
            output_audio = wav_scp.get(utt_id, None)
            if output_audio is None:
                logger.error("Anonymized audio not found in synthesis results.")
                return ""
            else:
                logger.info("Anonymized audio available at: %s", output_audio)
                return output_audio
