import torch
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_read
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_file(file_path):
    """Check file details and system configuration."""
    try:
        logger.info(f"Checking file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
            
        # Check file size
        size = os.path.getsize(file_path)
        logger.info(f"File size: {size} bytes")
        
        # Check file content (first few bytes)
        with open(file_path, 'rb') as f:
            header = f.read(16)
            logger.info(f"File header (hex): {header.hex()}")
        
        # Check soundfile library version and backend
        logger.info(f"soundfile version: {sf.__version__}")
        logger.info(f"libsndfile version: {sf.__libsndfile_version__}")
        
        return True
    except Exception as e:
        logger.error(f"Error checking file: {e}")
        return False

def load_audio(file_path):
    """Load and preprocess audio file using ffmpeg."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    try:
        with open(file_path, "rb") as f:
            input_bytes = f.read()
            
        if len(input_bytes) == 0:
            raise ValueError("Audio file is empty")
            
        return ffmpeg_read(input_bytes, 16000)
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        raise

def run_benchmark(model, processor, audio, num_runs=5, padding_size=1500, max_tokens=100):
    """Run inference multiple times and return average time."""
    times = []
    logger.info(f"Starting benchmark with {num_runs} runs")
    logger.info(f"Configuration: padding_size={padding_size}, max_tokens={max_tokens}")
    
    # Prepare input features (do this once outside the timing loop)
    logger.info("Preparing input features...")
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    inputs.input_features = inputs.input_features.to(dtype=torch.float16, device=model.device)
    
    # Log original input size
    orig_size = inputs.input_features.shape[-1]
    logger.info(f"Original input size: {orig_size}")
    
    # Pad input features if needed
    if inputs.input_features.shape[-1] < padding_size:
        pad_length = padding_size - inputs.input_features.shape[-1]
        logger.info(f"Padding input features from {inputs.input_features.shape[-1]} to {padding_size}")
        inputs.input_features = torch.nn.functional.pad(
            inputs.input_features,
            (0, pad_length),
            mode='constant',
            value=0.0
        )
    elif inputs.input_features.shape[-1] > padding_size:
        logger.info(f"Truncating input features from {inputs.input_features.shape[-1]} to {padding_size}")
        inputs.input_features = inputs.input_features[..., :padding_size]
    
    # Create attention mask
    attention_mask = torch.ones(
        (1, inputs.input_features.shape[-1]),
        dtype=torch.int64,
        device=model.device
    )
    
    # Generation config
    generation_config = {
        "max_new_tokens": max_tokens,
        "num_beams": 1,
        "do_sample": False,
        "language": "english",
        "task": "transcribe",
        "use_cache": True,
        "return_dict_in_generate": True,  # Enable to see cache info
        "output_attentions": False,
        "output_hidden_states": False
    }
    logger.info(f"Generation config: {generation_config}")
    
    # Warmup run with cache monitoring
    logger.info("Performing warmup run...")
    with torch.inference_mode():
        outputs = model.generate(
            inputs.input_features,
            attention_mask=attention_mask,
            **generation_config
        )
        # Log cache info
        if hasattr(outputs, 'past_key_values'):
            cache_size = sum(sum(t.nelement() * t.element_size() for t in p) 
                           for p in outputs.past_key_values) / (1024 * 1024)  # MB
            logger.info(f"KV-Cache size: {cache_size:.2f} MB")
    
    # Timed runs
    logger.info("Starting timed runs...")
    for run in range(num_runs):
        torch.mps.empty_cache()  # Clear GPU cache between runs
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                inputs.input_features,
                attention_mask=attention_mask,
                **generation_config
            )
        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)
        logger.info(f"Run {run + 1}/{num_runs}: {run_time:.3f}s")
        
        # Log sequence length
        if hasattr(outputs, 'sequences'):
            seq_len = outputs.sequences.size(1)
            logger.info(f"Generated sequence length: {seq_len} tokens")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    logger.info(f"Benchmark complete. Mean: {mean_time:.3f}s, Std: {std_time:.3f}s")
    return mean_time, std_time

def main():
    logger.info("Starting benchmark script")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load processor and model
    logger.info("Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-large-v3-turbo",
        language="english",
        use_safetensors=True,
        local_files_only=True
    )
    
    logger.info("Loading model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3-turbo",
        torch_dtype=torch.float16,
        use_safetensors=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)
    
    # Load audio
    logger.info("Loading audio...")
    audio_path = "test_data/jfk.wav"
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return
    audio = load_audio(audio_path)
    
    # Test different configurations
    configs = [
        {"padding_size": 1500, "max_tokens": 100},  # Reduced padding, 100 tokens
        {"padding_size": 1000, "max_tokens": 50},   # Minimal padding, 50 tokens
    ]
    
    for config in configs:
        logger.info(f"\nTesting configuration: {config}")
        mean_time, std_time = run_benchmark(
            model, 
            processor, 
            audio, 
            padding_size=config["padding_size"],
            max_tokens=config["max_tokens"]
        )
        print(f"\nConfig {config}:")
        print(f"Inference time: {mean_time:.3f}s ± {std_time:.3f}s")
    
    logger.info("All benchmarks complete")

if __name__ == "__main__":
    main() 