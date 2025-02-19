import os
# Silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import sys
import subprocess
from pathlib import Path
import tempfile
import pyaudio
import threading
import time
import torch
import numpy as np
import soundfile as sf
import json
import atexit
import pyperclip
from pynput import keyboard
from pynput.keyboard import Key, Controller
from pynput._util.darwin import ListenerMixin
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio
import accelerate

# Spinner frames for recording/processing indicator
SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

# Silence all debug logging
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow logging

# Load settings
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "trigger_key": "f20",
            "keep_model_loaded": True,
            "audio": {
                "format": "float32",
                "channels": 1,
                "sample_rate": 16000,
                "chunk_size": 2048
            },
            "whisper": {
                "model": "openai/whisper-large-v3-turbo",
                "use_gpu": True,
                "streaming": {
                    "buffer_duration": 3.0,
                    "temperature": 0.0,
                    "no_speech_threshold": 0.95,
                    "compression_ratio_threshold": 1.6,
                    "condition_on_previous_text": False
                },
                "final": {
                    "temperature": 0.0,
                    "no_speech_threshold": 0.6,
                    "compression_ratio_threshold": 1.5,
                    "condition_on_previous_text": False,
                    "initial_prompt": "This is a transcript with minimal punctuation.",
                    "max_new_tokens": 448
                }
            }
        }

def ensure_venv():
    """Ensure we're running in the correct virtual environment"""
    venv_path = Path('venv')
    venv_python = venv_path / 'bin' / 'python'
    
    if not sys.prefix.endswith('venv'):
        try:
            if not venv_path.exists():
                subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
                subprocess.run([str(venv_path / 'bin' / 'pip'), 'install', '-r', 'requirements.txt'], check=True)
            os.execv(str(venv_python), [str(venv_python), __file__])
        except Exception as e:
            print(f"Virtual environment error: {e}")
            sys.exit(1)

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\n=== Checking FFmpeg ===")
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ FFmpeg not found!")
        print("Whisper requires FFmpeg for audio processing.")
        print("\nTo install FFmpeg:")
        if sys.platform == "darwin":
            print("1. Open Terminal")
            print("2. Run: brew install ffmpeg")
            print("\nIf you don't have Homebrew installed:")
            print("1. First install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            print("2. Then run: brew install ffmpeg")
        sys.exit(1)

def check_microphone_permission():
    """Check and request microphone permission on macOS"""
    if sys.platform != "darwin":
        return True
    
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        stream.read(1024)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return True
    except Exception as e:
        print("❌ Microphone access denied - Please grant permission in System Settings > Privacy & Security > Microphone")
        return False

# First, ensure we're in the virtual environment
ensure_venv()

# Check microphone permission before proceeding
if not check_microphone_permission():
    sys.exit(1)

# Now we can safely import our dependencies
import tempfile
import pyaudio
import threading
import time
import whisper
import torch
import numpy as np
import soundfile as sf
import json

class VoiceRecorder:
    def __init__(self):
        """Initialize the voice recorder"""
        # Disable terminal input using stty
        os.system('stty -echo')
        
        loading_start = time.time()
        print("\n--- SnipeGPT, Speech-To-Text ---\n")
        
        self.settings = load_settings()
        self.frames = []
        self.start_time = None
        self.stream = None
        self.audio = None
        self.processor = None
        self.model = None
        self.device = None
        self.dtype = None
        self.temp_dir = tempfile.gettempdir()
        self.recording = False
        self.is_processing = False
        self.monitoring = True
        self.spinner_idx = 0
        self.last_f20_time = 0
        self.max_recording_time = 60
        self.loading_message = ""
        self.is_loading = False
        
        self.keyboard_controller = Controller()
        
        def custom_handle(listener_self, proxy, event_type, event, refcon):
            try:
                key = listener_self._event_to_key(event)
                if key is not None and key == keyboard.Key.f20 and event_type == 10:
                    toggle_thread = threading.Thread(target=self.toggle_recording)
                    toggle_thread.start()
                return event  # Allow all events to pass through
            except Exception as e:
                print(f"Key handler error: {e}")
                return event  # Allow events even on error
        
        ListenerMixin._handler = custom_handle
        self.keyboard_listener = keyboard.Listener(
            darwin_intercept=True,
            suppress=False,  # Don't suppress keyboard events
            _intercept=True
        )
        
        try:
            # Start loading animation thread
            self.is_loading = True
            loading_thread = threading.Thread(target=self.animate_loading, daemon=True)
            loading_thread.start()
            
            # Pre-initialize device
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.dtype = torch.float16
            
            # Setup audio first
            self.loading_message = "Microphone check"
            self.setup_audio()
            print("\r✓ Microphone check" + " " * 20)  # Clear spinner
            
            # Initialize processor
            self.loading_message = "Initialize processor"
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                language="english",
                use_safetensors=True,
                local_files_only=False
            )
            print("\r✓ Initialize processor" + " " * 20)  # Clear spinner
            
            # Load and optimize model
            self.model_start_time = time.time()  # Store as instance variable
            self.loading_message = f"Loading model [0.0s]"
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                use_safetensors=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=False
            )
            
            # Stop the loading animation
            self.is_loading = False
            model_load_time = time.time() - self.model_start_time
            print(f"\r✓ Loading model [{model_load_time:.1f}s]" + " " * 20)  # Clear spinner
            
            # Optimize model configuration
            self.model.config.use_cache = True
            self.model.config.return_dict = True
            self.model = self.model.half()
            
            # Configure generation settings
            generation_config = self.model.generation_config
            generation_config.max_new_tokens = 128
            generation_config.return_dict_in_generate = True
            generation_config.output_scores = False
            generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            generation_config.eos_token_id = self.processor.tokenizer.eos_token_id
            generation_config.num_beams = 1
            generation_config.do_sample = False
            generation_config.use_cache = True
            generation_config.return_dict = True
            generation_config.output_attentions = False
            generation_config.output_hidden_states = False
            generation_config.return_legacy_cache = False
            generation_config.language = "english"
            
            # Set model to eval mode
            self.model.eval()
            
            # Calculate total time
            total_time = time.time() - loading_start
            print(f"✓ Completed in {total_time:.1f}s\n")
            print("--- Press F20 to record ---\n")
            
        except Exception as e:
            print(f"\nModel loading error: {e}")
            sys.exit(1)
        
        self.transcription_thread = None
        self.streaming = False
        self.stream_buffer = []
        self.last_transcription = ""
        
        self.keyboard_listener.start()
        atexit.register(self.cleanup)

    def animate_loading(self):
        """Animate loading spinner with message"""
        while self.is_loading:
            for frame in SPINNER_FRAMES:
                if not self.is_loading:
                    break
                if "model" in self.loading_message.lower() and "[" in self.loading_message:
                    # Update time for model loading
                    current_time = time.time() - self.model_start_time
                    message = f"Loading model [{current_time:.1f}s]"
                else:
                    message = self.loading_message
                print(f"\r{frame} {message}", end='', flush=True)
                time.sleep(0.1)

    def display_loading_progress(self, step, total_steps, start_time, message):
        """Display loading progress with spinner, time, and percentage"""
        try:
            elapsed = time.time() - start_time
            percent = (step / total_steps) * 100
            spinner = SPINNER_FRAMES[self.spinner_idx % len(SPINNER_FRAMES)]
            print(f"\r{spinner} {message}: {elapsed:.1f}s [{percent:.0f}%]", end='', flush=True)
            self.spinner_idx += 1
        except Exception as e:
            print(f"\nProgress display error: {e}")

    def initialize_components(self):
        """Initialize all components with progress display"""
        start_time = time.time()
        total_steps = 4  # Total initialization steps
        current_step = 0
        
        # Step 1: Setup audio
        current_step += 1
        self.display_loading_progress(current_step, total_steps, start_time, "Setting up audio")
        self.setup_audio()
        
        # Step 2: Initialize device
        current_step += 1
        self.display_loading_progress(current_step, total_steps, start_time, "Initializing device")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float16
        time.sleep(0.5)  # Small delay to show progress
        
        try:
            # Step 3: Load processor
            current_step += 1
            self.display_loading_progress(current_step, total_steps, start_time, "Loading processor")
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                language="english",
                use_safetensors=True
            )
            
            # Step 4: Load and optimize model
            current_step += 1
            self.display_loading_progress(current_step, total_steps, start_time, "Loading model")
            
            # Use Accelerate for faster model loading and inference
            self.model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-large-v3-turbo",
                torch_dtype=torch.float16,
                use_safetensors=True,
                device_map="auto",  # Let Accelerate handle device placement
                low_cpu_mem_usage=True  # Re-enable for faster loading
            )
            
            # Optimize model configuration
            self.model.config.use_cache = True
            self.model.config.return_dict = True
            
            # Convert model inputs to fp16
            self.model = self.model.half()  # Ensure entire model is in fp16
            
            # Optimize generation config for speed
            generation_config = self.model.generation_config
            generation_config.max_new_tokens = 128  # Changed from 150 to 128
            generation_config.return_dict_in_generate = True
            generation_config.output_scores = False
            generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
            generation_config.eos_token_id = self.processor.tokenizer.eos_token_id
            generation_config.num_beams = 1
            generation_config.do_sample = False
            generation_config.use_cache = True
            generation_config.return_dict = True
            generation_config.output_attentions = False
            generation_config.output_hidden_states = False
            generation_config.return_legacy_cache = False
            generation_config.language = "english"
            
            # Set model to eval mode
            self.model.eval()
            
            # Clear the loading progress line
            print("\r" + " " * 80 + "\r", end='', flush=True)
            
        except Exception as e:
            print(f"\nModel loading error: {e}")
            sys.exit(1)
        
        self.transcription_thread = None
        self.streaming = False
        self.stream_buffer = []
        self.last_transcription = ""

    def setup_audio(self):
        try:
            self.audio = pyaudio.PyAudio()
            info = self.audio.get_host_api_info_by_index(0)
            
            self.input_device = info.get('defaultInputDevice')
            for i in range(0, info.get('deviceCount')):
                device_info = self.audio.get_device_info_by_index(i)
                if (device_info.get('maxInputChannels') > 0 and 
                    device_info.get('name') == 'ThomasPods'):
                    self.input_device = i
                    break
            
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=self.input_device,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            
        except Exception as e:
            print(f"\nAudio setup error: {e}")
            sys.exit(1)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        try:
            if self.recording:
                self.frames.append(in_data)
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)

    def toggle_recording(self):
        try:
            if not self.recording:
                if not self.stream.is_active():
                    self.stream.start_stream()
                print("\r[Recording...]", end='', flush=True)
                self.start_recording()
            else:
                print("\r[Processing...]", end='', flush=True)
                self.stop_recording()
        except Exception as e:
            print(f"\nRecording error: {e}")
            self.recording = False
            self.frames = []

    def start_recording(self):
        try:
            self.frames = []
            self.recording = True
            self.start_time = time.time()
            
            self.timer_thread = threading.Thread(target=self.display_timer, daemon=True)
            self.timer_thread.start()
            
        except Exception as e:
            print(f"\nStart recording error: {e}")
            self.recording = False
            self.frames = []

    def display_timer(self):
        try:
            self.spinner_idx = 0
            while self.recording and self.start_time is not None:  # Ensure start_time exists
                try:
                    elapsed = time.time() - self.start_time
                    spinner = SPINNER_FRAMES[self.spinner_idx % len(SPINNER_FRAMES)]
                    
                    # Check if we're approaching the time limit
                    if elapsed >= self.max_recording_time:
                        print("\rMaximum recording time reached (1 minute). Processing...", end='', flush=True)
                        self.stop_recording()  # Directly call stop_recording
                        break
                    elif elapsed >= self.max_recording_time - 5:  # Warning 5 seconds before limit
                        print(f"\r⚠️ {spinner} Recording: {elapsed:.1f}s (Time limit approaching)", end='', flush=True)
                    else:
                        print(f"\r{spinner} Recording: {elapsed:.1f}s", end='', flush=True)
                    
                    self.spinner_idx += 1
                    time.sleep(0.1)
                except TypeError:
                    print("\nDEBUG: Timer error - invalid time values")
                    break
            
            # Clear line
            print("\r", end='', flush=True)
            
        except Exception as e:
            print(f"\nDEBUG: Timer display error: {e}")
            self.recording = False

    def stop_recording(self):
        if not self.recording:
            return
        
        try:
            # Keep recording for a short moment to catch the last word
            time.sleep(0.2)
            
            processing_start = time.time()
            self.recording = False
            
            # Start processing timer in a new thread
            self.processing_timer = threading.Thread(target=self.display_processing_timer, args=(processing_start,), daemon=True)
            self.processing_timer.start()
            
            if self.frames:  # Only process if we have audio data
                self.process_audio(processing_start)
            else:
                print("\rNo audio detected")
            self.frames = []
            
        except Exception as e:
            self.frames = []
            self.recording = False
            self.is_processing = False

    def display_processing_timer(self, start_time):
        try:
            spinner_idx = 0
            self.is_processing = True
            while self.is_processing and start_time is not None:  # Ensure start_time exists
                try:
                    elapsed = time.time() - start_time
                    spinner = SPINNER_FRAMES[spinner_idx % len(SPINNER_FRAMES)]
                    print(f"\r{spinner} Processing: {elapsed:.1f}s", end='', flush=True)
                    spinner_idx += 1
                    time.sleep(0.1)
                except TypeError:
                    break
            print("\r", end='', flush=True)  # Clear line when done
        except Exception as e:
            self.is_processing = False

    def process_audio(self, processing_start):
        if not self.frames:
            self.is_processing = False
            print("\rNo audio detected")
            return
        
        try:
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.float32)
            
            if len(audio_data) == 0 or np.sqrt(np.mean(np.square(audio_data))) < 0.003:
                self.is_processing = False
                print("\rNo audio detected")
                return
            
            max_level = np.max(np.abs(audio_data))
            if max_level > 0:
                audio_data = audio_data / max_level
            
            with torch.inference_mode():
                # Process audio features and convert to fp16
                inputs = self.processor(
                    audio_data, 
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                )
                
                # Convert inputs to fp16 and move to device
                inputs.input_features = inputs.input_features.to(dtype=torch.float16, device=self.device)
                
                # Adjust padding for longer sequences
                if inputs.input_features.shape[-1] < 3000:
                    pad_length = 3000 - inputs.input_features.shape[-1]
                    inputs.input_features = torch.nn.functional.pad(
                        inputs.input_features,
                        (0, pad_length),
                        mode='constant',
                        value=0.0
                    )
                
                # Create attention mask in fp16
                attention_mask = torch.ones(
                    (1, inputs.input_features.shape[-1]),
                    dtype=torch.int64,
                    device=self.device
                )
                
                # Keep 128 tokens for faster processing
                outputs = self.model.generate(
                    inputs.input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    return_dict_in_generate=True,
                    output_scores=False,
                    forced_decoder_ids=[[0, 50359]],  # Use forced_decoder_ids instead of language
                    num_beams=1,
                    do_sample=False,
                    use_cache=True
                )
                
                text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
                
                # Add token count to output for analysis
                token_count = len(outputs.sequences[0])
                
                # Ensure proper sentence ending punctuation
                if text:
                    # Add period if the text doesn't end with punctuation
                    if not text[-1] in '.!?':
                        text += '.'
                    
                    # Capitalize first letter if it's not already
                    if len(text) > 0 and text[0].isalpha() and not text[0].isupper():
                        text = text[0].upper() + text[1:]
                    
                    self.is_processing = False
                    pyperclip.copy(text)
                    with self.keyboard_controller.pressed(Key.cmd):
                        self.keyboard_controller.press('v')
                        self.keyboard_controller.release('v')
                    
                    # Calculate times
                    recording_time = int(time.time() - self.start_time - 0.2)  # Full seconds
                    processing_time = time.time() - processing_start
                    
                    # Put icons before text: 🎤 for recording time, 📝 for tokens, ⚡ for processing
                    print(f"\r[🎤 {recording_time}s 📝 {token_count} ⚡ {processing_time:.1f}s] {text}\n")
                
        except Exception as e:
            self.is_processing = False
            print(f"\nTranscription error: {e}")

    def cleanup(self):
        if not hasattr(self, '_cleanup_done'):
            # Restore terminal input
            os.system('stty echo')
            
            self.recording = False
            self.streaming = False
            
            if hasattr(self, 'stream') and self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
            
            if hasattr(self, 'audio') and self.audio:
                try:
                    self.audio.terminate()
                except Exception:
                    pass
            
            self._cleanup_done = True

    def run(self):
        try:
            while True:
                if not self.keyboard_listener.is_alive():
                    self.keyboard_listener = keyboard.Listener(
                        darwin_intercept=True,
                        suppress=False,  # Don't suppress keyboard events
                        _intercept=True
                    )
                    self.keyboard_listener.start()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.cleanup()
            sys.exit(0)

if __name__ == "__main__":
    recorder = VoiceRecorder()
    
    # Check and warn about non-optimal settings
    warnings = []
    if recorder.device == "cpu":
        warnings.append("⚠️ Running on CPU - transcription will be slower. Enable MPS (Apple Silicon) for better performance.")
    if not torch.backends.mps.is_available():
        warnings.append("⚠️ MPS (Apple Silicon acceleration) is not available. Check that you're running macOS 12.3+ and have a compatible device.")
    
    if warnings:
        print("\nPerformance Warnings:")
        for warning in warnings:
            print(warning)
        print()  # Empty line after warnings
    
    recorder.run() 