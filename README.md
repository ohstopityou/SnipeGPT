# Voice-to-ChatGPT

This application allows you to trigger voice recording using a Logitech mouse button, transcribe the speech using Whisper locally, and send it to ChatGPT for a response.

## Requirements

- Python 3.7+
- FFmpeg (required by Whisper for audio processing)
- macOS system
- Apple Silicon GPU (optional, for faster processing)

## Why FFmpeg is Required

FFmpeg is a critical dependency because Whisper uses it internally for audio processing:
1. Converting between different audio formats
2. Resampling audio to 16kHz (required by Whisper's model)
3. Converting stereo to mono channel
4. Normalizing audio data

## Setup

1. Install FFmpeg (required):
```bash
# On macOS using Homebrew
brew install ffmpeg

# If you don't have Homebrew installed:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

2. Set up a Python virtual environment (recommended):
```bash
# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Your prompt should now show (venv) at the beginning
# All further commands should be run with the virtual environment activated
```

3. Install the required dependencies:
```bash
# Make sure your virtual environment is activated (you should see (venv) in your prompt)
python3 -m pip install --upgrade pip  # Upgrade pip first
python3 -m pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key (only needed for ChatGPT, not for Whisper):
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Activate the virtual environment (if not already activated):
```bash
source venv/bin/activate
```

2. Run the main script:
```bash
python3 main.py
```

3. Wait for the Whisper model to load (this may take a few moments)
4. Press F20 to start recording
5. Speak your message
6. Press F20 again to stop recording and process the audio
7. The transcribed text will be displayed

## Features

- Local speech recognition using Whisper (no API calls needed for transcription)
- Supports multiple languages automatically
- Uses the "base" Whisper model by default (can be changed to "tiny", "small", "medium", or "large")
- Real-time audio recording with keyboard control (F20 key)
- GPU acceleration on Apple Silicon (if available)
- Automatic benchmarking of different Whisper model sizes on first run

## Troubleshooting

1. If you see "command not found: pip", make sure you're using the correct Python installation:
```bash
# Check Python version and location
which python3
python3 --version

# Install pip if needed
python3 -m ensurepip --upgrade
```

2. If you have permission errors installing packages:
```bash
# Use the --user flag
python3 -m pip install --user -r requirements.txt
```

3. To deactivate the virtual environment when you're done:
```bash
deactivate
```

## Requirements

- Python 3.7+
- FFmpeg
- CUDA-capable GPU (optional, for faster transcription)
- Logitech mouse
- OpenAI API key (for ChatGPT only)
- macOS system 