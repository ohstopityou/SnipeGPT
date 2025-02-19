import pyaudio
import numpy as np
import time

def test_microphone():
    print("\n=== Microphone Test ===")
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    try:
        # List all audio devices
        print("\nAvailable Audio Devices:")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        input_devices = []
        for i in range(0, numdevices):
            device_info = audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                input_devices.append((i, device_info))
                print(f"\nDevice {i}:")
                print(f"  Name: {device_info.get('name')}")
                print(f"  Default Sample Rate: {int(device_info.get('defaultSampleRate'))} Hz")
                print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
                print(f"  Default Input: {'Yes' if info.get('defaultInputDevice') == i else 'No'}")
        
        if not input_devices:
            print("\nNo input devices found!")
            return
        
        # Test each input device
        for device_id, device_info in input_devices:
            print(f"\n=== Testing Device: {device_info.get('name')} ===")
            
            # Try different sample rates
            for rate in [16000, 44100, 48000]:
                print(f"\nTrying sample rate: {rate} Hz")
                
                try:
                    stream = audio.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=1024
                    )
                    
                    print("Recording for 3 seconds...")
                    levels = []
                    start_time = time.time()
                    
                    while time.time() - start_time < 3:
                        try:
                            data = stream.read(1024, exception_on_overflow=False)
                            audio_data = np.frombuffer(data, dtype=np.float32)
                            level = np.max(np.abs(audio_data))
                            levels.append(level)
                            print(f"Current level: {level:.4f}", end='\r')
                        except Exception as e:
                            print(f"Error reading: {e}")
                            break
                    
                    print(f"\nMax level: {max(levels):.4f}")
                    print(f"Average level: {np.mean(levels):.4f}")
                    
                    stream.stop_stream()
                    stream.close()
                    
                except Exception as e:
                    print(f"Error with sample rate {rate} Hz: {e}")
                    continue
                
                print(f"Successfully tested at {rate} Hz")
            
            print("\n---")
    
    finally:
        audio.terminate()

if __name__ == "__main__":
    test_microphone() 