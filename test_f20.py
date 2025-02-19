from pynput import keyboard
from pynput._util.darwin import ListenerMixin
import time

print("\n=== F20 Key Test ===")
print("Press F20 or ESC to exit")

# Track the last key event to prevent duplicates
last_event_time = 0
DEBOUNCE_TIME = 0.1  # 100ms debounce

def custom_handle(self, proxy, event_type, event, refcon):
    """Custom handle method that avoids thread issues"""
    global last_event_time
    
    try:
        key = self._event_to_key(event)
        if key is not None:
            current_time = time.time()
            
            # Debug information
            print(f"\nDEBUG - Event type: {event_type}")
            if hasattr(key, 'vk'):
                print(f"DEBUG - Virtual key code: {key.vk}")
            print(f"DEBUG - Key object: {key}")
            
            # Only process keydown events (type 10) and debounce
            if event_type == 10 and (current_time - last_event_time) > DEBOUNCE_TIME:
                last_event_time = current_time
                
                # Check for F20 (virtual key 65283)
                if hasattr(key, 'vk') and key.vk == 65283:
                    print("\nF20 KEY PRESSED!")
                    # Don't pass this to the regular handler
                    return event
                
                # Handle other keys through the regular handler
                if self.on_press:
                    self.on_press(key)
                    
        return event
    except Exception as e:
        print(f"Error in custom_handle: {e}")
        return event

# Replace the handler
ListenerMixin._handler = custom_handle

def on_press(key):
    """Handle key press"""
    try:
        if key == keyboard.Key.esc:
            print("\nESC pressed - Exiting...")
            return False
            
        # Only handle non-F20 keys here
        if not (hasattr(key, 'vk') and key.vk == 65283):
            if hasattr(key, 'char'):
                print(f"Key pressed: {key.char}")
            else:
                print(f"Key pressed: {key}")
            
        return True
    except Exception as e:
        print(f"Press error: {e}")
        return True

# Create and start listener
listener = keyboard.Listener(
    on_press=on_press,
    darwin_intercept=True,
    suppress=False,
    _intercept=True
)

print("\nStarting listener...")
listener.start()
print("Ready!")

try:
    while listener.running:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped") 