import numpy as np
from scipy.io import wavfile

# Generate a beep sound
sample_rate = 44100  # CD quality audio
duration = 0.2  # seconds
frequency = 1000  # Hz
t = np.linspace(0, duration, int(sample_rate * duration), False)
beep = np.sin(2 * np.pi * frequency * t)

# Normalize and convert to 16-bit integer
beep = np.int16(beep * 32767)

# Save as WAV file
wavfile.write('static/sounds/beep.wav', sample_rate, beep) 