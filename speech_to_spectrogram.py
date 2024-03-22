import matplotlib
matplotlib.use('Agg')  # Use Agg backend explicitly
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# Load audio
audio_file = 'media/speech_01.wav'
y, sr = librosa.load(audio_file)

# Compute spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibels
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot the mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()

# Save
plt.savefig('media/speech_01.png')
plt.close()
