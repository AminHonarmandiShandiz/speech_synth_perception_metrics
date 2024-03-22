import numpy as np
import librosa
import librosa.display
import matplotlib
import soundfile as sf
# Use Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load audio
audio_file = 'media/speech_01.wav'
y, sr = librosa.load(audio_file)

# Compute spectogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

# Convert to decibels
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.savefig('mel_spectrogram.png')  # Save the plot as an image
plt.close()  # Close the plot to avoid backend errors

# Reconstruct audio
y_reconstructed = librosa.feature.inverse.mel_to_audio(mel_spectrogram)

# Save
output_audio_file = 'media/reconstructed_audio.wav'
sf.write(output_audio_file, y_reconstructed, sr)
