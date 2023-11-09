import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
import os

# Define a function to convert frequency to musical note


def frequency_to_note(frequency):
    A4_freq = 440
    C0_freq = A4_freq * np.power(2, -4.75)
    note_names = ['C', 'C#', 'D', 'D#', 'E',
                  'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    if frequency == 0:
        return None  # No sound or error

    # Calculate the note number
    h = round(12 * np.log2(frequency / C0_freq))
    octave = h // 12
    n = h % 12
    return note_names[n] + str(octave)

# Function to detect the note of a single WAV file


def detect_note_in_wav(file_path):
    # Read the wav file
    sample_rate, data = wavfile.read(file_path)

    # Ensure we're working with a mono signal
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Apply FFT and get frequencies
    fft_spectrum = np.fft.fft(data)
    freq = np.fft.fftfreq(len(fft_spectrum), 1 / sample_rate)

    # Find the peak frequency
    y = np.abs(fft_spectrum)
    peaks, _ = find_peaks(y, height=np.max(y)/4)
    peak_freq = freq[peaks][np.argmax(y[peaks])]

    # Convert peak frequency to musical note
    note = frequency_to_note(abs(peak_freq))
    return note

# Function to process all WAV files in the directory


def process_wav_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            note = detect_note_in_wav(file_path)
            print(f"File: {filename}, Note: {note}")


# Assuming your wav files are stored in a folder named 'wav_files'
process_wav_files('wav_files')
