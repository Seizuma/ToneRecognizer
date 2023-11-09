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

# Function to find harmonies (up to two notes) in a single WAV file


def find_harmonies(freq, fft_spectrum, sample_rate):
    # Find peaks in the FFT magnitude
    magnitude = np.abs(fft_spectrum)
    peaks, _ = find_peaks(magnitude, height=np.max(magnitude)/10, distance=20)

    # Sort peaks by magnitude
    peaks = sorted(peaks, key=lambda peak: magnitude[peak], reverse=True)

    # Get frequencies for the peaks
    peak_freqs = freq[peaks]

    # Convert to notes
    # Taking the first two peaks as the harmonies
    notes = [frequency_to_note(abs(f)) for f in peak_freqs[:2]]

    return notes

# Modified function to detect the note or harmonies of a single WAV file


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

    # Detect harmonies (up to two notes)
    notes = find_harmonies(freq, fft_spectrum, sample_rate)

    # Convert peak frequency to musical note
    main_note = frequency_to_note(abs(peak_freq))

    # If we have more than one note detected, return them as harmonies
    if len(notes) > 1 and main_note not in notes:
        return notes
    else:
        return [main_note]

# Function to process all WAV files in the directory and detect notes or harmonies


def process_wav_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            notes = detect_note_in_wav(file_path)
            print(f"File: {filename}, Notes: {notes}")


# Assuming your wav files are stored in a folder named 'wav_files'
process_wav_files('wav_files')
