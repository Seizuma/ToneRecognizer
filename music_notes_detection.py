import numpy as np
import io
import wave
import os
import struct


def read_and_convert_24bit_wave(filename):
    # Open the original WAV file
    with wave.open(filename, 'rb') as wav:
        num_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        framerate = wav.getframerate()
        num_frames = wav.getnframes()
        comp_type = wav.getcomptype()
        comp_name = wav.getcompname()
        original_data = wav.readframes(num_frames)

    # Convert the 24-bit data to 16-bit
    data = bytearray(original_data)
    converted_data = bytearray()
    for i in range(0, len(data), 3):
        # Skip the least significant byte
        converted_data.append(data[i+1])
        converted_data.append(data[i+2])

    # Create an in-memory buffer with the converted data
    mem_wave = io.BytesIO()
    # Create a new Wave_write object to write the converted data
    with wave.open(mem_wave, 'wb') as mem_wav:
        mem_wav.setnchannels(num_channels)
        mem_wav.setsampwidth(2)  # 16-bit
        mem_wav.setframerate(framerate)
        mem_wav.setnframes(num_frames)
        mem_wav.setcomptype(comp_type, comp_name)
        mem_wav.writeframes(bytes(converted_data))
    # Move back to the beginning of the buffer
    mem_wave.seek(0)

    # Return a new Wave_read object from the in-memory buffer
    return wave.open(mem_wave, 'rb')


def note_detect(audio_file):
    file_length = audio_file.getnframes()
    f_s = audio_file.getframerate()  # sampling frequency
    channels = audio_file.getnchannels()  # number of channels mono/stereo
    sampwidth = audio_file.getsampwidth()  # sample width in bytes
    sound = np.zeros(file_length)

    for i in range(file_length):
        # Read the correct amount of data based on the number of channels and sample width
        wdata = audio_file.readframes(1)
        # Check if we have reached the end of the file
        if len(wdata) < channels * sampwidth:
            break
        if channels == 2:  # Stereo
            data = struct.unpack("<hh", wdata)
            # Average the two channels
            sound[i] = int((data[0] + data[1]) / 2)
        else:  # Mono
            data = struct.unpack("<h", wdata)
            sound[i] = int(data[0])

    sound = np.divide(sound, float(2**15))  # scaling it to 0 - 1

    # Fourier transformation
    fourier = np.fft.fft(sound)
    fourier = np.absolute(fourier)
    imax = np.argmax(fourier[0:int(file_length/2)])  # index of max element

    # Convert index to frequency
    freq = (imax*f_s)/(file_length*channels)

    # Frequency database
    name = np.array(["C0", "C#0", "D0", "D#0", "E0", "F0", "F#0", "G0", "G#0", "A0", "A#0", "B0", "C1", "C#1", "D1", "D#1", "E1", "F1", "F#1", "G1", "G#1", "A1", "A#1", "B1", "C2", "C#2", "D2", "D#2", "E2", "F2", "F#2", "G2", "G2#", "A2", "A2#", "B2", "C3", "C3#", "D3", "D3#", "E3", "F3", "F3#", "G3", "G3#", "A3", "A3#", "B3", "C4", "C4#", "D4", "D4#", "E4", "F4",
                    "F4#", "G4", "G4#", "A4", "A4#", "B4", "C5", "C5#", "D5", "D5#", "E5", "F5", "F5#", "G5", "G5#", "A5", "A5#", "B5", "C6", "C6#", "D6", "D6#", "E6", "F6", "F6#", "G6", "G6#", "A6", "A6#", "B6", "C7", "C7#", "D7", "D7#", "E7", "F7", "F7#", "G7", "G7#", "A7", "A7#", "B7", "C8", "C8#", "D8", "D8#", "E8", "F8", "F8#", "G8", "G8#", "A8", "A8#", "B8", "Beyond B8"])
    frequencies = np.array([16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87, 32.70, 34.65, 36.71, 38.89, 41.20, 43.65, 46.25, 49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30,
                           440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53, 2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07, 4186.01, 4434.92, 4698.64, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040.00, 7458.62, 7902.13, 8000])

   # Searching for matched frequencies
    note = "None"
    for i in range(0, frequencies.size-1):
        if (freq < frequencies[0]):
            note = name[0]
            break
        if (freq > frequencies[-1]):
            note = name[-1]
            break
        if freq >= frequencies[i] and frequencies[i+1] >= freq:
            if freq-frequencies[i] < (frequencies[i+1]-frequencies[i])/2:
                note = name[i]
            else:
                note = name[i+1]
            break

    return note


if __name__ == "__main__":
    audio_file_path = './wav_files/LipRollB.wav'
    # First, convert the file
    converted_audio = read_and_convert_24bit_wave(audio_file_path)
    # Then, pass the converted audio to your note detection function
    Detected_Note = note_detect(converted_audio)
    print("\n\tDetected Note = " + str(Detected_Note))
