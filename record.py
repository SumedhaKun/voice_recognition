import pyaudio
import wave
import os
FORMAT = pyaudio.paInt16
CHANNELS = 1               
RATE = 44100                
CHUNK = 1024 
def record_audio(file_name):
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
    frames = []
    print("Started recording.")
    for i in range(int(RATE / CHUNK * 20)):
            data = stream.read(CHUNK)
            frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    if not os.path.exists(file_name[0:len(file_name)-1]):
        os.makedirs(file_name[0:len(file_name)-1])
    
    # Construct the full file path
    file_path = os.path.join(file_name[0:len(file_name)-1], file_name + ".wav")
    file="/"+file_name+".wav"
    print(file)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
