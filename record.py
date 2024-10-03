import pyaudio
import wave
import os
from pydub import AudioSegment
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
    for i in range(int(RATE / CHUNK * 21)):
            data = stream.read(CHUNK)
            frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
        
    file_path =file_name+".wav"
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))


def clip_recordings(file_name,i=0):
    limit=5000
    audio = AudioSegment.from_file(file_name+".wav")
    if len(audio)<limit:
        return []
    tmp=audio
    files=[]
    while len(tmp)>limit:
        file=tmp[0:limit]
        name="clip"+str(i)+".wav"
        file.export(name,format="wav")
        files.append(name)
        tmp=tmp[limit:]
        i+=1
    return files

    




    
    
