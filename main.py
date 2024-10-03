import record
import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment
import process
import os
from collections import Counter

from dotenv import load_dotenv
load_dotenv()
import test
from pymongo import MongoClient

URI=os.getenv("MONGO_URL")
mongodb_client = MongoClient(URI)
database = mongodb_client["VoiceRec"]
collection = database["voices"]

# records audio and stores in file 'test.wav'
file=record.record_audio("test")
device="cuda"
model = whisperx.load_model('base',device=device,compute_type='float32')
TOKEN=os.getenv("MODEL_TOKEN")
audio = whisperx.load_audio(file)

# transcribes audio
result = model.transcribe(audio, batch_size=16)
seg=result["segments"]

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


diarize_model=Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token=TOKEN)
# diarize the audio file
diarize_segments = diarize_model(file)

result = []
    
for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
    result.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

# combine clips with same voice identifier if adjacent
res_stack=[]
for item in result:
    if not res_stack:
        res_stack.append(item)
        continue
    if item["speaker"]==res_stack[-1]["speaker"]:
        last=res_stack.pop()
        last["end"]=item["end"]
        res_stack.append(last)
    else:
        res_stack.append(item)

audio = AudioSegment.from_file(file)
i=0
new_files=[]
# Stores 5s clips of all diarized pieces
for item in res_stack:
    segment = audio[item["start"]*1000:item["end"]*1000+1]
    segment.export("output.wav", format="wav")
    clips=record.clip_recordings("output",i)
    if clips:
        new_files.append(clips)
    i+=len(clips)

speaker_res=[]
matrices=[]
for i in range(len(new_files)):
    speaker=[]
    transcription=[]
    for j in range(len(new_files[i])):
        matrix=process.process_file(new_files[i][j])
        speaker.append(test.search(matrix))
        t=model.transcribe(whisperx.load_audio(new_files[i][j]))["segments"][0]["text"]
        transcription.append(t)
    counter=Counter(speaker)
    spkr=counter.most_common(1)[0][0]
    speaker_res.append(spkr)
    print(spkr+":"+''.join(transcription))







