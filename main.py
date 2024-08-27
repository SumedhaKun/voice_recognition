import record
import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment
import process
import os
from dotenv import load_dotenv
load_dotenv()
import test
from pymongo import MongoClient

URI=os.getenv("MONGO_URL")
mongodb_client = MongoClient(URI)
database = mongodb_client["VoiceRec"]
collection = database["voices"]

record.record_audio("test1")
file="test/test1.wav"

model = whisperx.load_model('base',device="cpu",compute_type='float32')
TOKEN=os.getenv("MODEL_TOKEN")
audio = whisperx.load_audio(file)

result = model.transcribe(audio, batch_size=16)
seg=result["segments"]

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu", return_char_alignments=False)


diarize_model=Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token=TOKEN)

diarize_segments = diarize_model(file)

result = []
    
for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
    print(segment)
    result.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

audio = AudioSegment.from_file(file)
i=0
files=[]
for item in result:
    segment = audio[item["start"]*1000:item["end"]*1000]
    segment.export("output"+str(i)+".wav", format="wav")
    files.append("output"+str(i))
    i+=1
matrices=[]
for file in files:
    matrices.append(process.process_file(file))

speakers=[]
for matrix in matrices:
    print(matrix.shape)
    res=test.search(matrix)
    speakers.append(res)
for k in range(len(speakers)):
    result=model.transcribe(whisperx.load_audio("output"+str(k)+".wav"))
    print(speakers[k]+": "+result["segments"][0]["text"])

feedback=input("Was this correct? (Y/N) ")
if feedback=="Y":
    for i in range(len(matrices)):
        obj={"matrix":matrices[i].tolist(), "name":speakers[i]}
        collection.insert_one(obj)
if feedback=="N":
    wrongs=input("Which ones are wrong? (0-index, comma spaces) ")
    wrongs=wrongs.split(",")
    for idx in wrongs:
        name=input("Who is correct owner for "+idx+"? ")
        idx=int(idx)
        obj={"matrix":matrices[idx].tolist(), "name":name}
        print(obj)
        collection.insert_one(obj)
    for i in range(len(matrices)):
        if str(i) not in wrongs:
            obj={"matrix":matrices[i].tolist(), "name":speakers[i]}
            collection.insert_one(obj)



