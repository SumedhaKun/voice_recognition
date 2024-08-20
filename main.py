import record
import whisperx
from pyannote.audio import Pipeline
from pydub import AudioSegment
import process
import os
from dotenv import load_dotenv
load_dotenv()
import test

record.record_audio("diri")
model = whisperx.load_model('base',device="cpu",compute_type='float32')

TOKEN=os.getenv("MODEL_TOKEN")
audio = whisperx.load_audio("dir/diri.wav")

result = model.transcribe(audio, batch_size=16)
seg=result["segments"]

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu", return_char_alignments=False)


diarize_model=Pipeline.from_pretrained('pyannote/speaker-diarization-3.1',use_auth_token=TOKEN)
print(diarize_model)

diarize_segments = diarize_model("dir/diri.wav")

result = []
    
for segment, _, speaker in diarize_segments.itertracks(yield_label=True):
    print(segment)
    result.append({
        "start": segment.start,
        "end": segment.end,
        "speaker": speaker
    })

audio = AudioSegment.from_file("diri.wav")
i=0
files=[]
for item in result:
    segment = audio[item["start"]*1000:item["end"]*1000]
    segment.export("output"+str(i)+".wav", format="wav")
    files.append("output"+str(i))
    i+=1

matrices=process.process_files(files)
speakers=[]
for matrix in matrices:
    
    print("initial: ")
    print(matrix.shape)
    flat=matrix.flatten()[0:390]
    res=test.search(flat)
    speakers.append(res[-1][-1])
for k in range(len(speakers)):
    result=model.transcribe(whisperx.load_audio("output"+str(k)+".wav"))
    print(speakers[k]["entity"]["name"]+": "+result["segments"][0]["text"])


