import whisper

model = whisper.load_model("tiny.en")
result = model.transcribe("recordings/Output.mp3")
print(result["text"])