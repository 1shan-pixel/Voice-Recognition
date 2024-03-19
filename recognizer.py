from pydub import AudioSegment 

try:
    sound = AudioSegment.from_file("recordings/Recording.m4a", format="m4a")
    sound.export("recordings/Output.mp3" ,format="mp3")
except FileNotFoundError:
    print("Error: Recording.m4a not found. Please check the filename and its location.")