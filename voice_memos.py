import whisper
import os
import sys

# Load the model (large, medium, small, or tiny depending on your performance needs)
model = whisper.load_model("large")


def transcribe_audio(file_path):
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    print("Transcribing:", file_path)
    result = model.transcribe(file_path)

    # Print and save the transcription
    text_output = file_path.replace(".mp3", ".txt").replace(".wav", ".txt").replace(".m4a", ".txt")
    with open(text_output, "w", encoding="utf-8") as f:
        f.write(result["text"])

    print("Transcription saved to:", text_output)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file>")
    else:
        transcribe_audio(sys.argv[1])