import whisper
import os
import sys
import argparse
import subprocess

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace('.', ',')

def generate_srt(segments, srt_path):
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    print(f"SRT file generated at: {srt_path}")

def burn_subtitles(video_path, srt_path, output_video_path):
    style = (
        "FontName=Arial,"
        "FontSize=24,"
        "PrimaryColour=&H0000A5FF,"
        "Bold=1,"
        "BorderStyle=1,"
        "Outline=1,"
        "Shadow=0,"
        "MarginV=50"
    )
    subtitles_filter = f"subtitles='{srt_path}':force_style='{style}'"
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", subtitles_filter,
        "-c:a", "copy",
        output_video_path
    ]
    print("Burning customized subtitles into video...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Stylish subtitled video saved as: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print("Error burning subtitles:", e)

def transcribe_media(model, file_path, language=None):
    print(f"Transcribing with word-level timestamps: {file_path}")
    try:
        result = model.transcribe(file_path, language=language, word_timestamps=True)
        return result.get("segments", [])
    except Exception as e:
        print(f"Error during transcription: {e}")
        return []


def split_into_short_segments(segments, max_duration=1.5):
    """
    Splits subtitles dynamically based on **time**, not word count.

    - max_duration (seconds): Adjust this to control subtitle length.
    - If a word group exceeds this duration, a new subtitle block starts.

    """
    short_segments = []
    current_segment = []
    segment_start = None

    for segment in segments:
        for word_info in segment.get('words', []):
            if not current_segment:
                segment_start = word_info['start']

            current_segment.append(word_info)

            # Calculate total duration so far
            segment_duration = word_info['end'] - segment_start

            if segment_duration >= max_duration:
                short_segments.append({
                    'start': segment_start,
                    'end': word_info['end'],
                    'text': ' '.join(w['word'].strip() for w in current_segment)
                })
                current_segment = []
                segment_start = None  # Reset start time

    # Add any remaining words as the final subtitle block
    if current_segment:
        short_segments.append({
            'start': current_segment[0]['start'],
            'end': current_segment[-1]['end'],
            'text': ' '.join(w['word'].strip() for w in current_segment)
        })

    return short_segments

def main():
    parser = argparse.ArgumentParser(description="Transcribe video with dynamic stylish subtitles.")
    parser.add_argument("media_file", help="Path to audio or video file")
    parser.add_argument("--model", type=str, default="large", help="Whisper model (tiny, base, small, medium, large)")
    parser.add_argument("--language", type=str, help="Spoken language")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--burn", action="store_true", help="Burn subtitles into video")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = whisper.load_model(args.model)

    if not os.path.exists(args.media_file):
        print("File not found:", args.media_file)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    segments = transcribe_media(model, args.media_file, args.language)
    if not segments:
        print("No segments found.")
        sys.exit(1)

    segments = split_into_short_segments(segments, max_duration=1.5)

    base_name = os.path.splitext(os.path.basename(args.media_file))[0]
    srt_path = os.path.join(args.output_dir, f"{base_name}.srt")
    generate_srt(segments, srt_path)

    if args.burn:
        video_extensions = {".mp4", ".mov", ".mkv", ".avi"}
        ext = os.path.splitext(args.media_file)[1].lower()
        if ext not in video_extensions:
            print("Burn option only works with video files.")
        else:
            output_video = os.path.join(args.output_dir, f"{base_name}_subtitled{ext}")
            burn_subtitles(args.media_file, srt_path, output_video)

if __name__ == "__main__":
    main()
