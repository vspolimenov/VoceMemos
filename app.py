import streamlit as st
import os
import tempfile
import subprocess
import whisper
from datetime import datetime
import time
import math
import pandas as pd
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import av
import queue
import numpy as np
import soundfile as sf

#########################
# Utility Functions
#########################

def format_timestamp(seconds):
    """Format a float `seconds` into SRT-compatible HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - (hours * 3600) - (minutes * 60)
    return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace('.', ',')

def split_into_short_segments(segments, max_duration=1.5):
    """Split subtitles into short chunks by duration."""
    short_segments = []
    current_words = []
    segment_start = None

    for seg in segments:
        for word_info in seg.get('words', []):
            if not current_words:
                segment_start = word_info['start']
            current_words.append(word_info)

            segment_duration = word_info['end'] - segment_start
            if segment_duration >= max_duration:
                short_segments.append({
                    'start': segment_start,
                    'end': word_info['end'],
                    'text': ' '.join(w['word'].strip() for w in current_words)
                })
                current_words = []
                segment_start = None

    if current_words:
        short_segments.append({
            'start': current_words[0]['start'],
            'end': current_words[-1]['end'],
            'text': ' '.join(w['word'].strip() for w in current_words)
        })

    return short_segments

def transcribe_media(model_name, file_path, language=None, max_duration=1.5, srt_mode=False):
    """
    Use Whisper to transcribe the file.
    - If srt_mode=False, return plain text without timestamps.
    - If srt_mode=True, return segments with timestamps for SRT.
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, language=language, word_timestamps=srt_mode)
    if not srt_mode:
        return result.get("text", "").strip()
    else:
        segments = result.get("segments", [])
        short_segments = split_into_short_segments(segments, max_duration=max_duration)
        return short_segments

def rgb_to_bbggrr(hex_color):
    """Convert a #RRGGBB color to &H00BBGGRR format for FFmpeg."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return "&H00FFFFFF"  # Fallback to white
    bb = hex_color[4:6]
    gg = hex_color[2:4]
    rr = hex_color[0:2]
    return f"&H00{bb}{gg}{rr}"

def get_style(template_name, user_color, font_size, margin):
    """Return a style string for FFmpeg based on the chosen template."""
    ffmpeg_color = rgb_to_bbggrr(user_color)
    if template_name == "Classic":
        bold_val = 0
        outline_val = 1
    elif template_name == "Bold Outline":
        bold_val = 1
        outline_val = 3
    elif template_name == "Minimalist":
        bold_val = 0
        outline_val = 1
    else:  # "Custom" fallback
        bold_val = 1
        outline_val = 1

    style = (
        "FontName=Arial,"
        f"FontSize={font_size},"
        f"PrimaryColour={ffmpeg_color},"
        f"Bold={bold_val},"
        "BorderStyle=1,"
        f"Outline={outline_val},"
        "Shadow=0,"
        f"MarginV={margin}"
    )
    return style

def dataframe_to_segments(df):
    """Convert edited DataFrame back to segments, skipping invalid entries."""
    new_segs = []
    for _, row in df.iterrows():
        try:
            start = float(row["Start (sec)"])
            end = float(row["End (sec)"])
        except (ValueError, TypeError):
            continue

        if (math.isnan(start) or math.isnan(end) or
                start < 0 or end <= 0 or start >= end):
            continue

        text = str(row["Text"]).strip()
        if not text:
            continue

        new_segs.append({"start": start, "end": end, "text": text})
    return new_segs

def generate_srt(segments, srt_path):
    """Write segments to an SRT file."""
    segments = sorted(segments, key=lambda x: x['start'])
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['text'].strip()
            if not text:
                continue
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    return srt_path

def burn_subtitles(video_path, srt_path, output_video_path, style_str):
    """Burn subtitles into the video using FFmpeg."""
    subtitles_filter = f"subtitles='{srt_path}':force_style='{style_str}'"
    cmd = ["ffmpeg", "-i", video_path, "-vf", subtitles_filter, "-c:a", "copy", output_video_path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.success("Subtitles burned successfully!")
        st.write("FFmpeg output:", proc.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"FFmpeg error: {e.stderr}")
        raise

def generate_preview_frame(video_path, text, style_str, output_image_path, timestamp=5.0):
    """Generate a single frame preview with text overlay using FFmpeg."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
        tmp_srt.write(f"1\n00:00:00,000 --> 00:00:10,000\n{text}".encode('utf-8'))
        tmp_srt_path = tmp_srt.name

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vf", f"subtitles='{tmp_srt_path}':force_style='{style_str}'",
        "-frames:v", "1",
        "-y",
        output_image_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.remove(tmp_srt_path)
        return output_image_path
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to generate preview: {e.stderr}")
        return None

#########################
# Streamlit App
#########################

def main():
    st.title("Video Transcription with Editable Subtitles or Plain Text")

    # Output mode selection
    st.subheader("Transcription Output Mode")
    output_mode = st.radio("Choose an output type:", ["Just Text", "SRT + Optional Burn"])

    # Input method selection for Just Text mode
    if output_mode == "Just Text":
        st.subheader("Input Method")
        input_method = st.radio("Choose input method:", ["Upload File", "Record Voice"])
    else:
        input_method = "Upload File"  # Default for SRT mode

    # Handle input based on method
    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload a video/audio file",
            type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"]
        )
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state["tmp_media_path"] = tmp_file.name
                st.session_state["is_video"] = uploaded_file.type.startswith("video")
        else:
            st.session_state["tmp_media_path"] = None
            st.session_state["is_video"] = False

    elif input_method == "Record Voice":
        st.write("Record your voice for transcription:")

        class AudioProcessor(AudioProcessorBase):
            def __init__(self):
                self.audio_queue = queue.Queue()

            def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                self.audio_queue.put(frame)
                return frame

        ctx = webrtc_streamer(
            key="audio-recorder",
            audio_processor_factory=AudioProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"audio": True, "video": False},
        )

        if ctx.state.playing:
            st.write("Recording... Press stop when done.")

        if ctx.state.playing == False and not ctx.state.playing:
            audio_frames = []
            while ctx.audio_processor and not ctx.audio_processor.audio_queue.empty():
                frame = ctx.audio_processor.audio_queue.get()
                audio_frames.append(frame.to_ndarray())

            if audio_frames:
                audio_data = np.concatenate(audio_frames)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    sf.write(tmp_file.name, audio_data, 16000)  # 16kHz sample rate
                    st.session_state["tmp_media_path"] = tmp_file.name
                    st.session_state["is_video"] = False  # Audio only
            else:
                st.session_state["tmp_media_path"] = None
                st.session_state["is_video"] = False

    # Subtitle options (only for SRT mode)
    if output_mode == "SRT + Optional Burn":
        st.subheader("Subtitle Segmentation")
        max_subtitle_duration = st.slider(
            "Max duration (seconds) per segment",
            min_value=0.5, max_value=5.0, value=1.5, step=0.1
        )
        st.subheader("Subtitle Styling Options")
        template = st.selectbox("Style template", ["Classic", "Bold Outline", "Minimalist", "Custom"])
        chosen_color = st.color_picker("Subtitle Color", value="#FFA500")
        chosen_font_size = st.number_input("Font Size", min_value=12, max_value=60, value=24)
        chosen_margin = st.number_input("Margin from bottom", min_value=10, max_value=200, value=50)
        burn_it = st.checkbox("Burn subtitles into video (for video files only)")
    else:
        # Style options for preview in Just Text mode
        st.subheader("Text Style Preview Options (Optional)")
        template = "Classic"  # Default for preview
        chosen_color = st.color_picker("Text Color", value="#FFFFFF")
        chosen_font_size = st.number_input("Font Size", min_value=12, max_value=60, value=24)
        chosen_margin = st.number_input("Margin from Bottom", min_value=10, max_value=200, value=50)
        max_subtitle_duration = 1.5  # Not used in Just Text
        burn_it = False

    # Model and language selection
    st.subheader("Transcription Model & Language")
    model_name = st.selectbox("Whisper Model", ["tiny", "base", "small", "medium", "large"])
    lang_dict = whisper.tokenizer.LANGUAGES
    lang_map = {v.title(): k for k, v in lang_dict.items()}
    selected_language = st.selectbox("Language", ["Auto Detect"] + sorted(lang_map.keys()))
    language_code = None if selected_language == "Auto Detect" else lang_map[selected_language]

    # Transcription button
    if st.session_state.get("tmp_media_path") and st.button("Transcribe"):
        st.info("Transcribing...")
        start_time = time.perf_counter()
        try:
            if output_mode == "Just Text":
                transcript_text = transcribe_media(
                    model_name,
                    st.session_state["tmp_media_path"],
                    language=language_code,
                    srt_mode=False
                )
                if not transcript_text.strip():
                    st.error("No text found or empty file.")
                    return
                st.session_state["transcript_text"] = transcript_text
            else:
                segments = transcribe_media(
                    model_name,
                    st.session_state["tmp_media_path"],
                    language=language_code,
                    max_duration=max_subtitle_duration,
                    srt_mode=True
                )
                if not segments:
                    st.error("No segments found.")
                    return
                st.session_state["segments"] = segments
            elapsed = time.perf_counter() - start_time
            st.success(f"Transcription completed in {elapsed:.2f} seconds!")
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return

    # Display results based on mode
    if output_mode == "Just Text" and "transcript_text" in st.session_state:
        st.subheader("Transcribed Text")
        st.text_area("Text", value=st.session_state["transcript_text"], height=200)
        text_file_path = os.path.join(
            tempfile.gettempdir(),
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(text_file_path, "w", encoding="utf-8") as tf:
            tf.write(st.session_state["transcript_text"])
        with open(text_file_path, "rb") as tf:
            st.download_button(
                "Download Transcript as .txt",
                data=tf,
                file_name="transcript.txt",
                mime="text/plain"
            )

        # Preview option (only if a video was uploaded)
        if st.session_state.get("is_video", False):
            if st.button("Preview Text Style on Video"):
                style_str = get_style(template, chosen_color, chosen_font_size, chosen_margin)
                preview_image_path = os.path.join(
                    tempfile.gettempdir(),
                    f"preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                generate_preview_frame(
                    st.session_state["tmp_media_path"],
                    st.session_state["transcript_text"][:50],  # First 50 chars for preview
                    style_str,
                    preview_image_path,
                    timestamp=5.0
                )
                if os.path.exists(preview_image_path):
                    st.image(preview_image_path, caption="Text Style Preview", use_column_width=True)
                else:
                    st.error("Failed to generate preview.")

    elif output_mode == "SRT + Optional Burn" and "segments" in st.session_state and st.session_state["segments"]:
        # Display and edit segments
        df = pd.DataFrame(
            st.session_state["segments"],
            columns=["start", "end", "text"]
        ).rename(columns={"start": "Start (sec)", "end": "End (sec)", "text": "Text"})
        st.subheader("Edit Captions")
        edited_df = st.data_editor(df, num_rows="dynamic")

        if st.button("Apply Edits"):
            st.session_state["segments"] = dataframe_to_segments(edited_df)
            st.success("Edits applied! Ready to generate SRT.")

        # Generate SRT
        if st.button("Generate SRT"):
            base_name = os.path.splitext(os.path.basename(st.session_state["tmp_media_path"]))[0]
            srt_path = os.path.join(
                tempfile.gettempdir(),
                f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt"
            )
            st.session_state["srt_path"] = generate_srt(st.session_state["segments"], srt_path)
            st.success("SRT generated!")
            with open(st.session_state["srt_path"], "rb") as srt_file:
                st.download_button(
                    "Download SRT",
                    srt_file,
                    file_name=os.path.basename(st.session_state["srt_path"]),
                    mime="text/plain"
                )

        # Burn subtitles
        file_ext = os.path.splitext(st.session_state["tmp_media_path"])[1].lower()
        if ("srt_path" in st.session_state and "tmp_media_path" in st.session_state and
                burn_it and file_ext in [".mp4", ".mov", ".mkv", ".avi"]):
            st.subheader("Burn Subtitles into Video")
            if st.button("Burn Now"):
                with st.spinner("Burning subtitles..."):
                    tmp_media_path = st.session_state["tmp_media_path"]
                    base_name = os.path.splitext(os.path.basename(tmp_media_path))[0]
                    file_ext = os.path.splitext(tmp_media_path)[1].lower()

                    style_str = get_style(template, chosen_color, chosen_font_size, chosen_margin)
                    output_video = os.path.join(
                        tempfile.gettempdir(),
                        f"{base_name}_subtitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_ext}"
                    )
                    burn_subtitles(st.session_state["tmp_media_path"], st.session_state["srt_path"],
                                   output_video, style_str)
                    with open(output_video, "rb") as f:
                        st.download_button(
                            "Download Subtitled Video",
                            f,
                            file_name=os.path.basename(output_video),
                            mime="video/mp4"
                        )

if __name__ == "__main__":
    main()