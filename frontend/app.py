import streamlit as st
import subprocess
import os
import shutil
import time
import re
from pathlib import Path
import librosa
import numpy as np
from pydub import AudioSegment

# === Directory Setup ===
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STEMS_DIR = BASE_DIR / "stems"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STEMS_DIR, exist_ok=True)

# Clean up old ZIPs to avoid clutter
for zip_file in BASE_DIR.glob("*.zip"):
    zip_file.unlink()

# === Streamlit App Config ===
st.set_page_config(page_title="Stemster", layout="wide")
st.title("🎶 Stemster Audio Separator")
st.markdown("Upload an audio file, analyze it, and extract clean stems using Demucs with GPU acceleration.")

# === Sidebar: Configuration Panel ===
st.sidebar.header("⚙️ Demucs Settings")
model_choice = st.sidebar.selectbox("Model:", ["htdemucs", "demucs48_hq", "mdx_extra"])
shifts = st.sidebar.slider("Shifts (quality vs speed):", 1, 10, 1)
overlap = st.sidebar.slider("Overlap (blending):", 0.0, 1.0, 0.25, step=0.01)
stem_options = st.sidebar.multiselect("Stems to keep:", ["vocals", "drums", "bass", "other"], default=["vocals", "drums", "bass", "other"])
export_format = st.sidebar.selectbox("Export Format:", ["mp3", "wav"], index=0)

# === Main Upload Section ===
st.subheader("📤 Upload Your Audio")
uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "flac"])

if uploaded_file:
    audio_path = UPLOAD_DIR / uploaded_file.name
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🗑️ Delete Uploaded File"):
        os.remove(audio_path)
        st.success(f"Deleted {uploaded_file.name}")
        st.rerun()

    # === Audio Analysis ===
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)
        key_index = chroma_mean.argmax()
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = note_names[key_index]
        estimated_tempo = float(tempo[0]) if isinstance(tempo, (list, tuple, np.ndarray)) else float(tempo)

        st.info(f"**Estimated Key**: {estimated_key}")
        st.info(f"**Estimated Tempo**: {round(estimated_tempo)} BPM")
    except Exception as e:
        st.warning(f"Tempo/key analysis failed: {e}")

    # === Generate Stems ===
    if st.button("🎛️ Generate Stems"):
        with st.spinner("Running Demucs... please wait."):
            progress_bar = st.progress(0)
            output_lines = []

            command = [
                "/root/stemster/backend/demucs/demucs_env/bin/demucs",
                "-n", model_choice,
                "--shifts", str(shifts),
                "--overlap", str(overlap),
                "-d", "cuda",
                str(audio_path)
            ]

            step_map = {
                "Loading": 10,
                "Separating": 30,
                "Computing": 60,
                "Saving": 90,
                "Done": 100
            }

            process = subprocess.Popen(
                command,
                cwd="/root/stemster/backend/demucs",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            last_progress = 0
            start_time = time.time()

            for line in process.stdout:
                line = line.strip()
                output_lines.append(line)
                for keyword, prog in step_map.items():
                    if re.search(keyword, line, re.IGNORECASE) and prog > last_progress:
                        progress_bar.progress(prog)
                        last_progress = prog

            process.wait()
            elapsed = time.time() - start_time
            progress_bar.empty()

            output_root = BASE_DIR / "backend" / "demucs" / "separated" / model_choice
            output_folders = sorted(output_root.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

            if process.returncode == 0 and output_folders:
                demucs_output_dir = output_folders[0]
                base_name = uploaded_file.name.rsplit(".", 1)[0]
                target_stem_dir = STEMS_DIR / base_name

                if target_stem_dir.exists():
                    shutil.rmtree(target_stem_dir)
                shutil.move(str(demucs_output_dir), str(target_stem_dir))

                if export_format == "mp3":
                    for wav_file in target_stem_dir.glob("*.wav"):
                        audio = AudioSegment.from_wav(wav_file)
                        mp3_path = wav_file.with_suffix(".mp3")
                        audio.export(mp3_path, format="mp3", bitrate="192k")
                        wav_file.unlink()

                st.success(f"✅ Stems generated in **{round(elapsed, 2)} seconds**.")
            else:
                st.error("❌ Demucs failed or no output found.")
                with st.expander("🪵 Show Demucs Log"):
                    st.text("\n".join(output_lines))

# === Sidebar: Track Manager ===
st.sidebar.markdown("---")
st.sidebar.header("🎵 Track Manager")

uploaded_files = list(UPLOAD_DIR.glob("*"))
stem_projects = [d for d in STEMS_DIR.glob("*") if d.is_dir()]

if uploaded_files:
    st.sidebar.subheader("📁 Uploaded Files")
    for file in uploaded_files:
        with st.sidebar.expander(f"🎧 {file.name}"):
            st.audio(str(file))
            st.download_button("⬇️ Download", file.read_bytes(), file.name, key=f"download_{file.name}")
            if st.button("🗑️ Delete", key=f"del_upload_{file.name}"):
                os.remove(file)
                st.success(f"Deleted {file.name}")
                st.rerun()

if stem_projects:
    st.sidebar.subheader("🧬 Stems Generated")
    for track in stem_projects:
        with st.sidebar.expander(f"🎶 {track.name}"):
            for stem_file in sorted(track.glob("*.mp3")):
                st.audio(str(stem_file))
                stem_label = stem_file.stem.lower()
                icon_map = {
                    "vocals": "🎤",
                    "drums": "🥁",
                    "bass": "🎸",
                    "other": "🎛️"
                }
                label_icon = icon_map.get(stem_label, "🎵")
                st.download_button(f"{label_icon} {stem_file.name}", stem_file.read_bytes(), file_name=stem_file.name, key=f"download_{track.name}_{stem_file.name}")

            zip_path = Path(f"{track}.zip")
            if not zip_path.exists():
                shutil.make_archive(str(track), 'zip', track)

            with open(zip_path, "rb") as f:
                st.download_button("📦 Download All (ZIP)", f, file_name=zip_path.name, key=f"zip_{track.name}")

            if st.button("🗑️ Delete Project", key=f"del_project_{track.name}"):
                if zip_path.exists():
                    zip_path.unlink()
                shutil.rmtree(track)
                st.success(f"Deleted {track.name}")
                st.rerun()
