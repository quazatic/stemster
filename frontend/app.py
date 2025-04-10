import streamlit as st
import subprocess
import os
import shutil
import time
from pathlib import Path
import librosa
import numpy as np

# === Directory Setup ===
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
STEMS_DIR = BASE_DIR / "stems"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STEMS_DIR, exist_ok=True)

# === Streamlit App ===
st.set_page_config(page_title="🎵 Stemster", layout="wide")
st.title(":musical_note: Stemster Audio Separator")
st.markdown("Upload an audio file, analyze it, and extract clean stems using Demucs.")

# === Layout Columns ===
settings_col, upload_col = st.columns([1, 2])

# === Sidebar: Project History ===
st.sidebar.header(":open_file_folder: Past Stem Projects")
all_dirs = sorted([d for d in STEMS_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
track_names = [d.name for d in all_dirs]
selected_track = st.sidebar.selectbox("Select a track:", track_names if track_names else ["No tracks yet"])

# === Demucs Settings ===
with settings_col:
    st.subheader(":control_knobs: Demucs Settings")
    model_choice = st.selectbox("Model:", ["htdemucs", "demucs48_hq", "mdx_extra"])
    shifts = st.slider("Shifts (quality vs speed):", 1, 10, 1)
    overlap = st.slider("Overlap (blending):", 0.0, 1.0, 0.25, step=0.01)

# === File Upload ===
with upload_col:
    st.subheader(":inbox_tray: Upload Your Audio")
    uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "flac"])

    if uploaded_file:
        audio_path = UPLOAD_DIR / uploaded_file.name
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

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
        if st.button(":microphone: Generate Stems"):
            with st.spinner("Running Demucs... please wait."):
                progress = st.progress(0)
                for i in range(10):
                    time.sleep(0.1)
                    progress.progress((i + 1) * 10)

                try:
                    # Define dynamic output root based on selected model
                    output_root = BASE_DIR / "backend" / "demucs" / "separated" / model_choice

                    # Run Demucs
                    result = subprocess.run([
                        "/root/stemster/backend/demucs/demucs_env/bin/demucs",
                        "-n", model_choice,
                        "--shifts", str(shifts),
                        "--overlap", str(overlap),
                        "-d", "cpu",
                        str(audio_path)
                    ], check=True, cwd="/root/stemster/backend/demucs", capture_output=True, text=True)

                    output_folders = sorted(output_root.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
                    if not output_folders:
                        raise FileNotFoundError("No Demucs output found.")
                    demucs_output_dir = output_folders[0]

                    base_name = uploaded_file.name.rsplit(".", 1)[0]
                    target_stem_dir = STEMS_DIR / base_name

                    # ✅ Overwrite if the folder already exists
                    if target_stem_dir.exists():
                        shutil.rmtree(target_stem_dir)

                    shutil.move(str(demucs_output_dir), str(target_stem_dir))

                    st.success("✅ Stems Generated!")

                    for stem_file in sorted(target_stem_dir.glob("*.wav")):
                        st.audio(str(stem_file))
                        with open(stem_file, "rb") as f:
                            st.download_button(
                                label=f"⬇️ Download {stem_file.name}",
                                data=f,
                                file_name=stem_file.name,
                                mime="audio/wav",
                                key=f"dl_new_{stem_file.name}"
                            )

                except subprocess.CalledProcessError as e:
                    st.error(f"Demucs failed with error:\n{e.stderr}")
                except Exception as e:
                    st.error(f"Something went wrong while processing: {e}")

                progress.empty()

# === History Preview ===
if selected_track and selected_track != "No tracks yet":
    st.subheader(f":headphones: Preview: {selected_track}")
    stem_dir = STEMS_DIR / selected_track
    for stem_file in sorted(stem_dir.glob("*.wav")):
        st.audio(str(stem_file))
        with open(stem_file, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {stem_file.name}",
                data=f,
                file_name=stem_file.name,
                mime="audio/wav",
                key=f"dl_hist_{stem_file.name}"
            )
