# 🎵 Stemster – Open Source Stem Separator

Stemster is a powerful, open-source stem separator designed for musicians, producers, and audio enthusiasts. Built with Python and Streamlit, Stemster leverages the Demucs model to extract individual stems (vocals, drums, bass, and more) from mixed audio tracks — now with CUDA GPU acceleration, MP3 compression, and streamlined web performance.

## 🚀 Features
 - Upload and analyze audio (MP3, WAV, FLAC)
 - Automatically detects tempo and key using librosa
 - Choose from multiple Demucs models: htdemucs, demucs48_hq, mdx_extra
 - Runs on CPU or GPU (CUDA-enabled)
 - Adjust shifts and overlap for performance/quality balance
 - Extract clean stems (vocals, bass, drums, other)
 - Export stems in MP3 (default) or WAV
 - Download individual stems or ZIP bundles
 - Automatically cleans up unused zip files
 - Hosted with NGINX reverse proxy + SSL on a Proxmox-based Linux container

## 🖥️ Tech Stack
 - Python (Streamlit)
 - Demucs (Facebook AI)
 - Librosa (audio analysis)
 - Torch (CUDA-enabled)
 - NGINX + Certbot (Reverse Proxy + HTTPS)
 - Self-hosted in Proxmox VE

## 🌐 Live Demo
 - Visit: https://stemster.lkdpsy.com
 - GPU-accelerated and running in production.

## 🧰 Setup Instructions
### 1. Clone the Repo
git clone https://github.com/quazatic/stemster.git
cd stemster

### 2. Setup Python Environments
#### Frontend (Streamlit)
cd frontend
python3 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
#### Backend (Demucs)
cd ../backend/demucs
python3 -m venv demucs_env
source demucs_env/bin/activate
pip install -r requirements.txt

### 3. Run the App
cd frontend
source streamlit_env/bin/activate
streamlit run app.py

App runs at http://localhost:8501 by default.

## ⚡ Performance Notes
 - For GPU acceleration, ensure you're running on a compatible CUDA device (e.g., RTX 4060 Ti).
 - Output stems are exported as MP3 by default, dramatically improving performance vs WAV.
 - Tracks and stem zips are stored in /uploads and /stems, but are .gitignored.

## 🤝 Credits
 - Demucs by Facebook AI Research
 - Streamlit for frontend
 - Librosa for key/tempo analysis
 - Hosted with love and maintained by @quazatic

## 📄 License
MIT License © 2025 Stemster Contributors
