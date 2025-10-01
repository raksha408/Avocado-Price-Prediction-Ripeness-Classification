import streamlit as st
import subprocess
import webbrowser
import socket
import time
import os
import sys
import base64

# 🥑 Page configuration
st.set_page_config(
    page_title="🥑 Avocado Hub: Price & Ripeness",
    page_icon="🥑",
    layout="centered"
)

# ✅ Convert local image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ✅ Your image path
image_path = r"C:\Users\sraks\OneDrive\Pictures\Avocado_Pic.png"

# ✅ Load and embed as CSS background
if os.path.exists(image_path):
    base64_img = get_base64_image(image_path)
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(229, 244, 227, 0.9), rgba(229, 244, 227, 0.9)),
                        url("data:image/png;base64,{base64_img}") no-repeat center center fixed;
            background-size: cover;
            backdrop-filter: blur(8px);
        }}
        </style>
    """, unsafe_allow_html=True)
else:
    st.error(f"❌ Image not found at: {image_path}")

# ✅ Style setup
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Luckiest+Guy&family=Comic+Neue:wght@700&display=swap');

    .main-title {
        text-align: center;
        font-family: 'Luckiest Guy', cursive;
        font-size: clamp(42px, 6vw, 80px);
        color: #3A743A;
        text-shadow: 2px 2px #B5E7B5;
        margin-top: 30px;
        margin-bottom: 8px;
    }

    .subtitle-wrapper {
        display: flex;
        justify-content: center;
    }

    .subtitle {
        text-align: center;
        font-family: 'Comic Neue', cursive;
        font-size: 20px;
        color: #2E6B2E;
        margin-bottom: 30px;
        background: rgba(255,255,255,0.7);
        padding: 12px 20px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    .button-container {
        width: 65%;
        margin: 0 auto 40px auto;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(120deg, #d9fdd3, #fffbd5);
        color: #2E6B2E;
        font-family: 'Comic Neue', cursive;
        border: 2px solid #c2e5c2;
        border-radius: 28px;
        padding: 14px 0;
        font-size: 20px;
        font-weight: bold;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.25s ease, box-shadow 0.25s ease, background 0.3s ease;
        position: relative;
        z-index: 2;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        background: linear-gradient(120deg, #c9f5c0, #f8f4c3);
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    }

    /* 🔒 Hide Streamlit menu, header, footer */
    #MainMenu, header, footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Main Title
st.markdown("<div class='main-title'>AVOCADO HUB</div>", unsafe_allow_html=True)

# ✅ Subtitle


# ✅ Button container
st.markdown("<div class='button-container'>", unsafe_allow_html=True)

# ✅ Price button
if st.button("💰 Predict Price with ₹ Price Peek"):
    def is_port_open(port):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            return False

    def launch_streamlit_app(script_path, port):
        if is_port_open(port):
            st.warning(f"🔄 Already running on port {port}. Opening…")
            webbrowser.open_new_tab(f"http://localhost:{port}")
            return
        cmd = [sys.executable, "-m", "streamlit", "run", script_path, "--server.port", str(port)]
        kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if os.name == 'nt':
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(cmd, **kwargs)
        for _ in range(20):
            if is_port_open(port):
                webbrowser.open_new_tab(f"http://localhost:{port}")
                return
            time.sleep(0.5)
        st.error(f"❌ Could not start {script_path}. Is Streamlit installed?")

    launch_streamlit_app("avocado_price/price_app.py", 8502)

st.markdown("</div>", unsafe_allow_html=True)

# ✅ Ripeness button
if st.button("🥑 Detect Ripeness with a Squeeze"):
    def is_port_open(port):
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except OSError:
            return False

    def launch_streamlit_app(script_path, port):
        if is_port_open(port):
            st.warning(f"🔄 Already running on port {port}. Opening…")
            webbrowser.open_new_tab(f"http://localhost:{port}")
            return
        cmd = [
            sys.executable, "-m", "streamlit", "run", script_path,
            "--server.port", str(port)
        ]
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.STDOUT,
            "cwd": os.getcwd()
        }
        if os.name == 'nt':
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(cmd, **kwargs)
        for _ in range(20):
            if is_port_open(port):
                webbrowser.open_new_tab(f"http://localhost:{port}")
                return
            time.sleep(0.5)
        st.error(f"❌ Could not start {script_path}. Is Streamlit installed?")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "ripe_app.py"),
        os.path.join(base_dir, "avocado_price", "ripe_app.py"),
        os.path.join(base_dir, "avocado_ripeness", "ripe_app.py"),
    ]

    for path in candidates:
        if os.path.exists(path):
            launch_streamlit_app(path, 8503)
            break
    else:
        st.error(
            "❌ Could not find ripe_app.py in any of:\n" +
            "\n".join(f" • {p}" for p in candidates)
        )
