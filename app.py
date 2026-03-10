import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="L&T Precast Tracker", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1 { font-size: 1.8rem !important; font-weight: 600 !important; color: #1e293b; margin-bottom: 0;}
h3 { font-size: 1.2rem !important; font-weight: 400 !important; color: #475569; margin-top: 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>L&T Varanasi Precast Yard</h1>", unsafe_allow_html=True)
st.markdown("<h3>Automated Mould Status Tracking System</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = "best.pt"   # ✅ GitHub compatible path
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. SESSION STATE ---
if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Timestamp","Time","Assembled","Dismantled"])

if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = 0

if 'last_frame_bytes' not in st.session_state:
    st.session_state.last_frame_bytes = None

# --- 4. SIDEBAR ---
with st.sidebar:

    st.markdown("### System Controls")

    source_type = st.radio(
        "Input Source",
        ["Live IP Camera","Image capture and detect","Image Upload"]
    )

    st.markdown("---")

    st.markdown("### Detection Parameters")

    conf_threshold = st.slider("Confidence Threshold",0.10,1.00,0.70)

    log_interval = st.number_input("Log Interval (sec)",1,60,5)

# --- 5. DASHBOARD LAYOUT ---
col_main,col_data = st.columns([6,4])

with col_main:

    frame_placeholder = st.empty()
    cam_placeholder = st.empty()

with col_data:

    st.markdown("### Current Status")

    c1,c2 = st.columns(2)

    metric_assembled = c1.empty()
    metric_dismantled = c2.empty()

    st.markdown("### Production Trend")

    trend_placeholder = st.empty()

    with st.expander("View Raw Detection Log"):
        log_placeholder = st.empty()

# --- 6. DETECTION FUNCTION ---
def update_dashboard(frame,conf_val):

    results = model(frame,conf=conf_val)[0]

    annotated_frame = results.plot()

    success,buffer = cv2.imencode('.jpg',annotated_frame)

    if success:
        st.session_state.last_frame_bytes = buffer.tobytes()

    classes = results.boxes.cls.cpu().numpy()

    count_assembled = int(np.count_nonzero(classes==0))
    count_dismantled = int(np.count_nonzero(classes==1))

    now = datetime.now()

    timestamp_full = now.strftime("%Y-%m-%d %H:%M:%S")
    time_only = now.strftime("%H:%M:%S")

    frame_rgb = cv2.cvtColor(annotated_frame,cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame_rgb,use_container_width=True)

    metric_assembled.metric("Assembled (Rebar)",count_assembled)
    metric_dismantled.metric("Dismantled (Empty)",count_dismantled)

    current_time = time.time()

    if current_time - st.session_state.last_log_time >= log_interval:

        new_log = pd.DataFrame([{

            "Timestamp":timestamp_full,
            "Time":time_only,
            "Assembled":count_assembled,
            "Dismantled":count_dismantled

        }])

        st.session_state.log_df = pd.concat(
            [st.session_state.log_df,new_log]
        ).tail(60)

        st.session_state.last_log_time = current_time

        log_placeholder.dataframe(
            st.session_state.log_df.iloc[::-1],
            use_container_width=True
        )

        if len(st.session_state.log_df)>1:

            fig = px.area(

                st.session_state.log_df,
                x="Time",
                y=["Assembled","Dismantled"]

            )

            trend_placeholder.plotly_chart(
                fig,
                use_container_width=True
            )

# --- 7. INPUT SOURCES ---

if source_type=="Image Upload":

    uploaded_file = st.sidebar.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes,1)

        update_dashboard(frame,conf_threshold)


elif source_type=="Image capture and detect":

    cam_image = st.camera_input("Capture Yard Image")

    if cam_image is not None:

        file_bytes = np.asarray(
            bytearray(cam_image.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes,1)

        update_dashboard(frame,conf_threshold)
