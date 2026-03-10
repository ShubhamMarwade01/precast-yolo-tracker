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

# Minimal Custom CSS for clean padding and tight UI
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.8rem !important; font-weight: 600 !important; color: #1e293b; margin-bottom: 0; padding-bottom: 0;}
    h3 { font-size: 1.2rem !important; font-weight: 400 !important; color: #475569; margin-top: 0; padding-top: 0;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>L&T Varanasi Precast Yard</h1>", unsafe_allow_html=True)
st.markdown("<h3>Automated Mould Status Tracking System</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Shubham\OneDrive - National Institute of Construction Management & Research\02 Personal\004 SEM 4\Research Work\mould_detection\best.pt"
    return YOLO(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model. Please check the path. Details: {e}")
    st.stop()

# --- 3. SESSION STATE ---
if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=["Timestamp", "Time", "Assembled", "Dismantled"])
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = 0
if 'last_frame_bytes' not in st.session_state:
    st.session_state.last_frame_bytes = None

# --- 4. SIDEBAR (Clean Configuration) ---
with st.sidebar:
    st.markdown("**System Controls**")
    source_type = st.radio("Input Source", ["Live IP Camera", "Image capture and detect", "Image Upload"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Detection Parameters**")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.70, 0.05)
    log_interval = st.number_input("Log Interval (sec)", 1, 60, 5)
    
    st.markdown("---")
    st.markdown("**Export Data**")
    
    # Screenshot Download Button
    if st.session_state.last_frame_bytes is not None:
        st.download_button(
            label="📸 Save Screenshot",
            data=st.session_state.last_frame_bytes,
            file_name=f"Detection_Capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True,
            help="Download the latest annotated image. Note: Clicking this pauses the live feed."
        )
    
    # CSV Data Download Button
    if not st.session_state.log_df.empty:
        csv = st.session_state.log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Session Log (CSV)",
            data=csv,
            file_name=f"LnT_Mould_Log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv',
            use_container_width=True
        )

# --- 5. MAIN DASHBOARD LAYOUT ---
col_main, col_data = st.columns([6, 4], gap="large")

with col_main:
    frame_placeholder = st.empty()
    cam_placeholder = st.empty() 

with col_data:
    st.markdown("**Current Status**")
    kpi_col1, kpi_col2 = st.columns(2)
    metric_assembled = kpi_col1.empty()
    metric_dismantled = kpi_col2.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("**Production Trend**")
    trend_placeholder = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("View Raw Detection Log"):
        log_placeholder = st.empty()

# --- 6. LOGIC & VISUALIZATION ---
def update_dashboard(frame, conf_val):
    # Run YOLO Detection
    results = model(frame, conf=conf_val)[0]
    annotated_frame = results.plot()
    
    # Encode the frame to a JPG byte array so it can be downloaded as a screenshot
    success, buffer = cv2.imencode('.jpg', annotated_frame)
    if success:
        st.session_state.last_frame_bytes = buffer.tobytes()
    
    # Extract Counts
    classes = results.boxes.cls.cpu().numpy()
    count_assembled = int(np.count_nonzero(classes == 0))
    count_dismantled = int(np.count_nonzero(classes == 1))
    
    now = datetime.now()
    timestamp_full = now.strftime("%Y-%m-%d %H:%M:%S")
    time_only = now.strftime("%H:%M:%S")
    
    # 1. Update Video/Image (THIS IS THE LINE THAT GOT CUT OFF PREVIOUSLY)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # 2. Update Clean KPIs
    metric_assembled.metric("Assembled (Rebar)", count_assembled)
    metric_dismantled.metric("Dismantled (Empty)", count_dismantled)
    
    # 3. Optimized Logging
    current_time = time.time()
    if current_time - st.session_state.last_log_time >= log_interval:
        new_log = pd.DataFrame([{
            "Timestamp": timestamp_full, 
            "Time": time_only,
            "Assembled": count_assembled, 
            "Dismantled": count_dismantled
        }])
        st.session_state.log_df = pd.concat([st.session_state.log_df, new_log]).tail(60) 
        st.session_state.last_log_time = current_time
        
        log_placeholder.dataframe(st.session_state.log_df.iloc[::-1], use_container_width=True, hide_index=True)
        
        # 4. Update Minimal Plotly Graph
        if len(st.session_state.log_df) > 1:
            fig = px.area(
                st.session_state.log_df, 
                x="Time", 
                y=["Assembled", "Dismantled"],
                color_discrete_map={"Assembled": "#2563eb", "Dismantled": "#94a3b8"}
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor='#e2e8f0', title=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="")
            )
            trend_placeholder.plotly_chart(fig, use_container_width=True, key=f"trend_{time.time()}")

# --- 7. EXECUTION TRIGGER ---
if source_type == "Live IP Camera":
    default_ip = "http://10.122.7.66:8080/video" 
    ip_url = st.sidebar.text_input("Network Stream URL:", value=default_ip)
    
    col_start, col_stop = st.sidebar.columns(2)
    start_button = col_start.button("▶ Connect", use_container_width=True)
    stop_button = col_stop.button("⏹ Stop", use_container_width=True)
    
    if start_button and ip_url:
        cap = cv2.VideoCapture(ip_url)
        
        if not cap.isOpened():
            st.error("Connection failed. Verify IP, network, and mobile app.")
        else:
            while cap.isOpened() and not stop_button:
                success, frame = cap.read()
                if not success:
                    st.warning("Stream disconnected.")
                    break
                update_dashboard(frame, conf_threshold)
            cap.release()

elif source_type == "Image capture and detect":
    with cam_placeholder:
        cam_image = st.camera_input("Capture Yard State via Webcam")
    if cam_image is not None:
        file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        update_dashboard(frame, conf_threshold)

elif source_type == "Image Upload":
    uploaded_file = st.sidebar.file_uploader("Test Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        update_dashboard(frame, conf_threshold)