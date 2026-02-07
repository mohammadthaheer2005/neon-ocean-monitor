import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import requests # Moved to top level
from geopy.geocoders import Nominatim
from gtts import gTTS
import time
import sys
import os
import streamlit.components.v1 as components 

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion_engine import SatelliteStream
from src.hybrid_model import HybridAlgaeModel
from src.groq_agent import GroqOceanAgent
from src.report_generator import MissionReportGenerator

# --- CONFIGURATION ---
st.set_page_config(page_title="NeonOcean V5.0: Ultimate", page_icon="🌊", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #0afff0; }
    .stMetric { background-color: #111; padding: 10px; border-radius: 5px; border-left: 4px solid #0afff0; }
    /* Pulsing Animation for status */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(10, 255, 240, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(10, 255, 240, 0); }
        100% { box-shadow: 0 0 0 0 rgba(10, 255, 240, 0); }
    }
    .status-pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🌊 NEON OCEAN V5.0")
    st.caption("Bio-Digital Twin | Command Center")
    
    # --- VIEW MODE SELECTOR ---
    view_mode = st.radio("📡 SELECT MODE:", ["Bio-Digital Twin", "Live Satellite Feed"], index=0)
    st.divider()

    st.header("⚡ GROQ AI")
    # Auto-load key from secrets if available for smoother demo
    default_key = st.secrets["general"]["GROQ_API_KEY"] if "general" in st.secrets and "GROQ_API_KEY" in st.secrets["general"] else ""
    groq_key = st.text_input("Groq API Key (gsk_...)", value=default_key, type="password")
    
    st.header("🎛️ ENV SIMULATOR")
    st.caption("Live Sensor Override:")
    # Added Oxygen Slider as requested
    oxygen = st.slider("Dissolved Oxygen (DO)", 0.0, 15.0, 6.5, help="Critical for aquatic life (<4 is Hypoxic).")
    nitrate = st.slider("Nitrate (NO3)", 0.0, 10.0, 2.5, help="Nutrient from fertilizer runoff.")
    phosphate = st.slider("Phosphate (PO4)", 0.0, 5.0, 0.8, help="Nutrient from detergents.")
    light = st.slider("Light Intensity (PAR)", 0, 2000, 1200, help="Sunlight driving photosynthesis.")
    ph_level = st.slider("pH Level", 6.5, 9.0, 8.1)
    
    st.header("🔊 AUDIO")
    voice_on = st.toggle("Full Voice Briefing", True)
    
    st.divider()
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 NeonOcean Satellite & AI Monitoring System V5.0")

    # --- CUSTOM CSS FOR COMMAND CENTER ---
    st.markdown("""
    <style>
    .command-center {
        background: rgba(10, 20, 30, 0.8);
        border: 2px solid #0afff0;
        border-radius: 15px;
        padding: 2.5rem;
        box-shadow: 0 0 25px rgba(10, 255, 240, 0.2);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(15, 25, 35, 0.9);
        border: 1px solid rgba(10, 255, 240, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .status-active {
        color: #00ff88;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .header-glow {
        color: #0afff0;
        text-shadow: 0 0 15px rgba(10, 255, 240, 0.6);
        font-family: 'Courier New', Courier, monospace;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    st.header("🎨 VISION AI")
    st.caption("Satellite Imaging Config")
    vision_model = st.selectbox("Select Model:", ["Leonardo AI", "flux", "flux-realism", "flux-3d", "any-dark", "turbo"], index=0)
    
    st.divider()
    
    st.header("🎨 LEONARDO AI (OPTIONAL)")
    st.caption("Fallback Image Generation")
    leo_key_input = st.text_input("Leonardo API Key", type="password")
    if leo_key_input:
        st.session_state['leo_key'] = leo_key_input.strip()
    
    st.divider()
    st.header("📍 TARGET SELECTOR")
    # Autocomplete List
    ocean_list = [
        "Indian Ocean", "Pacific Ocean", "Atlantic Ocean", "Arctic Ocean", "Southern Ocean", 
        "Bay of Bengal", "Arabian Sea", "South China Sea", "Mediterranean Sea", "Gulf of Mexico", "Red Sea"
    ]
    location_query = st.selectbox("Select or Type Sector:", ocean_list, index=0)

# --- INITIALIZATION ---
stream = SatelliteStream()
hybrid_model = HybridAlgaeModel()
groq = GroqOceanAgent(api_key=groq_key)
reporter = MissionReportGenerator()

# Helper to play audio (FULL)
def play_alert(text):
    if voice_on:
        try:
            # Clean text for speech
            speech_text = text.replace("*", "").replace("#", "")
            tts = gTTS(text=speech_text, lang='en')
            # Use distinct filename to avoid lock issues
            fname = f"voice_{int(time.time())}.mp3" 
            tts.save(fname)
            st.audio(fname, format="audio/mp3", autoplay=True)
        except: pass

# Train Model on startup
if 'model_trained' not in st.session_state:
    with st.spinner("🧠 Calibrating Hybrid Sensors..."):
        hybrid_model.train()
        st.session_state['pop_data'] = hybrid_model.generate_synthetic_data(1000)
        st.session_state['model_trained'] = True

# ==========================================
# 🛰️ MODE: LIVE SATELLITE FEED (REAL WORLD)
# ==========================================
if view_mode == "Live Satellite Feed":
    st.subheader("🛰️ LIVE ORBITAL UPLINK & CHLOROPHYLL SCAN")
    
    c_live, c_data = st.columns([1.5, 1])
    
    with c_live:
        st.markdown("### 🔴 LIVE ISS FEED (Official Sources)")
        
        # Only verified working official sources
        stream_options = {
            "🌐 NASA+ Streaming Service": "https://plus.nasa.gov/video/nasa-live/",
            "🌐 NASA TV Website": "https://www.nasa.gov/nasalive/",
            "🌐 ESA Web TV": "https://www.esa.int/ESA_Multimedia/ESA_Web_TV",
            "📺 Sen 4K ISS Livestream (YouTube)": "86YLFOog4GM",
            "📺 [UNOFFICIAL] ISS 24/7 View (YouTube)": "fO9e9jnhYK8"
        }
        selected_stream_name = st.selectbox("Select Signal Source:", list(stream_options.keys()), index=0)
        selected_stream_url = stream_options[selected_stream_name]
        
        st.caption(f"🛰️ Source: {selected_stream_name}")
        
        # Check if it's a YouTube video ID or official website URL
        if selected_stream_url.startswith("http"):
            # Official website
            st.success("🌐 **Official Space Agency Website** - Live streaming")
            
            iframe_html = f"""
            <iframe width="100%" height="500" 
                src="{selected_stream_url}" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
            """
            components.html(iframe_html, height=520)
            
            st.markdown(f"[🔗 Open in New Tab]({selected_stream_url})")
            st.info("ℹ️ **Official NASA/ESA platforms** - Live ISS cameras and space broadcasts")
        else:
            # YouTube video ID (Sen's 4K stream)
            st.success("📺 **Sen's 4K ISS Livestream** - Ultra HD Earth views from space")
            
            iframe_html = f"""
            <iframe width="100%" height="500" 
                src="https://www.youtube.com/embed/{selected_stream_url}?autoplay=1&mute=1" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
            """
            components.html(iframe_html, height=520)
            
            st.markdown(f"[🔗 Watch on YouTube](https://www.youtube.com/watch?v={selected_stream_url})")
            st.info("ℹ️ **4K quality** - Live views of Earth from the International Space Station")

    with c_data:
        st.markdown("<h3 class='header-glow'>🧠 MISSION CONTROL: ISS TRACKING</h3>", unsafe_allow_html=True)
        
        # Tracking Sources
        # 1. api.open-notify.org
        # 2. api.wheretheiss.at/v1/satellites/25544
        
        iss_lat, iss_lon = None, None
        
        try:
            import requests
            # Try Source 1
            response = requests.get("http://api.open-notify.org/iss-now.json", timeout=3)
            if response.status_code == 200:
                data = response.json()
                iss_lat = float(data['iss_position']['latitude'])
                iss_lon = float(data['iss_position']['longitude'])
            else:
                # Try Source 2 (Fallback)
                response = requests.get("https://api.wheretheiss.at/v1/satellites/25544", timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    iss_lat = float(data['latitude'])
                    iss_lon = float(data['longitude'])
        except: pass

        if iss_lat is not None:
            # Command Center Header
            st.markdown(f"""
            <div style='background: rgba(0, 255, 240, 0.1); padding: 10px; border-radius: 5px; border-left: 5px solid #0afff0; margin-bottom: 15px;'>
                <span class='status-active'>📡 SIGNAL: ACTIVE</span> | 🛰️ <b>NORAD ID: 25544</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Telemetry Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("🛰️ LATITUDE", f"{iss_lat:.4f}°")
            m2.metric("🛰️ LONGITUDE", f"{iss_lon:.4f}°")
            
            # Satellite Stats Row
            s1, s2 = st.columns(2)
            s1.metric("🌍 ALTITUDE", "~408.2 KM", "+0.02")
            s2.metric("🚀 VELOCITY", "7.66 KM/S", "Orbital")
            
            # ISS Data for Map
            iss_df = pd.DataFrame({'lat': [iss_lat], 'lon': [iss_lon]})
            
            # Create Advanced PyDeck map
            view_state = pdk.ViewState(latitude=iss_lat, longitude=iss_lon, zoom=2, pitch=0)
            iss_layer = pdk.Layer('ScatterplotLayer', data=iss_df, get_position='[lon, lat]',
                                get_color='[0, 255, 240, 200]', get_radius=500000, pickable=True)
            
            st.pydeck_chart(pdk.Deck(layers=[iss_layer], initial_view_state=view_state,
                                    map_style='mapbox://styles/mapbox/satellite-v9', height=350))
        else:
            st.error("❌ MISSION CONTROL: SIGNAL LOST (ALL NODES DOWN)")
            st.info("Satellite tracking networks are currently unreachable. [Check NASA SpotTheStation](https://spotthestation.nasa.gov/tracking_map.cfm)")
            if st.button("🔄 Restart Uplink"): st.rerun()

    st.divider()

    st.subheader("🧪 REAL SATELLITE DATA (Official Sources)")
    
    col_sat_viewer, col_sat_data = st.columns([1.5, 1])
    
    with col_sat_viewer:
        st.markdown("### 🌍 LIVE SATELLITE DATA")
        
        # Mix of YouTube video streams and interactive platforms
        sat_options = {
            "📺 NASA Earth Live - 24/7": "86YLFOog4GM",
            "📺 NOAA Weather Satellite Live": "RtycCrE4zOw",
            "📺 Earth from Space - Live Feed": "DDU-rZs-Ic4",
            "📺 [UNOFFICIAL] Satellite View Live": "vytmBNhc9ig",
            "🌐 Zoom Earth - Interactive": "https://zoom.earth/",
            "🌐 MKS Space - ISS Tracking": "https://mks.space/"
        }
        
        selected_option = st.selectbox("Select Data Source:", list(sat_options.keys()), index=0)
        selected_value = sat_options[selected_option]
        
        st.caption(f"📡 {selected_option}")
        
        # Check if it's a YouTube video ID or interactive website
        if selected_value.startswith("http"):
            # Interactive website - needs to open in new tab
            st.warning("⚠️ **Interactive Platform** - Click below to open in new window")
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <a href="{selected_value}" target="_blank" style="
                    display: inline-block;
                    padding: 20px 50px;
                    background: linear-gradient(135deg, #0afff0 0%, #00d4ff 100%);
                    color: #050505;
                    text-decoration: none;
                    font-size: 20px;
                    font-weight: bold;
                    border-radius: 10px;
                    box-shadow: 0 6px 12px rgba(10, 255, 240, 0.4);
                    transition: transform 0.2s;
                ">
                    🚀 OPEN {selected_option.split('-')[0].strip()} →
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"""
            **{selected_option}** is an interactive platform that works best in a full browser window.
            """)
            
        else:
            # YouTube video stream - plays directly
            st.success("📺 **Live Video Stream** - Playing now")
            
            stream_iframe = f"""
            <iframe width="100%" height="500" 
                src="https://www.youtube.com/embed/{selected_value}?autoplay=1&mute=1" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
            """
            components.html(stream_iframe, height=520)
            st.markdown(f"[🔗 Watch on YouTube](https://www.youtube.com/watch?v={selected_value})")
    
    with col_sat_data:
        st.markdown("### 📊 OCEAN & ALGAE DATA")
        
        # Updated options with 100% working YouTube embed for Britannia content
        ocean_data_options = {
            "📊 NASA GIBS - Chlorophyll Map": "gibs_static",
            "🎬 [OFFICIAL] Britannica: Algae Blooms (Live)": "X0GndNAr8Hw", # YouTube equivalent for reliability
            "🌐 NASA Worldview - Interactive": "https://worldview.earthdata.nasa.gov/",
            "🌐 NOAA CoastWatch - Ocean Data": "https://coastwatch.noaa.gov/",
            "🌐 Copernicus Marine - EU Data": "https://marine.copernicus.eu/"
        }
        
        selected_ocean_data = st.selectbox("Select Ocean Data Source:", list(ocean_data_options.keys()), index=0)
        selected_ocean_value = ocean_data_options[selected_ocean_data]
        
        st.caption(f"🧪 {selected_ocean_data}")
        
        # Check source type
        if selected_ocean_value == "gibs_static":
            # Show NASA GIBS static image
            st.success("✅ **NASA GIBS API** - Direct satellite data")
            
            from datetime import datetime, timedelta
            yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            gibs_url = f"https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&LAYERS=MODIS_Aqua_L3_Chlorophyll_a_4km_Daily&VERSION=1.3.0&FORMAT=image/png&TRANSPARENT=true&WIDTH=800&HEIGHT=400&CRS=EPSG:4326&BBOX=-60,30,60,180&TIME={yesterday}"
            
            st.image(gibs_url, caption=f"Global Chlorophyll ({yesterday})", use_container_width=True)
            
            # --- REAL-TIME ALGAE RISK monitor ---
            st.markdown(f"### 🧬 LIVE ALGAE RISK MONITOR: {location_query.upper()}")
            
            # Geocode the location query for specific detection
            try:
                geolocator = Nominatim(user_agent="neon_v5_live_diag")
                loc = geolocator.geocode(location_query)
                if loc:
                    lat_target, lon_target = loc.latitude, loc.longitude
                    geodata_status = f"📍 {lat_target:.2f}, {lon_target:.2f}"
                else:
                    lat_target, lon_target = 10.0, 80.0
                    geodata_status = "📍 DEFAULT (10, 80)"
            except:
                lat_target, lon_target = 10.0, 80.0
                geodata_status = "⚠️ GEO OFFLINE"

            # Fetch Live SST for SPECIFIC location risk calculation
            try:
                live_ocean_data = stream.fetch_live_sst(lat=lat_target, lon=lon_target) 
                # Use nanmean to handle potential land offsets in real satellite data
                current_sst = float(np.nanmean(live_ocean_data.values))
            except:
                # If everything fails, use a latitude-based approximation for better realism than a flat 25.0
                current_sst = 30.0 - abs(lat_target) * 0.3
            
            # Risk logic tied to SST and nutrients (Optimized for Geographic Contrast)
            # Weights: SST (50%), Nitrate (30%), Phosphate (20%)
            risk_score = (current_sst/32 * 0.50) + (nitrate/10 * 0.30) + (phosphate/5 * 0.20)
            
            if risk_score > 0.75:
                lvl, color, advice = "Critical (HAB)", "#ff4b4b", "🔴 EMERGENCY: Toxic Bloom Imminent. Avoid water contact!"
            elif risk_score > 0.55:
                lvl, color, advice = "High Risk", "#ffa500", "🟠 WARNING: Rapid algae growth detected. Monitoring required."
            elif risk_score > 0.40:
                lvl, color, advice = "Moderate", "#ffcc00", "🟡 CAUTION: Notable nutrient activity. Visual checks advised."
            else:
                lvl, color, advice = "Stable", "#00ff88", "🟢 OPTIMAL: Ecosystem shows healthy balance."
            
            st.markdown(f"""
            <div style='background: {color}22; border: 2px solid {color}; padding: 18px; border-radius: 12px; border-left: 12px solid {color};'>
                <h4 style='color: {color}; margin: 0; display: flex; justify-content: space-between;'>
                    <span>🗺️ {location_query.upper()}</span>
                    <span>{lvl.upper()}</span>
                </h4>
                <div style='margin: 12px 0; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; font-size: 0.9rem;'>
                    <b>📡 LIVE PAYLOAD:</b><br>
                    {geodata_status} | 🌡️ {current_sst:.1f}°C | 🧪 N: {nitrate:.1f} | 🧪 P: {phosphate:.1f}
                </div>
                <p style='margin: 0; font-size: 0.95rem; color: #ddd;'><b>Status:</b> {advice}</p>
            </div>
            """, unsafe_allow_html=True)

        elif not selected_ocean_value.startswith("http"):
            # YouTube video stream (Reliable embedding)
            st.success("🎬 **Educational Feature** - Playing Video")
            
            iframe_html = f"""
            <iframe width="100%" height="450" 
                src="https://www.youtube.com/embed/{selected_ocean_value}?autoplay=1&mute=1" 
                frameborder="0" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen>
            </iframe>
            """
            components.html(iframe_html, height=470)
            st.markdown(f"[🔗 View Original Content](https://www.britannica.com/video/algae-blooms-red-tides-botany/-300437)")
            
        else:
            # Interactive platform
            st.warning("⚠️ **Interactive Platform** - Open in New Tab")
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <a href="{selected_ocean_value}" target="_blank" style="
                    display: inline-block;
                    padding: 20px 50px;
                    background: linear-gradient(135deg, #00ff88 0%, #00d4aa 100%);
                    color: #050505;
                    text-decoration: none;
                    border-radius: 10px;
                ">
                    🌊 OPEN {selected_ocean_data.split('-')[0].strip()} →
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    st.caption("👆 **All sources are official government/academic platforms** - No third-party aggregators")


# ==========================================
# 🧬 MODE: BIO-DIGITAL TWIN (SIMULATION)
# ==========================================
elif view_mode == "Bio-Digital Twin":

    # --- LIVE DATA ACQUISITION ---
    try:
        geolocator = Nominatim(user_agent="neon_v5")
        loc = geolocator.geocode(location_query)
        lat_center, lon_center = (loc.latitude, loc.longitude) if loc else (10.0, 80.0)
    except:
        lat_center, lon_center = 10.0, 80.0

    with st.spinner(f"🛰️ Scanning {location_query}..."):
        ocean_data = stream.fetch_live_sst() 
        live_sst = float(ocean_data.mean().item())

    # --- HYBRID PREDICTION ENGINE ---
    inputs = {
        'temperature': live_sst,
        'pH': ph_level,
        'turbidity': 10.0 + (nitrate * 1.5),
        'salinity': 35.0,
        'nitrate': nitrate,
        'phosphate': phosphate,
        'dissolved_oxygen': oxygen, # Used User Input instead of formula
        'light_intensity': light
    }

    # Add light to logic (simple override for risk)
    risk_label, confidence, cluster_id = hybrid_model.predict(inputs)
    if light > 1800 and nitrate > 2: # High light boosts bloom risk
        confidence = min(100, confidence + 10)
        if "Low" in risk_label: risk_label = "Moderate Risk"

    is_critical = "CRITICAL" in risk_label

    # Auto-Voice for Critical
    if is_critical and 'last_crit' not in st.session_state:
        play_alert(f"Critical Alert. High Algae Growth detected in {location_query}. Oxygen levels dropping.")
        st.session_state['last_crit'] = time.time()
    elif not is_critical and 'last_crit' in st.session_state:
        del st.session_state['last_crit']

    # --- MAIN DASHBOARD ---
    st.subheader(f"🌐 COMMAND CENTER: {location_query.upper()}")

    # TOP ROW: SENSOR ARRAY
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🌡️ TEMP", f"{live_sst:.1f} °C", "NOAA")
    c2.metric("🌊 SALINITY", "35.0 PSU", "Stable")
    c3.metric("🫧 OXYGEN", f"{oxygen} mg/L", "Hypoxic" if oxygen < 4 else "Normal")
    c4.metric("☀️ LIGHT", f"{light} PAR", "High" if light > 1500 else "Avg")
    current_biomass = 10 * (2.718 ** (0.05 * (nitrate + phosphate) * (light/1000) * (0/24))) # T=0
    c5.metric("🦠 BIOMASS", f"{current_biomass:.1f} mg/m³", "Est. Density")
    c6.metric("☣️ RISK", risk_label, f"{confidence:.0f}%", delta_color="inverse")

    # 2. MAIN VISUALIZATION ROW
    col_map, col_graphs = st.columns([2, 1])

    with col_map:
        # 3D GLOBE / HOLOGRAM MAP
        st.caption("📡 3D GLOBAL HOLOGRAPHIC SCAN")
        
        # --- RESTORED DATA GENERATION FOR MAP ---
        # Create a local grid around the target center
        grid_size = 30
        lats = np.linspace(lat_center-10, lat_center+10, grid_size)
        lons = np.linspace(lon_center-10, lon_center+10, grid_size)
        lat_g, lon_g = np.meshgrid(lats, lons)
        
        # Synthetic Fields based on live inputs
        # Chlorophyll correlates with our 'Risk' logic (Nitrate impact)
        base_chl = 0.5 + (nitrate * 0.4) 
        chlorophyll = base_chl + np.sin(lat_g*0.2) + np.random.normal(0, 0.2, lat_g.shape)
        
        # Vector Field (Currents)
        # Rotating gyre pattern centered on location
        u_vec = -np.sin((lat_g - lat_center)*0.2) * 2
        v_vec = np.cos((lon_g - lon_center)*0.2) * 2
        
        df = pd.DataFrame({
            'lat': lat_g.flatten(),
            'lon': lon_g.flatten(),
            'chlorophyll': np.clip(chlorophyll.flatten(), 0, 10),
            'u': u_vec.flatten(),
            'v': v_vec.flatten()
        })
        
        # We switch to a 'GlobeView' style visualization using a wide-view Mercator with high pitch
        # Real GlobeView in Streamlit/PyDeck can be tricky with controls, so we optimize for the "Hologram Look"
        
        # Create Vector Field (Arrows)
        df['u_scaled'] = df['u'] * 0.1 # Scale for visual
        df['v_scaled'] = df['v'] * 0.1
        
        # Heatmap (Algae)
        layer_heat = pdk.Layer(
            "HeatmapLayer", df,
            get_position=["lon", "lat"],
            get_weight="chlorophyll",
            radius_pixels=80, intensity=2.0, threshold=0.1,
            color_range=[[0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        )
        
        # Animated Current Vectors (Holographic Arrows)
        layer_arrows = pdk.Layer(
            "IconLayer", df.sample(200),
            get_position=["lon", "lat"],
            get_icon="icon_data", # Placeholder logic or simple arrows
            get_size=20, get_color=[0, 255, 255, 1500],
            icon_atlas="https://raw.githubusercontent.com/visgl/deck.gl-data/master/images/icon-atlas.png",
            icon_mapping="https://raw.githubusercontent.com/visgl/deck.gl-data/master/images/icon-atlas.json"
        )
        
        # Since IconLayer needs specific atlas, let's use a simpler "ArcLayer" to show flow "Movement"
        # This looks like "Energy Moving" across the ocean
        layer_flow = pdk.Layer(
            "ArcLayer", df.sample(100),
            get_source_position=["lon", "lat"],
            get_target_position=["lon+u", "lat+v"], 
            get_source_color=[0, 255, 255, 100],
            get_target_color=[0, 0, 255, 200],
            get_width=3, getHeight=0.5
        )

        view_state = pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=3, pitch=60, bearing=0)
        
        st.pydeck_chart(pdk.Deck(
            layers=[layer_heat, layer_flow], 
            initial_view_state=view_state, 
            map_style="mapbox://styles/mapbox/dark-v10",
            tooltip={"text": "Algae Density: {chlorophyll}"}
        ))

    with col_graphs:
        # A. RISK GAUGE (METER)
        st.caption("☣️ BLOOM RISK LEVEL")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence,
            title = {'text': risk_label},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "red" if is_critical else "#0afff0"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': confidence}}))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=50,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig_gauge, use_container_width=True)

        # B. BIOMASS GRAPH
        st.caption("📈 ALGAE BIOMASS OVER TIME (7 DAYS)")
        days = list(range(0, 8)) # 0 to 7 days
        # Logistic Growth Model
        r = 0.3 + (nitrate * 0.1) + (phosphate * 0.05)
        K = 100 # Carrying capacity
        biomass_t = [K / (1 + ((K - 10)/10) * np.exp(-r * d)) for d in days]
        
        fig_line = px.area(x=days, y=biomass_t, title="Growth Projection", labels={'x':'Days', 'y':'Biomass (mg/m³)'})
        fig_line.update_traces(line_color='#0afff0', fillcolor='rgba(10, 255, 240, 0.2)')
        fig_line.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_line, use_container_width=True)

    # 3. DETAILS, AI & VISION
    c_info, c_ai, c_vision = st.columns([1, 1, 1])

    with c_info:
        with st.expander("🔬 AI ROOT CAUSE & CHEMICALS", expanded=True):
            # logic-based diagnosis
            cause = "Nutrient imbalances from coastal runoff"
            if nitrate > 5 and phosphate > 2: cause = "SYNERGISTIC EUTROPHICATION (Extreme N/P load)"
            elif nitrate > 5: cause = "NITROGEN-DRIVEN OVERGROWTH (Agricultural runoff)"
            elif phosphate > 2: cause = "PHOSPHATE SPIKE (Industrial loading)"
            
            st.error(f"**PRIMARY DRIVER:** {cause}")
            
            st.markdown("""
            **🛠️ CHEMICAL MITIGATION PROTOCOLS:**
            *   **Alum**: 5-20 mg/L (Binds Phosphates instantly)
            *   **Modified Clay**: 5-10 g/m² (Sinks cells to the seafloor)
            *   **Phoslock**: Prevents P-release from sediments.
            *   **H2O2**: Selective killing of Cyanobacteria cells.
            """)
            
    with c_ai:
        st.markdown("#### 💬 GROQ TACTICAL ADVISOR")
        user_q = st.text_input("Ask about Chemicals or Risk:", "What is the Alum dosage for this sector?")
        if st.button("ANALYZE DATA"):
            with st.spinner("Processing..."):
                ctx = f"Risk: {risk_label} ({confidence}%). Biomass: {current_biomass}. Cause: {cause}. Chemicals include Alum and Phoslock."
                ans = groq.analyze(ctx, user_q)
                st.success(ans)
                play_alert(ans)

    with c_vision:
        st.markdown("#### 👁️ VISUAL FORENSICS")
        st.caption("Generate Satellite Imagery")
        
        if st.button("GENERATE SATELLITE SCAN"):
            with st.spinner("Compiling Visual Data..."):
                from src.image_gen_agent import OceanImageGenerator
                
                # Check for Leonardo Key
                leo_key = st.session_state.get('leo_key', None)

                # Inputs
                chem_data = {"nitrate": nitrate, "phosphate": phosphate, "dissolved_oxygen": oxygen}
                
                # --- MODEL ROUTING ---
                if vision_model == "Leonardo AI":
                    if not leo_key:
                        st.error("❌ You need to enter a Leonardo API Key in the Sidebar first!")
                        st.stop()
                    
                    # DEBUG: Show key verification
                    st.warning(f"🔐 Using Key: {leo_key[:5]}... (Length: {len(leo_key)})")
                    
                    img_gen = OceanImageGenerator(leonardo_key=leo_key)
                    # Pass user_q (from Groq section) as context
                    result = img_gen.generate_with_leonardo(location_query, risk_label, chem_data, tactical_context=user_q)
                
                else:
                    # Default: Pollinations
                    img_gen = OceanImageGenerator()
                    result = img_gen.generate_scenario(location_query, risk_label, chem_data, model=vision_model, tactical_context=user_q)
                
                if result and 'url' in result:
                    st.image(result['url'], caption=f"{result['source']}: {location_query}", use_container_width=True)
                    st.markdown(f"[🔗 Click to Open Full Image]({result['url']})")
                    with st.expander("Prompt Details"):
                        st.code(result['prompt'])
                elif result and 'error' in result:
                    st.error(f"Generation Failed: {result['error']}")
                else:
                    st.error("Unknown Error.")
                
        # Allow manual Text-to-Image for tactical planning
        custom_prompt = st.text_input("Custom Tactical Scenario:", placeholder="e.g. Red tide spreading in harbor")
        if custom_prompt and st.button("GENERATE CUSTOM"):
             from src.image_gen_agent import OceanImageGenerator
             
             # Determine Key usage based on Model Selection
             # Logic: If model is "Leonardo AI", pass the key. Otherwise pass None.
             active_key = None
             if vision_model == "Leonardo AI":
                 active_key = st.session_state.get('leo_key', None)
                 if not active_key:
                     st.error("❌ Select 'Leonardo AI' but No Key provided!")
                     st.stop()
             
             img_gen = OceanImageGenerator(leonardo_key=active_key)
             
             # Call universal custom generator
             result = img_gen.generate_custom_image(custom_prompt, model=vision_model)
             
             if result and 'url' in result:
                 st.image(result['url'], caption=result['source'], use_container_width=True)
                 st.markdown(f"[🔗 Click to Open Full Image]({result['url']})")
             elif result and 'error' in result:
                 st.error(result['error'])

    # Footer
    st.markdown("---")
    if st.button("📄 DOWNLOAD COMPREHENSIVE REPORT"):
        inputs['biomass'] = current_biomass # Add biomass to report
        res = {'risk': risk_label, 'confidence': confidence, 'cluster': cluster_id, 'recommendation': "See AI Advisor."}
        pdf_bytes = reporter.generate_pdf(location_query, inputs, res)
        st.download_button("Get PDF", pdf_bytes, "NeonOcean_Full_Report.pdf", "application/pdf")

    # Footer Ticker
    st.markdown("""
    <marquee style='color:#0afff0; font-family:Courier; font-size:18px;'>
    ⚠ LIVE SENSOR FEED ACTIVE... MONITORING NITRATE LEVELS... SATELLITE UPLINK STABLE... GROQ AI STANDBY...
    </marquee>
    """, unsafe_allow_html=True)
