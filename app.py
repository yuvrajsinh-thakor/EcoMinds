# ╔══════════════════════════════════════════════════════════════╗
# ║  🌿 EcoMinds v2.0 — Intelligent Energy Analysis System      ║
# ║  GTU Final Year Project | Premium UI Edition                ║
# ║  Design: Glassmorphism Dark + Emerald Neon Accents          ║
# ╚══════════════════════════════════════════════════════════════╝

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import requests, warnings, time
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

try:
    import anthropic
    CLAUDE_OK = True
except ImportError:
    CLAUDE_OK = False

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoMinds — Energy Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — Complete CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
  --bg:        #060D0A;
  --bg1:       #0C1810;
  --bg2:       #0F2016;
  --bg3:       #162B1E;
  --glass:     rgba(22,43,30,0.6);
  --glass2:    rgba(22,43,30,0.3);
  --green:     #00FF87;
  --green2:    #00CC6A;
  --green3:    #00994F;
  --green-dim: rgba(0,255,135,0.08);
  --green-glow:rgba(0,255,135,0.15);
  --amber:     #FFB347;
  --red:       #FF5C5C;
  --blue:      #60A5FA;
  --text:      #E8F5EE;
  --text2:     #8FB09A;
  --text3:     #4D7A5E;
  --border:    rgba(0,255,135,0.1);
  --border2:   rgba(0,255,135,0.2);
  --shadow:    0 8px 32px rgba(0,0,0,0.4);
  --glow:      0 0 20px rgba(0,255,135,0.2);
  --radius:    14px;
  --radius-lg: 20px;
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.main .block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1400px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--green3); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg1) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── Sidebar Logo Area ── */
.sidebar-logo {
  background: linear-gradient(135deg, var(--bg2), var(--bg3));
  border-bottom: 1px solid var(--border);
  padding: 1.5rem 1.2rem 1rem;
  margin: -1rem -1rem 1rem;
}
.sidebar-logo h1 {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  color: var(--green) !important;
  margin: 0 !important;
  letter-spacing: -0.5px;
  text-shadow: 0 0 20px rgba(0,255,135,0.4);
}
.sidebar-logo p {
  font-size: 0.7rem !important;
  color: var(--text3) !important;
  margin: 2px 0 0 !important;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Nav Item Styling ── */
.nav-section {
  font-family: 'Syne', sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  color: var(--text3);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  padding: 0.8rem 0 0.3rem;
  margin-bottom: 0.2rem;
}

/* ── Streamlit Radio as Nav ── */
div[data-testid="stRadio"] > label { display: none !important; }
div[data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: column !important;
  gap: 2px !important;
}
div[data-testid="stRadio"] > div > label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 9px 14px !important;
  border-radius: 10px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
  font-size: 0.85rem !important;
  font-weight: 500 !important;
  color: var(--text2) !important;
  border: 1px solid transparent !important;
}
div[data-testid="stRadio"] > div > label:hover {
  background: var(--green-dim) !important;
  color: var(--green) !important;
  border-color: var(--border) !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label[aria-checked="true"] {
  background: var(--green-glow) !important;
  color: var(--green) !important;
  border-color: var(--border2) !important;
  box-shadow: 0 0 12px rgba(0,255,135,0.1) !important;
}
div[data-testid="stRadio"] > div > label > div:first-child { display: none !important; }

/* ── Page Header ── */
.page-header {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 1.5rem;
  padding-bottom: 1.2rem;
  border-bottom: 1px solid var(--border);
}
.page-header-icon {
  width: 48px; height: 48px;
  background: var(--green-glow);
  border: 1px solid var(--border2);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem;
  flex-shrink: 0;
}
.page-header h2 {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.6rem !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  margin: 0 !important;
  letter-spacing: -0.5px;
}
.page-header p {
  font-size: 0.82rem !important;
  color: var(--text2) !important;
  margin: 2px 0 0 !important;
}

/* ── Glass Cards ── */
.glass-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
  backdrop-filter: blur(12px);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.glass-card:hover { border-color: var(--border2); box-shadow: var(--glow); }

.glass-card-green {
  background: linear-gradient(135deg, rgba(0,255,135,0.06), rgba(0,153,79,0.04));
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  padding: 1.2rem 1.4rem;
}

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 1.2rem; }
.kpi-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
  position: relative;
  overflow: hidden;
  transition: all 0.25s ease;
}
.kpi-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--green), transparent);
}
.kpi-card:hover { border-color: var(--border2); transform: translateY(-2px); box-shadow: var(--glow); }
.kpi-label { font-size: 0.68rem; color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 500; color: var(--green); line-height: 1; }
.kpi-sub   { font-size: 0.7rem; color: var(--text3); margin-top: 0.3rem; }
.kpi-icon  { position: absolute; top: 0.8rem; right: 0.8rem; font-size: 1.1rem; opacity: 0.5; }

/* ── Alert Banners ── */
.alert {
  border-radius: var(--radius);
  padding: 0.9rem 1.2rem;
  display: flex; align-items: flex-start; gap: 0.8rem;
  margin: 0.5rem 0;
  font-size: 0.85rem;
}
.alert-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 1px; }
.alert-green  { background: rgba(0,255,135,0.06); border: 1px solid rgba(0,255,135,0.2); color: #b7e4c7; }
.alert-amber  { background: rgba(255,179,71,0.06); border: 1px solid rgba(255,179,71,0.25); color: #ffe0b2; }
.alert-red    { background: rgba(255,92,92,0.06); border: 1px solid rgba(255,92,92,0.2); color: #ffcccc; }

/* ── Section Titles ── */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.75rem;
  font-weight: 700;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin: 1.5rem 0 0.8rem;
  display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Stat Highlight ── */
.stat-highlight {
  font-family: 'JetBrains Mono', monospace;
  font-size: 2.4rem;
  font-weight: 500;
  color: var(--green);
  line-height: 1;
  text-shadow: 0 0 30px rgba(0,255,135,0.3);
}
.stat-label { font-size: 0.75rem; color: var(--text3); margin-top: 0.3rem; }

/* ── Score Gauge Card ── */
.score-big {
  background: linear-gradient(135deg, var(--bg2), var(--bg3));
  border: 1px solid var(--border2);
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.score-big::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 0%, rgba(0,255,135,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.score-number {
  font-family: 'Syne', sans-serif;
  font-size: 4.5rem;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 0.2rem;
}
.score-grade { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; }

/* ── Badge Cards ── */
.badge-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; }
.badge-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  text-align: center;
  transition: all 0.2s;
  cursor: default;
}
.badge-card.earned {
  border-color: var(--border2);
  background: var(--green-dim);
  box-shadow: 0 0 16px rgba(0,255,135,0.08);
}
.badge-card .badge-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.badge-card .badge-name { font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700; color:var(--text); }
.badge-card .badge-desc { font-size:0.65rem; color:var(--text3); margin-top:0.2rem; }
.badge-card .badge-status { font-size:0.62rem; color:var(--green); font-weight:600; margin-top:0.3rem; letter-spacing:0.05em; }

/* ── Chat Interface ── */
.chat-wrap { display:flex; flex-direction:column; gap:10px; padding:1rem 0; max-height:420px; overflow-y:auto; }
.chat-msg { display:flex; gap:10px; align-items:flex-start; }
.chat-msg.user { flex-direction:row-reverse; }
.chat-avatar {
  width:32px; height:32px; border-radius:50%; display:flex;
  align-items:center; justify-content:center; font-size:0.9rem; flex-shrink:0;
}
.chat-avatar.ai   { background: var(--green-glow); border:1px solid var(--border2); }
.chat-avatar.user { background: rgba(96,165,250,0.15); border:1px solid rgba(96,165,250,0.2); }
.chat-bubble {
  max-width:75%; padding:10px 14px; border-radius:14px;
  font-size:0.85rem; line-height:1.5;
}
.chat-bubble.ai   { background:var(--glass); border:1px solid var(--border); color:var(--text); border-top-left-radius:4px; }
.chat-bubble.user { background:rgba(96,165,250,0.12); border:1px solid rgba(96,165,250,0.2); color:var(--text); border-top-right-radius:4px; }

/* ── Suggestion Pills ── */
.pill-grid { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1rem; }
.pill {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 6px 14px;
  font-size: 0.78rem;
  color: var(--text2);
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
}
.pill:hover { border-color: var(--border2); color: var(--green); background: var(--green-dim); }

/* ── IoT Appliance Card ── */
.appliance-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.appliance-card {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  transition: all 0.2s;
}
.appliance-card.on  { border-color: rgba(0,255,135,0.25); }
.appliance-card.standby { border-color: rgba(255,179,71,0.25); }
.appliance-card.off { border-color: rgba(255,92,92,0.15); opacity:0.7; }
.appliance-name { font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; color:var(--text); }
.appliance-room { font-size:0.7rem; color:var(--text3); margin-top:2px; }
.appliance-power { font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:var(--green); margin:0.5rem 0 0.2rem; }
.status-dot { width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:5px; }
.dot-on { background:var(--green); box-shadow:0 0 6px var(--green); animation:pulse 2s infinite; }
.dot-standby { background:var(--amber); box-shadow:0 0 6px var(--amber); }
.dot-off  { background:var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Slab Table ── */
.slab-table { width:100%; border-collapse:collapse; font-size:0.82rem; }
.slab-table th { padding:8px 12px; text-align:left; background:var(--bg2); color:var(--text3); font-weight:600; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid var(--border); }
.slab-table td { padding:8px 12px; border-bottom:1px solid var(--border); color:var(--text); }
.slab-table tr:hover td { background:var(--green-dim); }
.slab-active td { color:var(--green); font-weight:600; }

/* ── Solar Metrics ── */
.solar-hero {
  background: linear-gradient(135deg, rgba(0,255,135,0.05), rgba(0,153,79,0.03));
  border: 1px solid var(--border2);
  border-radius: var(--radius-lg);
  padding: 1.8rem;
  text-align: center;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--border2) !important;
  box-shadow: 0 0 0 2px rgba(0,255,135,0.1) !important;
}
.stSlider > div > div > div > div { background: var(--green) !important; }
.stButton > button {
  background: var(--green-dim) !important;
  border: 1px solid var(--border2) !important;
  color: var(--green) !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  padding: 0.45rem 1rem !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: rgba(0,255,135,0.15) !important;
  box-shadow: 0 0 12px rgba(0,255,135,0.15) !important;
}
.stButton > button[kind="primary"] {
  background: var(--green) !important;
  color: #000 !important;
}
div[data-testid="stMetric"] {
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
}
div[data-testid="stMetric"] label { color: var(--text3) !important; font-size:0.72rem !important; text-transform:uppercase; letter-spacing:0.08em; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--green) !important; font-family:'JetBrains Mono',monospace !important; font-size:1.4rem !important; }
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:var(--radius) !important; }
.stExpander { border:1px solid var(--border) !important; border-radius:var(--radius) !important; background:var(--glass) !important; }
.stExpander summary { color:var(--text) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg2); border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: var(--text2); font-family:'Syne',sans-serif; font-weight:600; font-size:0.8rem; }
.stTabs [aria-selected="true"] { background: var(--green-glow) !important; color: var(--green) !important; }
div[data-testid="stFileUploader"] { border:1px dashed var(--border2) !important; border-radius:var(--radius) !important; background:var(--glass) !important; }
.stCheckbox > label { color: var(--text2) !important; font-size:0.82rem !important; }
.stRadio > label { color: var(--text2) !important; font-size:0.82rem !important; }
.stInfo { background:rgba(96,165,250,0.06) !important; border:1px solid rgba(96,165,250,0.2) !important; border-radius:var(--radius) !important; color:var(--text) !important; }
.stSuccess { background:rgba(0,255,135,0.06) !important; border:1px solid var(--border2) !important; border-radius:var(--radius) !important; color:var(--text) !important; }
.stWarning { background:rgba(255,179,71,0.06) !important; border:1px solid rgba(255,179,71,0.25) !important; border-radius:var(--radius) !important; color:var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
GUJARAT_TARIFF = [
    {"slab":"0 – 50 units",    "rate":1.90,"upto":50},
    {"slab":"51 – 100 units",  "rate":2.55,"upto":100},
    {"slab":"101 – 200 units", "rate":3.45,"upto":200},
    {"slab":"201 – 300 units", "rate":4.60,"upto":300},
    {"slab":"301 – 500 units", "rate":5.25,"upto":500},
    {"slab":"500+ units",      "rate":5.65,"upto":9999},
]
SOLAR_IRRADIANCE = {
    "Ahmedabad":5.8,"Surat":5.5,"Vadodara":5.6,
    "Rajkot":5.9,"Gandhinagar":5.7,"Jamnagar":6.0,"Other":5.5,
}
CO2_FACTOR   = 0.82
INDIA_AVG_KWH = 3.0   # daily average

FEATURES = ["DayOfYear","DayOfWeek","Month","WeekOfYear","IsWeekend","Lag_1","Lag_7","Roll_7","Roll_30"]

# ── Plotly dark theme consistent with design ──────────────────
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,24,16,0.6)",
    font=dict(family="DM Sans", color="#8FB09A"),
    margin=dict(l=20,r=20,t=40,b=20),
    xaxis=dict(gridcolor="rgba(0,255,135,0.05)", zerolinecolor="rgba(0,255,135,0.1)"),
    yaxis=dict(gridcolor="rgba(0,255,135,0.05)", zerolinecolor="rgba(0,255,135,0.1)"),
)

# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    for k,v in {
        "chat_history":[],"iot_data":None,"iot_last_fetch":None,
        "page":"Dashboard","active_tab":"dashboard",
    }.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

# ══════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=600)
def generate_mock_data(n_days=730):
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    doy   = np.array([d.timetuple().tm_yday for d in dates])
    kwh   = np.clip(
        8.5
        + 3.5*np.cos(2*np.pi*(doy-15)/365)
        + np.where(np.array([d.weekday() for d in dates])>=5,1.5,0.0)
        + np.linspace(0,1,n_days)
        + np.random.normal(0,0.8,n_days),
        1.0, 25.0)
    return pd.DataFrame({"Date":dates,"kWh":np.round(kwh,3)})

def load_data(file=None):
    if file:
        try:
            raw  = pd.read_csv(file)
            dcol = [c for c in raw.columns if "date" in c.lower() or "time" in c.lower()]
            ncol = [c for c in raw.columns if raw[c].dtype in [np.float64,np.int64] and c not in dcol]
            if dcol and ncol:
                df = raw[[dcol[0],ncol[0]]].copy()
                df.columns = ["Date","kWh"]
                df["Date"] = pd.to_datetime(df["Date"])
                return df.dropna().sort_values("Date").reset_index(drop=True)
        except: pass
    return generate_mock_data()

def simulate_iot_mock():
    np.random.seed(int(time.time())%999)
    raw = [
        {"id":"a1","name":"Living Room AC",  "type":"AC",          "room":"Living Room","status":"ON",     "watts":1500},
        {"id":"a2","name":"Refrigerator",    "type":"Refrigerator","room":"Kitchen",    "status":"ON",     "watts":150},
        {"id":"a3","name":"Geyser",          "type":"Geyser",      "room":"Bathroom",   "status":"STANDBY","watts":2000},
        {"id":"a4","name":"LED TV",          "type":"TV",          "room":"Living Room","status":"ON",     "watts":80},
        {"id":"a5","name":"Ceiling Fan",     "type":"Fan",         "room":"Bedroom",    "status":"ON",     "watts":75},
        {"id":"a6","name":"WiFi Router",     "type":"Computer",    "room":"Office",     "status":"ON",     "watts":12},
    ]
    PF = {"AC":0.85,"Refrigerator":0.92,"Geyser":1.0,"TV":0.95,"Fan":0.9,"Computer":0.9}
    apps, total_p = [], 0
    for a in raw:
        pf   = PF.get(a["type"],0.9)
        pw   = a["watts"]*(1.0 if a["status"]=="ON" else 0.05)*np.random.uniform(0.92,1.05)
        v    = 230+np.random.uniform(-5,5)
        i    = pw/(v*pf) if pw>0 else 0
        ap   = v*i; rp=np.sqrt(max(0,ap**2-pw**2))
        kwh  = round(pw*(5/60)/1000,4); total_p+=pw
        apps.append({**a,"voltage_v":round(v,2),"current_a":round(i,3),
            "active_power_w":round(pw,2),"apparent_power_va":round(ap,2),
            "reactive_power_var":round(rp,2),"power_factor":round(pf,3),
            "energy_kwh":kwh,"cost_inr":round(kwh*6,3),"co2_kg":round(kwh*0.82,4),
            "frequency_hz":round(50+np.random.uniform(-0.05,0.05),3)})
    s = dict(total_active_power_w=round(total_p,2),
             total_current_a=round(sum(a["current_a"] for a in apps),3),
             total_energy_kwh_today=round(sum(a["energy_kwh"] for a in apps),4),
             total_cost_inr=round(sum(a["cost_inr"] for a in apps),3),
             total_co2_kg=round(sum(a["co2_kg"] for a in apps),4),
             grid_voltage_v=round(230+np.random.uniform(-2,2),2),
             grid_frequency_hz=round(50+np.random.uniform(-0.05,0.05),3),
             power_factor_avg=0.91)
    return {"api_version":"1.0","device_id":"ECOMINDS-METER-001",
            "location":"My Home","timestamp":datetime.now().isoformat(),
            "interval_minutes":5,"summary":s,"appliances":apps}

# ── ML helpers ─────────────────────────────────────────────────
def feat_eng(df):
    d = df.copy()
    d["DayOfYear"]  = d["Date"].dt.dayofyear
    d["DayOfWeek"]  = d["Date"].dt.dayofweek
    d["Month"]      = d["Date"].dt.month
    d["WeekOfYear"] = d["Date"].dt.isocalendar().week.astype(int)
    d["IsWeekend"]  = (d["DayOfWeek"]>=5).astype(int)
    d["Lag_1"]      = d["kWh"].shift(1)
    d["Lag_7"]      = d["kWh"].shift(7)
    d["Roll_7"]     = d["kWh"].shift(1).rolling(7).mean()
    d["Roll_30"]    = d["kWh"].shift(1).rolling(30).mean()
    return d.dropna().reset_index(drop=True)

def train_model(df, choice):
    d = feat_eng(df); X,y = d[FEATURES],d["kWh"]; sp = len(X)-30
    if choice=="XGBoost" and XGBOOST_OK:
        m = XGBRegressor(n_estimators=300,learning_rate=0.05,max_depth=6,random_state=42,verbosity=0)
        m.fit(X.iloc[:sp],y.iloc[:sp],eval_set=[(X.iloc[sp:],y.iloc[sp:])],verbose=False)
    elif choice=="Prophet" and PROPHET_OK:
        m = Prophet(yearly_seasonality=True,weekly_seasonality=True,interval_width=0.80)
        m.fit(df.rename(columns={"Date":"ds","kWh":"y"})); return m,{"MAE":0,"R2":0},d
    else:
        m = RandomForestRegressor(n_estimators=200,max_depth=10,random_state=42,n_jobs=-1) if choice!="Linear Regression" else LinearRegression()
        m.fit(X.iloc[:sp],y.iloc[:sp])
    pred = m.predict(X.iloc[sp:])
    return m,{"MAE":round(mean_absolute_error(y.iloc[sp:],pred),3),"R2":round(r2_score(y.iloc[sp:],pred),3)},d

def forecast(model,df_feat,n,choice):
    if choice=="Prophet" and PROPHET_OK:
        fc = model.predict(model.make_future_dataframe(periods=n)).tail(n)
        return pd.DataFrame({"Date":pd.to_datetime(fc["ds"]),"Forecasted_kWh":fc["yhat"].round(3).values,"Lower":fc["yhat_lower"].round(3).values,"Upper":fc["yhat_upper"].round(3).values})
    hist = list(df_feat["kWh"].values[-35:]); last = df_feat["Date"].iloc[-1]; rows=[]
    for i in range(1,n+1):
        fd=last+timedelta(days=i)
        row={"DayOfYear":fd.timetuple().tm_yday,"DayOfWeek":fd.weekday(),"Month":fd.month,
             "WeekOfYear":fd.isocalendar()[1],"IsWeekend":int(fd.weekday()>=5),
             "Lag_1":hist[-1],"Lag_7":hist[-7] if len(hist)>=7 else np.mean(hist),
             "Roll_7":np.mean(hist[-7:]),"Roll_30":np.mean(hist[-30:]) if len(hist)>=30 else np.mean(hist)}
        p=max(float(model.predict(pd.DataFrame([row])[FEATURES])[0]),0.5)
        hist.append(p)
        rows.append({"Date":fd,"Forecasted_kWh":round(p,3),"Lower":round(p*0.90,3),"Upper":round(p*1.10,3)})
    return pd.DataFrame(rows)

def calc_bill(units):
    bill,breakdown,rem,prev=0,[],units,0
    for s in GUJARAT_TARIFF:
        inn=min(rem,s["upto"]-prev);
        if inn<=0: break
        ch=inn*s["rate"]; bill+=ch
        breakdown.append({"Slab":s["slab"],"Units":round(inn,2),"Rate":s["rate"],"Charge ₹":round(ch,2)})
        rem-=inn; prev=s["upto"]
        if rem<=0: break
    bill+=50; breakdown.append({"Slab":"Fixed Charges","Units":"—","Rate":"—","Charge ₹":50})
    return round(bill,2),breakdown

def sustainability_score(avg,thr):
    r=avg/thr
    if r<=0.6:   return 95,"A+","#00FF87"
    elif r<=0.8: return 82,"A", "#00CC6A"
    elif r<=1.0: return 68,"B", "#FFB347"
    elif r<=1.2: return 50,"C", "#FF8C42"
    elif r<=1.5: return 32,"D", "#FF5C5C"
    else:        return 15,"F", "#CC0000"

def ask_ai(q, ctx):
    """
    Data-aware AI advisor. ctx must contain real computed stats from the
    loaded dataset (uploaded CSV, mock data, or IoT feed).
    Tries Claude API first; falls back to rich rule-based engine.
    """
    avg    = ctx.get("avg", 0)
    thr    = ctx.get("thr", 12)
    grade  = ctx.get("grade", "B")
    score  = ctx.get("score", 50)
    co2    = ctx.get("co2", 0)
    total  = ctx.get("total", 0)
    days   = ctx.get("days", 1)
    peak   = ctx.get("peak", avg)
    lowest = ctx.get("lowest", avg)
    top    = ctx.get("top", "your AC")
    streak = ctx.get("streak", 0)
    monthly_avg_kwh = ctx.get("monthly_avg", avg * 30)
    est_bill = ctx.get("est_bill", round(monthly_avg_kwh * 6, 0))
    data_src = ctx.get("data_src", "mock data")

    # ── Try Claude API (if key available) ───────────────────
    if CLAUDE_OK:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY","")
            if key:
                system_prompt = f"""You are EcoMinds AI, an intelligent home energy advisor integrated into the EcoMinds dashboard.
The user is asking about their own energy data. Answer using ONLY the real numbers below.
Be conversational, specific, and actionable. Keep answers to 3–5 sentences. Use ₹ for costs.

=== REAL USER DATA ({data_src}) ===
- Average daily usage: {avg:.2f} kWh/day
- Total consumption: {total:.1f} kWh over {days} days
- Peak day: {peak:.2f} kWh | Lowest day: {lowest:.2f} kWh
- Carbon footprint: {co2:.1f} kg CO₂ total
- Sustainability grade: {grade} (score {score}/100)
- Daily threshold set by user: {thr} kWh
- Top consuming appliance (IoT): {top}
- Monthly average: {monthly_avg_kwh:.0f} kWh
- Estimated monthly bill: ₹{est_bill:.0f} (Gujarat tariff)
- Current green streak: {streak} days under threshold
- Data source: {data_src}
===
When answering, always reference the actual numbers above. Never make up data."""
                c = anthropic.Anthropic(api_key=key)
                r = c.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=400,
                    system=system_prompt,
                    messages=[{"role":"user","content":q}])
                return r.content[0].text
        except: pass

    # ── Rich rule-based fallback (data-aware) ───────────────
    ql = q.lower()
    excess = max(0, avg - thr)
    saving_potential = round(excess * 30 * 6, 0)
    above_india = avg > 3.0   # India daily avg ~90kWh/month = 3kWh/day

    if any(w in ql for w in ["average","avg","daily","usage","consume","how much"]):
        cmp = "above" if above_india else "below"
        return (f"📊 Based on your **{data_src}**, your average is **{avg:.2f} kWh/day** "
                f"over {days} days (total: {total:.1f} kWh). This is **{cmp} India's national average** "
                f"of ~3.0 kWh/day per person. Your peak was {peak:.2f} kWh and lowest was {lowest:.2f} kWh. "
                f"Grade: **{grade}** ({score}/100).")

    if any(w in ql for w in ["high","spike","why","increase","peak","maximum"]):
        return (f"🔺 Your peak day hit **{peak:.2f} kWh**, which is "
                f"{round((peak/avg-1)*100)}% above your average of {avg:.2f} kWh. "
                f"Spikes are most likely from **{top}** — your highest consuming device per IoT data. "
                f"Check if it ran continuously during 6–10 PM peak hours. "
                f"A 5-star rated replacement or smart timer could reduce this significantly.")

    if any(w in ql for w in ["reduce","save","cut","lower","tip","improve","better"]):
        return (f"💰 You're currently averaging **{avg:.2f} kWh/day** — "
                f"{'above' if avg>thr else 'within'} your {thr} kWh threshold. "
                f"{'Reducing to ' + str(round(thr*0.8,1)) + ' kWh/day could save ~₹' + str(saving_potential) + '/month. ' if excess>0 else 'Keep it up! '}"
                f"Focus on **{top}** first — it's your biggest IoT-detected consumer. "
                f"Also consider off-peak scheduling (10 PM–6 AM) for washing machines and geysers.")

    if any(w in ql for w in ["bill","cost","money","rupee","₹","pay","expense"]):
        return (f"💰 Your estimated monthly bill is **₹{est_bill:.0f}** based on {monthly_avg_kwh:.0f} kWh/month "
                f"and Gujarat DGVCL/UGVCL tariff slabs. "
                f"Your current {avg:.2f} kWh/day puts you in the "
                f"{'cheaper lower slabs' if monthly_avg_kwh<200 else 'higher tariff slabs'}. "
                f"{'Reducing by ' + str(round(excess*30,0)) + ' kWh/month could save ~₹' + str(saving_potential) + '.' if excess>0 else 'You are already in a good slab!'}")

    if any(w in ql for w in ["carbon","co2","environment","emission","footprint","green","planet"]):
        trees = int(co2/21)
        daily_co2 = round(avg * 0.82, 2)
        return (f"🌍 Your total CO₂ footprint from this data is **{co2:.1f} kg** "
                f"({daily_co2} kg/day). That's equivalent to cutting down **{trees} trees**. "
                f"Reducing to {thr*0.8:.1f} kWh/day would save ~{round(excess*30*0.82,1)} kg CO₂/month. "
                f"Installing a 2kW solar panel in Gujarat could offset ~{round(2*5.7*0.8*365*0.82,0):.0f} kg CO₂/year!")

    if any(w in ql for w in ["solar","panel","renewable","sun","invest"]):
        return (f"☀️ With your avg of **{avg:.2f} kWh/day** ({monthly_avg_kwh:.0f} kWh/month), "
                f"a **2–3 kW solar system** would cover most of your needs. "
                f"Gujarat gets 5.5–6.0 kWh/m²/day — ideal for solar. "
                f"Cost ~₹90,000 before PM Surya Ghar Yojana subsidy (up to ₹78,000 for 3kW+). "
                f"Payback period: ~3–5 years. Visit the **Solar Advisor tab** for exact ROI!")

    if any(w in ql for w in ["grade","score","rating","rank","performance"]):
        next_grade = "A" if grade in ["B","C","D","F"] else "A+" if grade=="A" else "already at A+"
        target_kwh = round(thr * 0.79, 1)
        return (f"📊 Your sustainability grade is **{grade}** with a score of **{score}/100**. "
                f"Your avg {avg:.2f} kWh/day vs threshold {thr} kWh. "
                f"To reach **{next_grade}**, target below **{target_kwh} kWh/day**. "
                f"{'You have a ' + str(streak) + '-day green streak — keep going!' if streak>0 else 'Start a streak by staying under threshold daily!'}")

    if any(w in ql for w in ["streak","streak","badge","achievement","reward"]):
        return (f"🔥 Your current green streak is **{streak} days** under {thr} kWh. "
                f"Out of the last 30 days of your data, you were green on {ctx.get('green_days',0)} days. "
                f"Keep your streak alive to earn the **Week Warrior** badge (7 days) and "
                f"**Eco Champion** badge (Grade A). Small daily wins add up!")

    if any(w in ql for w in ["appliance","device","ac","fridge","geyser","fan","tv"]):
        return (f"🏠 According to your IoT data, **{top}** is your highest consuming device right now. "
                f"Your total household avg is {avg:.2f} kWh/day. "
                f"To get an appliance-level breakdown with % share and cost per device, "
                f"visit the **Appliance Breakdown** tab and enter your appliances.")

    if any(w in ql for w in ["forecast","predict","future","next","tomorrow","week","month"]):
        fc_kwh = round(avg * 30, 1)
        fc_bill = calc_bill(fc_kwh)[0]
        return (f"🔮 Based on your historical avg of **{avg:.2f} kWh/day**, "
                f"the next 30 days forecast is ~**{fc_kwh} kWh** (₹{fc_bill:.0f} estimated bill). "
                f"For a detailed ML forecast with confidence bands, visit the **ML Forecasting tab**. "
                f"XGBoost and Prophet models give the most accurate multi-week predictions.")

    # Generic fallback with real data
    return (
        f"⚡ EcoMinds summary from **{data_src}**: "
        f"Avg **{avg:.2f} kWh/day**, Grade **{grade}** ({score}/100), "
        f"Total {total:.1f} kWh over {days} days, CO2 {co2:.1f} kg. "
        f"Est. monthly bill Rs.{est_bill:.0f}, top device **{top}**. "
        f"Try asking: Why is my usage high? How to save money? Should I go solar?"
    )

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <h1>🌿 EcoMinds</h1>
      <p>Energy Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Unified navigation with section headers ────────────────
    NAV_PAGES = [
        "🏠  Dashboard",
        "🔌  IoT Live Feed",
        "📈  Analytics & EDA",
        "🤖  ML Forecasting",
        "🏠  Appliance Breakdown",
        "💰  Bill Predictor",
        "☀️  Solar Advisor",
        "🏆  Gamification",
        "🗣️  AI Chat",
    ]
    # Inject section separators via markdown (visual only, not selectable)
    nav_sel = st.radio("nav_main", NAV_PAGES,
                       label_visibility="collapsed",
                       key="nav_main_radio")
    # Add visual section labels via JS injection
    st.markdown("""
    <style>
    /* Auto-insert section headers before specific nav items */
    div[data-testid="stRadio"] > div > label:nth-child(1)::before
      { content:"MAIN"; display:block; font-size:0.6rem; color:var(--text3);
        letter-spacing:0.15em; padding:0.6rem 0 0.1rem; font-family:'Syne',sans-serif;
        font-weight:700; text-transform:uppercase; }
    div[data-testid="stRadio"] > div > label:nth-child(5)::before
      { content:"INSIGHTS"; display:block; font-size:0.6rem; color:var(--text3);
        letter-spacing:0.15em; padding:0.8rem 0 0.1rem; font-family:'Syne',sans-serif;
        font-weight:700; text-transform:uppercase; }
    div[data-testid="stRadio"] > div > label:nth-child(8)::before
      { content:"ENGAGE"; display:block; font-size:0.6rem; color:var(--text3);
        letter-spacing:0.15em; padding:0.8rem 0 0.1rem; font-family:'Syne',sans-serif;
        font-weight:700; text-transform:uppercase; }
    </style>
    """, unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:1rem 0'>", unsafe_allow_html=True)

    # ── Data Controls ─────────────────────────────────────────
    st.markdown('<div class="nav-section">Data</div>', unsafe_allow_html=True)
    upload       = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    model_opts   = ["Random Forest","Linear Regression"] + (["XGBoost"] if XGBOOST_OK else []) + (["Prophet"] if PROPHET_OK else [])
    model_choice = st.selectbox("🤖 ML Model", model_opts, label_visibility="visible")
    forecast_days = st.slider("Forecast Days", 7, 30, 14)
    threshold_kwh = st.slider("Daily Threshold kWh", 5.0, 25.0, 12.0, 0.5)
    co2_ui        = st.number_input("CO₂ Factor kg/kWh", 0.1, 1.5, CO2_FACTOR, 0.01, label_visibility="visible")

    # ── IoT Controls ──────────────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown('<div class="nav-section">IoT Simulator</div>', unsafe_allow_html=True)
    use_mock = st.checkbox("Use Mock Data", value=True)
    iot_url  = st.text_input("API URL", value="http://localhost:3000/api/live", label_visibility="collapsed")

    if st.button("⟳  Refresh IoT", width='stretch'):
        st.session_state.iot_data     = simulate_iot_mock() if use_mock else (lambda d: d if d else st.session_state.iot_data)(requests.get(iot_url,timeout=5).json() if not use_mock else None)
        st.session_state.iot_last_fetch = datetime.now()

    # ── System Status ─────────────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.68rem;color:var(--text3);line-height:2">
      <span style="color:{'var(--green)' if XGBOOST_OK else 'var(--text3)'}">{'●' if XGBOOST_OK else '○'}</span> XGBoost &nbsp;
      <span style="color:{'var(--green)' if PROPHET_OK else 'var(--text3)'}">{'●' if PROPHET_OK else '○'}</span> Prophet &nbsp;
      <span style="color:{'var(--green)' if CLAUDE_OK else 'var(--text3)'}">{'●' if CLAUDE_OK else '○'}</span> Claude AI<br>
      <span style="color:var(--text3)">GTU Final Year Project 2024–25</span>
    </div>
    """, unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────
df_main   = load_data(upload)
total_kwh = round(df_main["kWh"].sum(),2)
avg_daily = round(df_main["kWh"].mean(),2)
total_co2 = round(total_kwh*co2_ui,2)
score,grade,score_color = sustainability_score(avg_daily,threshold_kwh)

# Auto-load mock IoT on first visit
if not st.session_state.iot_data:
    st.session_state.iot_data     = simulate_iot_mock()
    st.session_state.iot_last_fetch = datetime.now()

# ── Resolve which page is active ──────────────────────────────
_ns = st.session_state.get("nav_main_radio","🏠  Dashboard")
if   "Dashboard"         in _ns: ACTIVE = "dashboard"
elif "IoT"               in _ns: ACTIVE = "iot"
elif "Analytics"         in _ns: ACTIVE = "eda"
elif "ML Forecasting"    in _ns: ACTIVE = "ml"
elif "Appliance"         in _ns: ACTIVE = "appliance"
elif "Bill"              in _ns: ACTIVE = "bill"
elif "Solar"             in _ns: ACTIVE = "solar"
elif "Gamification"      in _ns: ACTIVE = "game"
elif "AI Chat"           in _ns: ACTIVE = "chat"
else:                             ACTIVE = "dashboard"

# ══════════════════════════════════════════════════════════════
#  HELPER: PAGE HEADER
# ══════════════════════════════════════════════════════════════
def ph(icon, title, desc):
    st.markdown(f"""
    <div class="page-header">
      <div class="page-header-icon">{icon}</div>
      <div><h2>{title}</h2><p>{desc}</p></div>
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)

def kpi5(vals):
    """vals = list of (label, value, sub, icon)"""
    cols = st.columns(len(vals))
    for col,(label,val,sub,icon) in zip(cols,vals):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════
if ACTIVE == "dashboard":
    ph("🏠","Dashboard","Real-time overview of your energy consumption and sustainability metrics")

    kpi5([
        ("Total Consumption", f"{total_kwh:,}",       "kWh all time",        "⚡"),
        ("Avg Daily Usage",   f"{avg_daily}",          f"kWh · target {threshold_kwh}","📅"),
        ("Carbon Footprint",  f"{total_co2:,}",        "kg CO₂ total",        "🌫️"),
        ("Eco Grade",         grade,                   f"Score {score}/100",   "🏆"),
        ("Data History",      f"{len(df_main):,}",     "days tracked",        "📊"),
    ])

    # Alert
    if avg_daily > threshold_kwh:
        excess = round((avg_daily - threshold_kwh)*30*6,0)
        st.markdown(f"""<div class="alert alert-amber">
          <span class="alert-icon">⚠️</span>
          <div><strong>Above Threshold</strong> — Avg {avg_daily} kWh/day exceeds your {threshold_kwh} kWh target.
          Costing ~₹{excess:.0f} extra/month. Visit <em>Appliance Breakdown</em> to identify the culprits.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="alert alert-green">
          <span class="alert-icon">✅</span>
          <div><strong>On Track!</strong> — Avg {avg_daily} kWh/day is within your {threshold_kwh} kWh target.
          You're saving ~₹{round((threshold_kwh-avg_daily)*30*6,0):.0f}/month vs threshold. Keep it up! 🌱</div>
        </div>""", unsafe_allow_html=True)

    # Charts — 2 col layout
    sec("Energy Overview")
    ch1, ch2 = st.columns([2,1])

    with ch1:
        df_p = df_main.copy()
        df_p["Roll7"] = df_p["kWh"].rolling(7,min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p["Date"],y=df_p["kWh"],name="Daily kWh",
            line=dict(color="rgba(0,255,135,0.4)",width=1),
            fill="tozeroy",fillcolor="rgba(0,255,135,0.04)"))
        fig.add_trace(go.Scatter(x=df_p["Date"],y=df_p["Roll7"],name="7-Day Avg",
            line=dict(color="#00FF87",width=2)))
        fig.add_hline(y=threshold_kwh,line_dash="dash",line_color="#FF5C5C",
            annotation_text="Threshold",annotation_font_color="#FF5C5C",
            annotation_position="top right")
        fig.update_layout(**PLOT_LAYOUT,height=320,title="Daily Energy Consumption",
            legend=dict(orientation="h",y=1.12,x=0))
        st.plotly_chart(fig,width='stretch')

    with ch2:
        # Month-wise donut of last 6 months
        df_m = df_main.copy()
        df_m["Month"] = df_m["Date"].dt.strftime("%b %Y")
        last6 = df_m.groupby("Month")["kWh"].sum().tail(6)
        fig2 = go.Figure(go.Pie(labels=last6.index,values=last6.values,hole=0.6,
            marker=dict(colors=["#00FF87","#00CC6A","#00994F","#FFB347","#FF8C42","#FF5C5C"]),
            textfont=dict(size=10)))
        fig2.update_layout(**PLOT_LAYOUT,height=320,title="Monthly Split (6mo)",
            showlegend=True,legend=dict(font=dict(size=9),x=0,y=-0.1,orientation="h"))
        st.plotly_chart(fig2,width='stretch')

    # IoT live snapshot
    if st.session_state.iot_data:
        sec("IoT Live Snapshot")
        iot = st.session_state.iot_data; s = iot["summary"]
        ts  = st.session_state.iot_last_fetch
        st.caption(f"Last updated: {ts.strftime('%H:%M:%S')} · Device: {iot['device_id']}")
        kpi5([
            ("Live Power",    f"{s['total_active_power_w']:.0f} W",  "right now",       "⚡"),
            ("Current Draw",  f"{s['total_current_a']:.2f} A",       "total household", "🔋"),
            ("Grid Voltage",  f"{s['grid_voltage_v']:.1f} V",        f"{s['grid_frequency_hz']:.2f} Hz","🔌"),
            ("kWh Today",     f"{s['total_energy_kwh_today']:.3f}",  "since midnight",  "📊"),
            ("Cost Today",    f"₹{s['total_cost_inr']:.2f}",         "@ ₹6/kWh",       "💰"),
        ])

    # Quick stats bottom row
    sec("Period Statistics")
    s1,s2,s3,s4 = st.columns(4)
    with s1:
        st.markdown(f"""<div class="glass-card">
          <div class="stat-label">Peak Day</div>
          <div class="stat-highlight">{df_main['kWh'].max():.1f}</div>
          <div class="stat-label">kWh recorded</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="glass-card">
          <div class="stat-label">Lowest Day</div>
          <div class="stat-highlight">{df_main['kWh'].min():.1f}</div>
          <div class="stat-label">kWh recorded</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        monthly_avg = df_main.groupby(df_main["Date"].dt.to_period("M"))["kWh"].sum().mean()
        st.markdown(f"""<div class="glass-card">
          <div class="stat-label">Monthly Avg</div>
          <div class="stat-highlight">{monthly_avg:.0f}</div>
          <div class="stat-label">kWh per month</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""<div class="glass-card">
          <div class="stat-label">CO₂ / Day</div>
          <div class="stat-highlight">{round(avg_daily*co2_ui,2)}</div>
          <div class="stat-label">kg emitted daily</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE — IoT LIVE FEED
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "iot":
    ph("🔌","IoT Live Feed","Real-time appliance monitoring from your Smart Meter Simulator")

    col_r, col_s = st.columns([1,3])
    with col_r:
        if st.button("⟳  Refresh Now", width='stretch'):
            st.session_state.iot_data     = simulate_iot_mock() if use_mock else None
            st.session_state.iot_last_fetch = datetime.now()
            st.rerun()
    with col_s:
        if st.session_state.iot_last_fetch:
            st.caption(f"📡 Last update: {st.session_state.iot_last_fetch.strftime('%d %b %Y, %H:%M:%S')} · {'Mock Simulator' if use_mock else 'Live API'}")

    iot = st.session_state.iot_data
    s   = iot["summary"]

    kpi5([
        ("Active Power",   f"{s['total_active_power_w']:.1f} W",  "total load",      "⚡"),
        ("Total Current",  f"{s['total_current_a']:.2f} A",       "all appliances",  "🔋"),
        ("Grid Voltage",   f"{s['grid_voltage_v']:.1f} V",        f"{s['grid_frequency_hz']:.2f} Hz","🔌"),
        ("Power Factor",   f"{s['power_factor_avg']:.3f}",        "avg household",   "⚖️"),
        ("Cost Today",     f"₹{s['total_cost_inr']:.3f}",         "kWh today",       "💰"),
    ])

    sec("Appliance Status")
    apps = iot.get("appliances",[])
    cols = st.columns(3)
    for i,a in enumerate(apps):
        status = a["status"].lower()
        dot    = f'<span class="status-dot dot-{status}"></span>'
        pw     = a.get("active_power_w",0)
        with cols[i%3]:
            st.markdown(f"""
            <div class="appliance-card {status}">
              <div style="display:flex;justify-content:space-between;align-items:start">
                <div>
                  <div class="appliance-name">{a['name']}</div>
                  <div class="appliance-room">📍 {a['room']}</div>
                </div>
                <div style="font-size:0.7rem;color:var(--text3)">{dot}{a['status']}</div>
              </div>
              <div class="appliance-power">{pw:.1f} W</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.7rem;color:var(--text3)">
                <span>🔌 {a.get('voltage_v','—')} V</span>
                <span>⚡ {a.get('current_a','—')} A</span>
                <span>⚖️ PF {a.get('power_factor','—')}</span>
                <span>📊 {a.get('energy_kwh','—')} kWh</span>
              </div>
            </div>""", unsafe_allow_html=True)

    # Power pie
    sec("Power Distribution")
    pc1,pc2 = st.columns([1,1])
    on_apps = [a for a in apps if a.get("active_power_w",0)>0]
    if on_apps:
        with pc1:
            fig_p = go.Figure(go.Pie(
                labels=[a["name"] for a in on_apps],
                values=[a["active_power_w"] for a in on_apps],
                hole=0.55,
                marker=dict(colors=["#00FF87","#00CC6A","#00994F","#FFB347","#FF8C42","#60A5FA"]),
                textfont=dict(size=10,color="white")))
            fig_p.update_layout(**PLOT_LAYOUT,height=320,title="Live Power by Appliance",showlegend=True,
                legend=dict(font=dict(size=9),orientation="v",x=1,y=0.5))
            st.plotly_chart(fig_p,width='stretch')
        with pc2:
            # Bar by room
            room_pow = {}
            for a in apps:
                room_pow[a["room"]] = room_pow.get(a["room"],0) + a.get("active_power_w",0)
            fig_r = go.Figure(go.Bar(x=list(room_pow.keys()),y=list(room_pow.values()),
                marker_color="#00FF87",marker_line_width=0))
            fig_r.update_layout(**PLOT_LAYOUT,height=320,title="Power by Room (W)",
                showlegend=False,bargap=0.3)
            st.plotly_chart(fig_r,width='stretch')

    with st.expander("🔍 Raw JSON Payload"):
        st.json(iot)

# ══════════════════════════════════════════════════════════════
#  PAGE — ANALYTICS & EDA
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "eda":
    ph("📈","Analytics & EDA","Deep exploration of your energy data with anomaly detection")

    # Date range filter
    fc1,fc2 = st.columns(2)
    dmin,dmax = df_main["Date"].min().date(), df_main["Date"].max().date()
    with fc1: dfr = st.date_input("From", dmin, dmin, dmax)
    with fc2: dto = st.date_input("To",   dmax, dmin, dmax)
    df_f = df_main[(df_main["Date"]>=pd.Timestamp(dfr))&(df_main["Date"]<=pd.Timestamp(dto))].copy()

    # Anomaly detection
    feats = df_f[["kWh"]].copy()
    feats["roll7"] = df_f["kWh"].rolling(7,min_periods=1).mean()
    df_f["anomaly"] = IsolationForest(contamination=0.05,random_state=42).fit_predict(feats)
    df_f["is_anomaly"] = df_f["anomaly"]==-1
    n_anom = int(df_f["is_anomaly"].sum())

    kpi5([
        ("Period Days",    f"{len(df_f)}",                        "in selected range", "📅"),
        ("Avg Daily",      f"{df_f['kWh'].mean():.2f}",           "kWh / day",         "⚡"),
        ("Std Deviation",  f"{df_f['kWh'].std():.2f}",            "kWh variability",   "📊"),
        ("Anomalies",      str(n_anom),                           "unusual days",      "🔍"),
        ("Above Threshold",f"{(df_f['kWh']>threshold_kwh).sum()}","days over limit",   "⚠️"),
    ])

    # Main line chart with anomalies
    sec("Consumption Timeline")
    df_f["roll7"] = df_f["kWh"].rolling(7,min_periods=1).mean()
    adf = df_f[df_f["is_anomaly"]]
    fig_eda = go.Figure()
    fig_eda.add_trace(go.Scatter(x=df_f["Date"],y=df_f["kWh"],name="Daily kWh",
        line=dict(color="rgba(0,255,135,0.5)",width=1.2),
        fill="tozeroy",fillcolor="rgba(0,255,135,0.04)"))
    fig_eda.add_trace(go.Scatter(x=df_f["Date"],y=df_f["roll7"],name="7-Day Avg",
        line=dict(color="#00FF87",width=2)))
    if not adf.empty:
        fig_eda.add_trace(go.Scatter(x=adf["Date"],y=adf["kWh"],mode="markers",
            name="⚠️ Anomaly",marker=dict(color="#FF5C5C",size=10,symbol="x",line=dict(width=2))))
    fig_eda.add_hline(y=threshold_kwh,line_dash="dash",line_color="#FF5C5C",
        annotation_text="Threshold",annotation_position="top right",annotation_font_color="#FF5C5C")
    fig_eda.update_layout(**PLOT_LAYOUT,height=360,
        title=f"Energy Timeline — {n_anom} anomalies detected",
        legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig_eda,width='stretch')

    if n_anom>0:
        dates_str = ", ".join(adf["Date"].dt.strftime("%d %b").tolist()[:5])
        st.markdown(f"""<div class="alert alert-amber">
          <span class="alert-icon">🔍</span>
          <div><strong>Isolation Forest</strong> found {n_anom} anomalous days: {dates_str}{"…" if n_anom>5 else ""}.
          Possible causes: faulty appliance running overnight, unusual event, or meter error.</div>
        </div>""", unsafe_allow_html=True)

    # 3-col charts
    sec("Distribution Analysis")
    ec1,ec2,ec3 = st.columns(3)

    with ec1:
        fig_h = px.histogram(df_f,x="kWh",nbins=35,color_discrete_sequence=["#00FF87"],
            title="Usage Distribution")
        fig_h.add_vline(x=df_f["kWh"].mean(),line_dash="dash",line_color="#FFB347",
            annotation_text=f"Avg {df_f['kWh'].mean():.1f}",annotation_font_color="#FFB347")
        fig_h.update_layout(**PLOT_LAYOUT,height=280,showlegend=False,bargap=0.05)
        st.plotly_chart(fig_h,width='stretch')

    with ec2:
        df_f["DayName"] = df_f["Date"].dt.day_name()
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow = df_f.groupby("DayName")["kWh"].mean().reindex(order).reset_index()
        fig_d = px.bar(dow,x="DayName",y="kWh",color="kWh",
            color_continuous_scale=["#162B1E","#00994F","#00FF87"],
            title="Avg by Day of Week")
        fig_d.update_layout(**PLOT_LAYOUT,height=280,showlegend=False,
            coloraxis_showscale=False,bargap=0.25)
        st.plotly_chart(fig_d,width='stretch')

    with ec3:
        df_f["MonthLabel"] = df_f["Date"].dt.strftime("%b %y")
        mon = df_f.groupby("MonthLabel")["kWh"].mean().tail(12).reset_index()
        fig_m = px.bar(mon,x="MonthLabel",y="kWh",color="kWh",
            color_continuous_scale=["#162B1E","#00994F","#FFB347","#FF5C5C"],
            title="Monthly Avg (12 mo)")
        fig_m.add_hline(y=threshold_kwh,line_dash="dash",line_color="#FF5C5C")
        fig_m.update_layout(**PLOT_LAYOUT,height=280,showlegend=False,
            coloraxis_showscale=False,bargap=0.25)
        st.plotly_chart(fig_m,width='stretch')

# ══════════════════════════════════════════════════════════════
#  PAGE — ML FORECASTING
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "ml":
    ph("🤖","ML Forecasting",f"Predict energy usage with {model_choice} — {forecast_days}-day horizon")

    with st.spinner(f"Training {model_choice}…"):
        mdl,metrics,df_feat = train_model(df_main, model_choice)
        fc_df = forecast(mdl, df_feat, forecast_days, model_choice)

    kpi5([
        ("Model",       model_choice,             "selected algorithm",   "🤖"),
        ("MAE",         f"{metrics['MAE']} kWh"  if metrics['MAE']>0 else "N/A","mean absolute error","📐"),
        ("R² Score",    f"{metrics['R2']}"        if metrics['R2']>0 else "Prophet","variance explained","📊"),
        ("Forecast",    f"{forecast_days} days",  "ahead predicted",      "🔮"),
        ("Avg Forecast",f"{fc_df['Forecasted_kWh'].mean():.2f} kWh","expected daily","📅"),
    ])

    # Feature importance
    if hasattr(mdl,"feature_importances_"):
        sec("Feature Importance")
        fi = pd.DataFrame({"Feature":FEATURES,"Importance":mdl.feature_importances_}).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(x=fi["Importance"],y=fi["Feature"],orientation="h",
            marker=dict(color=fi["Importance"],colorscale=[[0,"#162B1E"],[0.5,"#00994F"],[1,"#00FF87"]]),
            text=[f"{v:.3f}" for v in fi["Importance"]],textposition="outside",
            textfont=dict(size=9,color="rgba(143,176,154,0.8)")))
        fig_fi.update_layout(**PLOT_LAYOUT,height=300,title="Which features drive predictions most?",
            showlegend=False,xaxis_title="Importance Score")
        st.plotly_chart(fig_fi,width='stretch')

    # Forecast chart
    sec("Forecast Visualization")
    hist30 = df_main.tail(30)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=hist30["Date"],y=hist30["kWh"],name="Historical (30d)",
        line=dict(color="rgba(0,255,135,0.6)",width=1.5),mode="lines+markers",
        marker=dict(size=3,color="#00FF87")))
    fig_fc.add_trace(go.Scatter(
        x=list(fc_df["Date"])+list(fc_df["Date"][::-1]),
        y=list(fc_df["Upper"])+list(fc_df["Lower"][::-1]),
        fill="toself",fillcolor="rgba(255,179,71,0.08)",
        line=dict(color="rgba(0,0,0,0)"),name="Confidence Band"))
    fig_fc.add_trace(go.Scatter(x=fc_df["Date"],y=fc_df["Forecasted_kWh"],
        name=f"Forecast ({forecast_days}d)",
        line=dict(color="#FFB347",width=2.5,dash="dot"),
        mode="lines+markers",marker=dict(size=5,symbol="diamond",color="#FFB347")))
    # Draw "Today" marker as shape (avoids string-date TypeError)
    today_x = df_main["Date"].max()
    fig_fc.add_shape(type="line",
        x0=today_x,x1=today_x,y0=0,y1=1,yref="paper",
        line=dict(color="rgba(255,255,255,0.2)",width=1,dash="dot"))
    fig_fc.add_annotation(x=today_x,y=1,yref="paper",
        text="Today",showarrow=False,
        font=dict(color="rgba(255,255,255,0.4)",size=10),
        xanchor="left",yanchor="top",xshift=4)
    fig_fc.add_hline(y=threshold_kwh,line_dash="dash",line_color="#FF5C5C",
        annotation_text="Threshold",annotation_position="top right",
        annotation_font_color="#FF5C5C")
    fig_fc.update_layout(**PLOT_LAYOUT,height=400,hovermode="x unified",
        title=f"🔮 {forecast_days}-Day Energy Forecast",
        legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig_fc,width='stretch')

    # Forecast table
    sec("Detailed Forecast Table")
    fdisp = fc_df.copy()
    fdisp["Date"]     = fdisp["Date"].dt.strftime("%a, %d %b %Y")
    fdisp["CO₂ (kg)"] = (fdisp["Forecasted_kWh"]*co2_ui).round(3)
    fdisp["Cost (₹)"] = (fdisp["Forecasted_kWh"]*6).round(2)
    fdisp["Status"]   = fdisp["Forecasted_kWh"].apply(lambda x:"🟢 Normal" if x<=threshold_kwh else "🔴 High")
    fdisp.index = range(1,len(fdisp)+1)
    st.dataframe(fdisp[["Date","Forecasted_kWh","Lower","Upper","CO₂ (kg)","Cost (₹)","Status"]],
        width='stretch',height=min(len(fdisp)*38+50,450))

    fs1,fs2,fs3,fs4 = st.columns(4)
    fs1.metric(f"Total ({forecast_days}d)",f"{fc_df['Forecasted_kWh'].sum():.2f} kWh")
    fs2.metric("Avg Forecast",             f"{fc_df['Forecasted_kWh'].mean():.2f} kWh")
    fs3.metric("Est. CO₂",                 f"{fc_df['Forecasted_kWh'].sum()*co2_ui:.2f} kg")
    fs4.metric("⚠️ High Days",             f"{(fc_df['Forecasted_kWh']>threshold_kwh).sum()} / {forecast_days}")

# ══════════════════════════════════════════════════════════════
#  PAGE — APPLIANCE BREAKDOWN
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "appliance":
    ph("🏠","Appliance Breakdown","See exactly which device is draining your wallet")

    # Use IoT data if available
    iot_apps = st.session_state.iot_data.get("appliances",[]) if st.session_state.iot_data else []
    if iot_apps:
        st.markdown("""<div class="alert alert-green">
          <span class="alert-icon">🔌</span>
          <div>Appliances loaded from <strong>IoT Simulator</strong> — showing live readings.</div>
        </div>""", unsafe_allow_html=True)

    types = ["AC","Refrigerator","Washing Machine","Geyser","TV","Fan","Light","Microwave","Computer","Other"]
    sec("Configure Appliances")
    n = st.number_input("Number of appliances",1,15,6)
    app_data = []
    cols_a = st.columns(2)
    for i in range(int(n)):
        with cols_a[i%2]:
            with st.expander(f"Appliance {i+1}", expanded=i<2):
                c1,c2,c3,c4 = st.columns(4)
                nm  = c1.text_input("Name",  value=f"Device {i+1}",  key=f"n{i}")
                tp  = c2.selectbox("Type",   types,                   key=f"t{i}")
                wt  = c3.number_input("Watts",5,5000,200,             key=f"w{i}")
                hr  = c4.number_input("Hrs/day",0.0,24.0,4.0,0.5,    key=f"h{i}")
                app_data.append({"name":nm,"type":tp,"watts":wt,"hours":hr,
                    "daily_kwh":round(wt*hr/1000,3),"monthly_kwh":round(wt*hr*30/1000,2),
                    "cost_mo":round(wt*hr*30/1000*6,2),"co2_mo":round(wt*hr*30/1000*co2_ui,2)})

    if app_data:
        sec("Consumption Analysis")
        df_ap = pd.DataFrame(app_data)
        tot   = df_ap["monthly_kwh"].sum()
        df_ap["pct"] = (df_ap["monthly_kwh"]/max(tot,0.01)*100).round(1)

        ac1,ac2 = st.columns([1,1])
        with ac1:
            fig_pie = go.Figure(go.Pie(
                labels=df_ap["name"],values=df_ap["monthly_kwh"],hole=0.55,
                marker=dict(colors=["#00FF87","#00CC6A","#00994F","#007A3D","#FFB347","#FF8C42"]),
                textfont=dict(size=9,color="white"),
                hovertemplate="<b>%{label}</b><br>%{value:.2f} kWh<br>%{percent}<extra></extra>"))
            fig_pie.update_layout(**PLOT_LAYOUT,height=320,title="Monthly kWh Share",
                legend=dict(font=dict(size=9),orientation="v",x=1.02,y=0.5))
            st.plotly_chart(fig_pie,width='stretch')

        with ac2:
            fig_bar = go.Figure(go.Bar(
                x=df_ap["name"],y=df_ap["cost_mo"],
                marker=dict(color=df_ap["cost_mo"],
                    colorscale=[[0,"#162B1E"],[0.5,"#00994F"],[1,"#FF5C5C"]]),
                text=[f"₹{v}" for v in df_ap["cost_mo"]],
                textposition="outside",textfont=dict(size=9)))
            fig_bar.update_layout(**PLOT_LAYOUT,height=320,title="Monthly Cost by Appliance (₹)",
                showlegend=False,coloraxis_showscale=False,bargap=0.3)
            st.plotly_chart(fig_bar,width='stretch')

        sec("Full Breakdown Table")
        st.dataframe(df_ap[["name","type","watts","hours","daily_kwh","monthly_kwh","cost_mo","co2_mo","pct"]].rename(
            columns={"name":"Appliance","type":"Type","watts":"Watts","hours":"Hrs/Day",
                     "daily_kwh":"Daily kWh","monthly_kwh":"Monthly kWh",
                     "cost_mo":"Cost ₹/mo","co2_mo":"CO₂ kg/mo","pct":"% Share"}),
            width='stretch')

        sec("Saving Recommendations")
        for _,r in df_ap.nlargest(3,"monthly_kwh").iterrows():
            sv = round(r["monthly_kwh"]*0.20*6,0)
            st.markdown(f"""<div class="alert alert-amber">
              <span class="alert-icon">💡</span>
              <div><strong>{r['name']}</strong> consumes {r['monthly_kwh']} kWh/month ({r['pct']}% of total).
              A 20% reduction saves ~<strong>₹{sv:.0f}/month</strong> and
              {round(r['monthly_kwh']*0.20*co2_ui,1)} kg CO₂. Consider smart scheduling or a 5-star replacement.</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE — BILL PREDICTOR
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "bill":
    ph("💰","Bill Predictor","Exact Gujarat DGVCL/UGVCL slab-wise electricity bill calculator")

    bc1,bc2 = st.columns([1,1])
    with bc1:
        sec("Input")
        method = st.radio("Estimate from",["ML Forecast","Manual Input"],horizontal=True)
        if method=="Manual Input":
            units = st.number_input("Expected units this month (kWh)",0,2000,250)
        else:
            with st.spinner("Forecasting 30 days…"):
                m_,_,df_ = train_model(df_main,"Random Forest")
                fc30     = forecast(m_,df_,30,"Random Forest")
            units = round(fc30["Forecasted_kWh"].sum(),1)
            st.info(f"ML forecast: **{units} kWh** next 30 days")
        discom = st.selectbox("DISCOM",["DGVCL","UGVCL","MGVCL","PGVCL"])

    with bc2:
        bill_total, breakdown = calc_bill(units)
        score_c = "#00FF87" if bill_total<500 else "#FFB347" if bill_total<1000 else "#FF5C5C"
        st.markdown(f"""<div class="score-big">
          <div class="stat-label">Estimated Monthly Bill — {discom}</div>
          <div class="score-number" style="color:{score_c}">₹{bill_total:,.2f}</div>
          <div style="color:var(--text3);font-size:0.8rem;margin-top:0.5rem">For {units} kWh consumed</div>
        </div>""", unsafe_allow_html=True)

    # Slab table
    sec("Tariff Slab Breakdown")
    st.markdown("""<table class="slab-table"><thead>
      <tr><th>Slab</th><th>Units</th><th>Rate ₹/unit</th><th>Charge ₹</th></tr></thead><tbody>""",
      unsafe_allow_html=True)
    for row in breakdown:
        active_cls = "slab-active" if isinstance(row["Units"],(int,float)) and row["Units"]>0 else ""
        st.markdown(f"""<tr class="{active_cls}">
          <td>{row['Slab']}</td><td>{row['Units']}</td>
          <td>{row['Rate']}</td><td>₹{row['Charge ₹']}</td></tr>""",
          unsafe_allow_html=True)
    st.markdown("</tbody></table>",unsafe_allow_html=True)

    # Slab alert
    for i,sl in enumerate(GUJARAT_TARIFF[:-1]):
        gap = GUJARAT_TARIFF[i+1]["upto"]-units
        if 0<gap<30:
            st.markdown(f"""<div class="alert alert-amber">
              <span class="alert-icon">⚠️</span>
              <div>Only <strong>{gap:.0f} kWh</strong> away from next slab
              (₹{GUJARAT_TARIFF[i+1]['rate']}/unit).
              Reduce by {gap:.0f} kWh to save ~₹{round(gap*(GUJARAT_TARIFF[i+1]['rate']-sl['rate']),0):.0f}.</div>
            </div>""", unsafe_allow_html=True)
            break

    # Historical bill chart
    if len(df_main)>=60:
        sec("Monthly Bill History")
        dm = df_main.copy()
        dm["Month"] = dm["Date"].dt.to_period("M")
        mdata = dm.groupby("Month")["kWh"].sum().reset_index()
        mdata["Bill ₹"] = mdata["kWh"].apply(lambda x: calc_bill(x)[0])
        mdata["Month"]  = mdata["Month"].astype(str)
        fig_mb = px.area(mdata.tail(18),x="Month",y="Bill ₹",
            color_discrete_sequence=["#00FF87"],title="Monthly Bills (₹)")
        fig_mb.update_traces(fill="tozeroy",fillcolor="rgba(0,255,135,0.06)",
            line=dict(color="#00FF87",width=2))
        fig_mb.update_layout(**PLOT_LAYOUT,height=280)
        st.plotly_chart(fig_mb,width='stretch')

# ══════════════════════════════════════════════════════════════
#  PAGE — SOLAR ADVISOR
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "solar":
    ph("☀️","Solar Advisor","Calculate your rooftop solar ROI with PM Surya Ghar subsidy")

    sc1,sc2 = st.columns([1,1])
    with sc1:
        sec("System Parameters")
        city         = st.selectbox("📍 City",list(SOLAR_IRRADIANCE.keys()))
        panel_kw     = st.slider("Panel Capacity (kW)",1.0,10.0,2.0,0.5)
        install_cost = st.number_input("Installation Cost (₹)",50000,1000000,90000,5000)
        tariff_r     = st.number_input("Your Tariff (₹/kWh)",1.0,10.0,6.0,0.25)
        pm_sub       = st.checkbox("PM Surya Ghar Yojana Subsidy",value=True)

    irr   = SOLAR_IRRADIANCE[city]
    ann   = round(panel_kw*irr*0.80*365,0)
    mo_g  = round(ann/12,1)
    ann_s = round(ann*tariff_r,0)
    mo_s  = round(ann_s/12,0)
    sub   = (30000 if panel_kw<=2 else 60000 if panel_kw<=3 else 78000) if pm_sub else 0
    net   = install_cost-sub
    pb    = round(net/ann_s,1) if ann_s>0 else 99
    co2os = round(ann*co2_ui,0)

    with sc2:
        st.markdown(f"""<div class="solar-hero">
          <div style="font-size:2rem">☀️</div>
          <div style="font-family:'Syne',sans-serif;font-size:0.7rem;color:var(--text3);
            text-transform:uppercase;letter-spacing:0.15em;margin:0.4rem 0">Annual Generation</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:3rem;color:var(--green);
            text-shadow:0 0 30px rgba(0,255,135,0.3);line-height:1">{ann:,.0f}</div>
          <div style="color:var(--text3);font-size:0.8rem">kWh / year · {mo_g} kWh/month</div>
          {'<div style="background:rgba(0,255,135,0.08);border:1px solid var(--border2);border-radius:10px;padding:0.6rem;margin-top:1rem;font-size:0.8rem;color:var(--green)">🏛️ PM Surya Ghar Subsidy: ₹'+f"{sub:,}</div>" if pm_sub and sub>0 else ""}
        </div>""", unsafe_allow_html=True)

    kpi5([
        ("Monthly Savings", f"₹{mo_s:,.0f}",   "per month",        "💰"),
        ("Annual Savings",  f"₹{ann_s:,.0f}",   "per year",         "📈"),
        ("Net Investment",  f"₹{net:,}",         "after subsidy",    "🏛️"),
        ("Payback Period",  f"{pb} yrs",          "break-even",       "⏱️"),
        ("CO₂ Offset",      f"{co2os:,.0f} kg",  "per year",         "🌳"),
    ])

    # ROI timeline
    sec("Return on Investment Timeline")
    yrs  = list(range(0,21))
    sav  = [min(y*ann_s,999999) for y in yrs]
    inv  = [net]*len(yrs)
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=yrs,y=sav,name="Cumulative Savings",
        line=dict(color="#00FF87",width=2.5),fill="tozeroy",
        fillcolor="rgba(0,255,135,0.06)"))
    fig_r.add_trace(go.Scatter(x=yrs,y=inv,name=f"Net Investment ₹{net:,}",
        line=dict(color="#FFB347",width=2,dash="dash")))
    if pb<=20:
        fig_r.add_vline(x=pb,line_dash="dot",line_color="#FF5C5C",
            annotation_text=f"Payback @ {pb} yr",
            annotation_font_color="#FF5C5C",annotation_position="top right")
    fig_r.update_layout(**PLOT_LAYOUT,height=360,title="Solar ROI Timeline (₹)",
        xaxis_title="Year",yaxis_title="₹",legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig_r,width='stretch')

    trees = int(co2os/21)
    st.markdown(f"""<div class="alert alert-green">
      <span class="alert-icon">🌳</span>
      <div>Your {panel_kw}kW system offsets <strong>{co2os:,.0f} kg CO₂/year</strong> —
      equivalent to planting <strong>{trees} trees</strong> annually!
      Gujarat's <strong>{irr} kWh/m²/day</strong> irradiance makes it ideal for solar.
      Apply for subsidy at <em>pmsuryaghar.gov.in</em></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE — GAMIFICATION
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "game":
    ph("🏆","Gamification","Your sustainability journey — scores, streaks & achievements")

    gc1,gc2 = st.columns([1,2])
    with gc1:
        clr = score_color
        st.markdown(f"""<div class="score-big">
          <div style="font-size:0.68rem;color:var(--text3);text-transform:uppercase;
            letter-spacing:0.15em;margin-bottom:0.5rem">Sustainability Score</div>
          <div class="score-number" style="color:{clr};text-shadow:0 0 40px {clr}44">{score}</div>
          <div class="score-grade" style="color:{clr}">Grade {grade}</div>
          <div style="color:var(--text3);font-size:0.78rem;margin-top:0.8rem">
            {avg_daily:.2f} kWh/day · target {threshold_kwh} kWh</div>
        </div>""", unsafe_allow_html=True)

    with gc2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",value=score,
            gauge=dict(axis=dict(range=[0,100],tickcolor="rgba(143,176,154,0.3)"),
                bar=dict(color=score_color),
                bgcolor="rgba(12,24,16,0.8)",
                steps=[{"range":[0,30],"color":"rgba(204,0,0,0.15)"},
                       {"range":[30,50],"color":"rgba(255,92,92,0.1)"},
                       {"range":[50,70],"color":"rgba(255,179,71,0.1)"},
                       {"range":[70,85],"color":"rgba(0,153,79,0.12)"},
                       {"range":[85,100],"color":"rgba(0,255,135,0.12)"}],
                threshold=dict(line=dict(color="white",width=2),value=score)),
            title=dict(text="Eco Score",font=dict(color="rgba(143,176,154,0.8)",size=13)),
            number=dict(font=dict(color=score_color,size=36))))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(color="#8FB09A"),
            height=240,margin=dict(l=20,r=20,t=40,b=10))
        st.plotly_chart(fig_g,width='stretch')

    # Streaks
    sec("Streaks & Progress")
    r30    = df_main.tail(30)
    green  = int((r30["kWh"]<=threshold_kwh).sum())
    streak = int((r30["kWh"]<=threshold_kwh)[::-1].cumprod().sum())
    best_m = df_main.groupby(df_main["Date"].dt.to_period("M"))["kWh"].sum().min()

    kpi5([
        ("Current Streak",  f"{streak} days",    f"under {threshold_kwh} kWh",   "🔥"),
        ("Green Days (30d)",f"{green} / 30",      f"{round(green/30*100)}% rate", "✅"),
        ("Best Month",      f"{best_m:.0f} kWh",  "personal record",              "🏅"),
        ("vs India Avg",    f"{round((INDIA_AVG_KWH-avg_daily)/INDIA_AVG_KWH*100,1)}%","{'below' if avg_daily<INDIA_AVG_KWH else 'above'} avg","🇮🇳"),
        ("CO₂ Saved",       f"{max(0,round((INDIA_AVG_KWH-avg_daily)*30*co2_ui,1))} kg","vs national avg","🌍"),
    ])

    if streak>=7:
        st.markdown(f"""<div class="alert alert-green">
          <span class="alert-icon">🔥</span>
          <div><strong>{streak}-day streak!</strong> Outstanding — you've stayed under {threshold_kwh} kWh for {streak} consecutive days! You're saving ₹{round(streak*(threshold_kwh-avg_daily)*6,0):.0f} vs threshold.</div>
        </div>""", unsafe_allow_html=True)

    # Benchmark chart
    sec("Benchmark Comparison")
    bench = pd.DataFrame({
        "Category":["You","India Avg","Your Target","Ideal Eco"],
        "kWh/day":[avg_daily,INDIA_AVG_KWH,threshold_kwh,threshold_kwh*0.65],
        "Color":["#FFB347","#FF5C5C","#00CC6A","#00FF87"]})
    fig_b = go.Figure()
    for _,row in bench.iterrows():
        fig_b.add_trace(go.Bar(x=[row["Category"]],y=[row["kWh/day"]],
            marker_color=row["Color"],name=row["Category"],
            text=f"{row['kWh/day']:.2f}",textposition="outside",
            textfont=dict(size=10)))
    fig_b.update_layout(**PLOT_LAYOUT,height=280,showlegend=False,
        title="Daily kWh Benchmark",yaxis_title="kWh/day",bargap=0.4)
    st.plotly_chart(fig_b,width='stretch')

    # Badges
    sec("Achievement Badges")
    badges = [
        ("🌱","First Step",   True,                                 "Started your journey"),
        ("⚡","Power Saver",  avg_daily<=threshold_kwh,             "Below daily threshold"),
        ("🔥","Week Warrior", streak>=7,                             "7-day green streak"),
        ("🌍","CO₂ Fighter",  total_co2<500,                        "Under 500 kg total"),
        ("📊","Data Nerd",    len(df_main)>=365,                    "1 year tracked"),
        ("☀️","Solar Scout",  False,                                "Set up solar system"),
        ("🏆","Eco Champion", grade in ["A","A+"],                  "Achieved grade A+"),
        ("💰","Budget Boss",  False,                                "Under budget 3 months"),
    ]
    bc = st.columns(4)
    for i,(icon,name,earned,desc) in enumerate(badges):
        with bc[i%4]:
            cls = "badge-card earned" if earned else "badge-card"
            st.markdown(f"""<div class="{cls}">
              <div class="badge-icon">{icon}</div>
              <div class="badge-name">{name}</div>
              <div class="badge-desc">{desc}</div>
              {"<div class='badge-status'>✓ EARNED</div>" if earned else ""}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE — AI CHAT
# ══════════════════════════════════════════════════════════════
elif ACTIVE == "chat":
    ph("🗣️","AI Energy Advisor","Ask me anything about your energy usage, bills, or sustainability")

    top_app = "your AC"
    if st.session_state.iot_data:
        apps = st.session_state.iot_data.get("appliances",[])
        if apps:
            top_app = max(apps,key=lambda a:a.get("active_power_w",0))["name"]

    # Build rich context from real loaded data
    r30       = df_main.tail(30)
    streak_c  = int((r30["kWh"]<=threshold_kwh)[::-1].cumprod().sum())
    green_d   = int((r30["kWh"]<=threshold_kwh).sum())
    mo_kwh    = round(avg_daily*30,1)
    est_bill_ai = calc_bill(mo_kwh)[0]
    data_source = "uploaded CSV" if upload else "IoT live feed" if st.session_state.iot_data else "mock demo data"
    ai_ctx = {
        "avg":     avg_daily,
        "grade":   grade,
        "score":   score,
        "thr":     threshold_kwh,
        "top":     top_app,
        "co2":     total_co2,
        "total":   total_kwh,
        "days":    len(df_main),
        "peak":    round(df_main["kWh"].max(),2),
        "lowest":  round(df_main["kWh"].min(),2),
        "monthly_avg": mo_kwh,
        "est_bill":    est_bill_ai,
        "streak":      streak_c,
        "green_days":  green_d,
        "data_src":    data_source,
    }

    # Quick suggestion buttons — 2×3 grid
    suggestions = [
        "What is my average daily usage?",
        "Why was my usage high recently?",
        "How can I save ₹200/month?",
        "What is my carbon footprint?",
        "Should I invest in solar?",
        "How do I improve my eco grade?",
    ]
    sec("💡 Quick Questions — click to ask instantly")
    pr1 = st.columns(3)
    pr2 = st.columns(3)
    for i, (col, q) in enumerate(zip(pr1+pr2, suggestions)):
        with col:
            if st.button(q, key=f"pill_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role":"user","content":q})
                with st.spinner("EcoMinds AI thinking…"):
                    ans = ask_ai(q, ai_ctx)
                st.session_state.chat_history.append({"role":"ai","content":ans})
                st.rerun()
    # Chat window
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2.5rem">
          <div style="font-size:3rem;margin-bottom:0.8rem">🤖</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:var(--text);font-weight:700">
            EcoMinds AI</div>
          <div style="color:var(--text2);font-size:0.85rem;margin-top:0.4rem">
            Your personal energy advisor. Ask me about usage patterns,<br>
            cost saving tips, carbon footprint, or solar investments.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="glass-card"><div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"]=="user":
                st.markdown(f"""<div class="chat-msg user">
                  <div class="chat-avatar user">👤</div>
                  <div class="chat-bubble user">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-msg">
                  <div class="chat-avatar ai">🌿</div>
                  <div class="chat-bubble ai">{msg["content"]}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Input row
    st.markdown("<br>", unsafe_allow_html=True)
    inp_c, btn_c = st.columns([5,1])
    with inp_c:
        user_q = st.text_input("Message",placeholder="Ask EcoMinds AI anything…",
            label_visibility="collapsed",key="chat_input")
    with btn_c:
        send = st.button("Send →",width='stretch')

    if send and user_q.strip():
        st.session_state.chat_history.append({"role":"user","content":user_q})
        with st.spinner("Thinking…"):
            ans = ask_ai(user_q,ai_ctx)
        st.session_state.chat_history.append({"role":"ai","content":ans})
        st.rerun()

    cc1,cc2 = st.columns([1,5])
    with cc1:
        if st.button("🗑️ Clear",width='stretch'):
            st.session_state.chat_history=[]
            st.rerun()
    with cc2:
        if not CLAUDE_OK:
            st.caption("ℹ️ Using smart rule-based responses. Add ANTHROPIC_API_KEY to secrets.toml for real Claude AI.")

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:3rem;padding:1.2rem;border-top:1px solid var(--border);
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem">
  <div style="font-family:'Syne',sans-serif;font-weight:700;color:var(--green);font-size:0.9rem">
    🌿 EcoMinds v2.0</div>
  <div style="font-size:0.72rem;color:var(--text3)">
    GTU Final Year Project 2024–25 · IoT · ML · Solar · AI</div>
  <div style="font-size:0.7rem;color:var(--text3);font-style:italic">
    "The greatest threat to our planet is the belief that someone else will save it."</div>
</div>
""", unsafe_allow_html=True)