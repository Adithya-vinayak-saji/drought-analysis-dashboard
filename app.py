import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from prophet import Prophet
from scipy.signal import find_peaks
import os
import streamlit.components.v1 as components 

# --- API KEY SECURITY ---
# This checks Streamlit Cloud Secrets first. If not found, it stays empty.
if "GEMINI_API_KEY" in st.secrets:
    HARDCODED_GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
else:
    # Leave this empty here for security when pushing to GitHub
    HARDCODED_GEMINI_KEY = "" 

# --- INITIALIZATION & STYLING ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Pro", page_icon="🛰️")

os.environ['STREAMLIT_DATAFRAME_ENCODING'] = 'legacy'

# Inject Custom CSS for a "Premium" look
st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00d4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e293b; border-radius: 4px; color: white; padding: 10px;
    }
    .element-container { width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

if 'saved_key' not in st.session_state:
    st.session_state.saved_key = ""

# --- CORE MATH ENGINE ---
def calculate_simplified_pet(temp, wind, humidity):
    es = 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
    ea = es * (humidity / 100)
    vpd = es - ea 
    pet = (0.0023 * (temp + 17.8) * (vpd**0.5) * (1 + 0.05 * wind)) * 10 
    return pet

def get_longest_duration(series, threshold=-1.0):
    is_dry = series <= threshold
    if not is_dry.any(): return 0
    return (is_dry.groupby((~is_dry).cumsum()).cumcount()).max()

def make_detailed_chart(data, column, label, unit, color):
    return alt.Chart(data).mark_line(color=color, size=2).encode(
        x=alt.X('Combined_Date:T', title='Timeline'),
        y=alt.Y(f'{column}:Q', title=f"{label} ({unit})", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('Combined_Date:T', format='%B %Y', title='Date'),
            alt.Tooltip(f'{column}:Q', format='.2f', title=f"{label} ({unit})")
        ]
    ).properties(height=450).interactive()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛰️ Drought Control")
    
    if HARDCODED_GEMINI_KEY.strip():
        st.session_state.saved_key = HARDCODED_GEMINI_KEY
        st.success("🔒 Using Secure API Key")
    else:
        input_key = st.text_input("Gemini API Key", value=st.session_state.saved_key, type="password")
        if st.checkbox("Remember API Key", value=bool(st.session_state.saved_key)):
            st.session_state.saved_key = input_key
    
    if st.button("🔍 Check API Connection"):
        if st.session_state.saved_key:
            try:
                genai.configure(api_key=st.session_state.saved_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success(f"Connected! Available: {len(models)} models")
            except Exception as e:
                st.error(f"Failed: {e}")
        else:
            st.warning("Please enter an API Key first.")
            
    st.subheader("📊 Chart Controls")
    zoom_range = st.slider("Focus on Deficit Range (Y-Axis)", min_value=-150, max_value=500, value=(-5, 10))
    chart_height = st.number_input("Main Chart Height", min_value=300, max_value=800, value=450)

    st.divider()
    uploaded_file = st.file_uploader("Upload Climate Data (Excel)", type=["xlsx"])
    selected_sheet = None
    if uploaded_file:
        xl = pd.ExcelFile(uploaded_file)
        selected_sheet = st.selectbox("Select the Sheet", xl.sheet_names)

# --- MAIN LOGIC ---
if uploaded_file and selected_sheet:
    df_raw = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    st.sidebar.subheader("Variable Mapping")
    all_cols = df_raw.columns.tolist()
    year_col = st.sidebar.selectbox("Year", all_cols, index=0)
    month_col = st.sidebar.selectbox("Month", all_cols, index=1)
    p_col = st.sidebar.selectbox("Precipitation", all_cols, index=4)
    t_col = st.sidebar.selectbox("Temperature", all_cols, index=2)
    w_col = st.sidebar.selectbox("Wind Speed", all_cols, index=5)
    h_col = st.sidebar.selectbox("Humidity", all_cols, index=3)

    df = df_raw.copy()
    try:
        df['Combined_Date'] = pd.to_datetime(df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01')
        for col in [p_col, t_col, w_col, h_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')
    except Exception:
        st.warning("⚠️ Invalid data format. Please select a sheet containing raw climate data.")
        st.stop()

    # --- SPEI CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['Status'] = np.where(df['D'] >= 0, 'Surplus', 'Deficit')
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    mk_res = mk.original_test(df['SPEI_Proxy'])
    max_duration = get_longest_duration(df['SPEI_Proxy'])

    # --- PROPHET FORECAST ---
    df_p_reg = df[['Combined_Date', 'SPEI_Proxy', t_col]].rename(columns={'Combined_Date': 'ds', 'SPEI_Proxy': 'y', t_col: 'temp'})
    m_reg = Prophet(yearly_seasonality=True, interval_width=0.95)
    m_reg.add_regressor('temp')
    m_reg.fit(df_p_reg)
    
    future_reg = m_reg.make_future_dataframe(periods=12, freq='MS')
    future_reg = future_reg.merge(df_p_reg[['ds', 'temp']], on='ds', how='left')
    monthly_avg = df_p_reg.groupby(df_p_reg['ds'].dt.month)['temp'].mean()
    future_reg['temp'] = future_reg.apply(lambda x: x['temp'] if pd.notnull(x['temp']) else monthly_avg[x['ds'].month], axis=1)
    
    forecast_reg = m_reg.predict(future_reg).rename(columns={
        'ds': 'Date', 'yhat': 'Predicted_SPEI', 'yhat_lower': 'Lower_Confidence', 'yhat_upper': 'Upper_Confidence'
    })
    future_only = forecast_reg[forecast_reg['Date'] > df_p_reg['ds'].max()]

    # --- UI LAYOUT ---
    st.title(f"🛰️ SPEI Intelligence: {selected_sheet}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trend Direction", mk_res.trend.upper(), delta=f"{mk_res.slope:.4f}")
    m2.metric("Max Dry Streak", f"{max_duration} Mo")
    m3.metric("Extreme Events", len(df[df['SPEI_Proxy'] <= -2.0]))
    m4.metric("Current Status", "DRYING" if df['SPEI_Proxy'].iloc[-1] < -1 else "STABLE")

    tabs = st.tabs(["🌡️ Temp", "🌧️ Precip", "💨 Wind", "💧 Humid", "📈 Water", "📉 SPEI", "🌡️ Heatmap", "📊 Stats", "📊 Timeline", "🔮 Forecast", "🤖 AI"])

    with tabs[0]: st.altair_chart(make_detailed_chart(df, t_col, "Temperature", "°C", "#FF4B4B"), use_container_width=True)
    with tabs[1]: st.altair_chart(make_detailed_chart(df, p_col, "Precipitation", "mm", "#00D4FF"), use_container_width=True)
    with tabs[2]: st.altair_chart(make_detailed_chart(df, w_col, "Wind Speed", "m/s", "#94A3B8"), use_container_width=True)
    with tabs[3]: st.altair_chart(make_detailed_chart(df, h_col, "Humidity", "%", "#00C853"), use_container_width=True)

    with tabs[4]:
        brush = alt.selection_interval(encodings=['x'])
        main_c = alt.Chart(df).mark_bar().encode(
            x=alt.X('Combined_Date:T', scale=alt.Scale(domain=brush)),
            y=alt.Y('D:Q', scale=alt.Scale(domain=zoom_range)),
            color=alt.Color('Status:N', scale=alt.Scale(domain=['Surplus', 'Deficit'], range=['#00d4ff', '#ff4b4b']))
        ).properties(height=chart_height)
        nav_c = alt.Chart(df).mark_area().encode(x='Combined_Date:T', y=alt.Y('D:Q', axis=None)).properties(height=80).add_params(brush)
        st.altair_chart(alt.vconcat(main_c, nav_c), use_container_width=True)

    with tabs[5]:
        line = alt.Chart(df).mark_line(color='#00d4ff').encode(x='Combined_Date:T', y='SPEI_Proxy:Q')
        st.altair_chart(line.properties(height=500).interactive(), use_container_width=True)

    with tabs[6]:
        df['Year'] = df['Combined_Date'].dt.year
        df['Month'] = df['Combined_Date'].dt.month_name().str[:3]
        heat = alt.Chart(df).mark_rect().encode(
            x=alt.X('Month:N', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
            y='Year:O', color=alt.Color('SPEI_Proxy:Q', scale=alt.Scale(scheme='redblue', domain=[-3, 3]))
        )
        st.altair_chart(heat.properties(height=500), use_container_width=True)

    with tabs[7]:
        st.subheader("🧪 Mann-Kendall Trend Analysis")
        report_color = "#ff4b4b" if (mk_res.p < 0.05 and mk_res.slope < 0) else "#00c853"
        st.markdown(f"<div style='padding:15px; border-radius:10px; border: 2px solid {report_color};'><h3 style='color:{report_color};'>{mk_res.trend.upper()}</h3><p style='color: var(--text-color);'>Slope: {mk_res.slope:.6f} | P-Value: {mk_res.p:.5f}</p></div>", unsafe_allow_html=True)

    with tabs[8]:
        hist = df_p_reg.rename(columns={'ds': 'Date', 'y': 'Val'}); hist['Type'] = 'Historical'
        pred = future_only[['Date', 'Predicted_SPEI']].rename(columns={'Predicted_SPEI': 'Val'}); pred['Type'] = 'Predicted'
        st.altair_chart(alt.Chart(pd.concat([hist, pred])).mark_line().encode(x='Date:T', y='Val:Q', color='Type:N'), use_container_width=True)

    with tabs[9]:
        line = alt.Chart(future_only).mark_line(color='#00d4ff').encode(x='Date:T', y='Predicted_SPEI:Q')
        band = alt.Chart(future_only).mark_area(opacity=0.2).encode(x='Date:T', y='Lower_Confidence:Q', y2='Upper_Confidence:Q')
        st.altair_chart((band + line).properties(height=450).interactive(), use_container_width=True)

    with tabs[10]:
        if st.button("Generate AI Assessment"):
            if st.session_state.saved_key:
                genai.configure(api_key=st.session_state.saved_key)
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"Analyze risk: MK Trend {mk_res.trend}, Slope {mk_res.slope:.6f}, Max Dry Streak {max_duration}mo. Forecast avg SPEI: {future_only['Predicted_SPEI'].mean():.2f}."
                with st.spinner("AI is thinking..."):
                    st.markdown(model.generate_content(prompt).text)

else:
    st.markdown("<h2 style='text-align: center;'>🛰️ Welcome to Drought Intelligence Pro</h2>", unsafe_allow_html=True)
    interactive_html = """
    <canvas id="c"></canvas>
    <script>
        const canvas = document.getElementById('c'); const ctx = canvas.getContext('2d');
        let w = canvas.width = window.innerWidth; let h = canvas.height = 400;
        let p = Array.from({length: 80}, () => ({x: Math.random()*w, y: Math.random()*h, vx: Math.random()-0.5, vy: Math.random()-0.5}));
        function draw() {
            ctx.clearRect(0,0,w,h); ctx.fillStyle = 'rgba(0,212,255,0.5)';
            p.forEach(i => {
                i.x += i.vx; i.y += i.vy; if(i.x<0||i.x>w) i.vx*=-1; if(i.y<0||i.y>h) i.vy*=-1;
                ctx.beginPath(); ctx.arc(i.x, i.y, 2, 0, 7); ctx.fill();
            });
            requestAnimationFrame(draw);
        } draw();
    </script>
    """
    components.html(interactive_html, height=400)