import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from prophet import Prophet 

# --- INITIALIZATION & STYLING ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Pro", page_icon="🛰️")

# Inject Custom CSS for a "Premium" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00d4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e293b; border-radius: 4px; color: white; padding: 10px;
    }
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛰️ Drought Control")
    input_key = st.text_input("Gemini API Key", value=st.session_state.saved_key, type="password")
    if st.checkbox("Remember API Key", value=bool(st.session_state.saved_key)):
        st.session_state.saved_key = input_key
    
    if st.button("🔍 Check API Connection"):
        if input_key:
            try:
                genai.configure(api_key=input_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success(f"Connected! Available: {len(models)} models")
            except Exception as e:
                st.error(f"Failed: {e}")
    
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

    # --- DATA PROCESSING ---
    df = df_raw.copy()
    df['Combined_Date'] = pd.to_datetime(df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01')
    for col in [p_col, t_col, w_col, h_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')

    # --- SPEI CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    df['SPEI_Rolling'] = df['SPEI_Proxy'].rolling(window=12, center=True).mean()
    
    extreme_count = len(df[df['SPEI_Proxy'] <= -2.0])
    max_duration = get_longest_duration(df['SPEI_Proxy'])
    mk_res = mk.original_test(df['SPEI_Proxy'])

    st.title(f"🛰️ SPEI Intelligence: {selected_sheet}")
    st.caption("Advanced Hydrological Analytics & Predictive Modeling")
    
    # --- METRICS (Updated with Delta) ---
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trend Direction", mk_res.trend.upper(), delta=f"{mk_res.slope:.6f} units/mo")
    with m2: st.metric("Max Streak", f"{max_duration} Months") 
    with m3: st.metric("Extreme Events", extreme_count)
    with m4: st.metric("Current Status", "MODERATE" if df['SPEI_Proxy'].iloc[-1] < -1 else "STABLE")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Water Dynamics", "🌡️ Intensity Heatmap", "📊 MK Statistics", "🤖 AI Brief", "🔮 Prophet Forecast"
    ])

    danger_lines = alt.Chart(pd.DataFrame({'y': [-1.5, -2.0], 'label': ['Severe', 'Extreme']})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q', tooltip='label:N')

    with tab1:
        st.subheader("Hydrological Balance ($D = P - PET$)")
        # Smooth Area Chart instead of jagged bars
        balance_chart = alt.Chart(df).mark_area(opacity=0.7).encode(
            x=alt.X('Combined_Date:T', title='Timeline'),
            y=alt.Y('D:Q', title='Water Balance'),
            color=alt.condition(alt.datum.D > 0, alt.value('#00d4ff'), alt.value('#ff4b4b')),
            tooltip=['Combined_Date', 'D']
        ).properties(height=400).interactive()
        
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='white', strokeWidth=1, opacity=0.5).encode(y='y')
        st.altair_chart(balance_chart + zero_line, use_container_width=True)

    with tab2:
        st.subheader("Decadal Drought Intensity")
        # Year-Month Heatmap
        df['Year'] = df['Combined_Date'].dt.year
        df['Month_Name'] = df['Combined_Date'].dt.month_name().str[:3]
        
        heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('Month_Name:N', title='Month', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
            y=alt.Y('Year:O', title='Year', axis=alt.Axis(values=list(range(df['Year'].min(), df['Year'].max(), 5)))),
            color=alt.Color('SPEI_Proxy:Q', scale=alt.Scale(scheme='redblue', domain=[-3, 3], reverse=False), title="SPEI"),
            tooltip=['Year', 'Month_Name', 'SPEI_Proxy']
        ).properties(height=500)
        st.altair_chart(heatmap, use_container_width=True)

    with tab3:
        st.subheader("🧪 Mann-Kendall Trend Analysis")
        st.markdown("The Mann-Kendall test determines if there is a monotonic upward or downward trend, robust against climate outliers.")

        # Check if mk_res exists and has data
    if mk_res:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("$p$-value", f"{mk_res.p:.5f}")
        with c2:
            st.metric("$z$-score", f"{mk_res.z:.4f}")
        with c3:
            # The library uses .tau for the original_test
            st.metric("Tau ($\\tau$)", f"{getattr(mk_res, 'tau', 0):.4f}")
            st.caption("Trend Strength")

        st.divider()
        is_significant = "SIGNIFICANT" if mk_res.p < 0.05 else "NOT SIGNIFICANT"
        color = "green" if mk_res.p < 0.05 else "red"
        st.markdown(f"**Conclusion:** The data shows a **{mk_res.trend.upper()}** trend that is statistically **:{color}[{is_significant}]**.")

        # Overlay Sen's Slope on Data
        df['Trend_Line'] = mk_res.intercept + mk_res.slope * np.arange(len(df))
        base = alt.Chart(df).encode(x=alt.X('Combined_Date:T', title='Timeline'))
        raw = base.mark_line(color='#1e293b', opacity=0.4).encode(y=alt.Y('SPEI_Proxy:Q', title='SPEI'))
        trend = base.mark_line(color='#00d4ff', strokeWidth=3).encode(y='Trend_Line:Q')
        st.altair_chart((raw + trend).properties(height=350).interactive(), use_container_width=True)

    with tab4:
        st.subheader("🤖 AI Strategic Briefing")
        if st.button("Analyze with Gemini"):
            if st.session_state.saved_key:
                try:
                    genai.configure(api_key=st.session_state.saved_key)
                    # Automatically select the first available text model to prevent 404 errors
                    models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    if models:
                        model = genai.GenerativeModel(models[0])
                        prompt = f"Climatologist view: Trend={mk_res.trend}, Slope={mk_res.slope:.6f}, Max Drought={max_duration} months, P-value={mk_res.p:.5f}. Briefly assess risk."
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    else:
                        st.error("No valid text generation models found for this API key.")
                except Exception as e:
                    st.error(f"AI Error: {e}")

    with tab5:
        st.subheader("🔮 12-Month Predictive Modeling (Prophet)")
        try:
            p_df = df[['Combined_Date', 'SPEI_Proxy']].rename(columns={'Combined_Date': 'ds', 'SPEI_Proxy': 'y'})
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(p_df)
            future = m.make_future_dataframe(periods=12, freq='MS')
            forecast = m.predict(future)
            
            forecast_results = forecast.tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            base_f = alt.Chart(forecast_results).encode(x=alt.X('ds:T', title='Future Timeline'))
            band = base_f.mark_area(opacity=0.3, color='#00d4ff').encode(y=alt.Y('yhat_lower:Q', title='Predicted SPEI'), y2='yhat_upper:Q')
            line = base_f.mark_line(color='#00d4ff', strokeDash=[5,5], point=True).encode(y='yhat:Q', tooltip=['ds', 'yhat'])
            
            st.altair_chart((band + line + danger_lines).properties(height=400).interactive(), use_container_width=True)
            st.success("Prophet has successfully detected the seasonal patterns in your data and projected them forward.")
            
        except Exception as e:
            st.error(f"Prophet Error: {e}")
else:
    st.info("👋 Upload climate data to begin.")