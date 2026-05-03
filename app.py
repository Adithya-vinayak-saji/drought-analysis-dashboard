import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from prophet import Prophet 
from scipy.signal import find_peaks
import os

# --- HARDCODED API KEY (Optional) ---
HARDCODED_GEMINI_KEY = "AIzaSyB7sfQdXshhvGqb5IWbtDXx53X6h9CsIj4" 

# --- INITIALIZATION & STYLING ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Pro", page_icon="🛰️")

# Inject Custom CSS for a "Premium" look

os.environ['STREAMLIT_DATAFRAME_ENCODING'] = 'legacy'

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #00d4ff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e293b; border-radius: 4px; color: white; padding: 10px;
    }
    /* Ensure Vega-Lite containers take full width */
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
    st.subheader("📊 Chart Controls")
    
    # This slider controls the Y-Axis of Tab 1
    zoom_range = st.slider(
        "Focus on Deficit Range (Y-Axis)", 
        min_value=-150, max_value=500, value=(-5, 10)
    )
    chart_height = st.number_input("Main Chart Height", min_value=300, max_value=800, value=450)

    st.divider()
    
    if HARDCODED_GEMINI_KEY.strip():
        st.session_state.saved_key = HARDCODED_GEMINI_KEY
        st.success("🔒 Using built-in API Key")
    else:
        input_key = st.text_input("Gemini API Key", value=st.session_state.saved_key, type="password")
        if st.checkbox("Remember API Key", value=bool(st.session_state.saved_key)):
            st.session_state.saved_key = input_key
    
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
    # --- CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['Status'] = np.where(df['D'] >= 0, 'Surplus', 'Deficit')
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    
    max_duration = get_longest_duration(df['SPEI_Proxy'])
    mk_res = mk.original_test(df['SPEI_Proxy'])
    
    # --- PROPHET FORECAST (Restored) ---
    df_p_reg = df[['Combined_Date', 'SPEI_Proxy', t_col]].rename(
        columns={'Combined_Date': 'ds', 'SPEI_Proxy': 'y', t_col: 'temp'}
    )
    m_reg = Prophet(yearly_seasonality=True, interval_width=0.95)
    m_reg.add_regressor('temp')
    m_reg.fit(df_p_reg)
    
    future_reg = m_reg.make_future_dataframe(periods=12, freq='MS')
    future_reg = future_reg.merge(df_p_reg[['ds', 'temp']], on='ds', how='left')
    monthly_avg = df_p_reg.groupby(df_p_reg['ds'].dt.month)['temp'].mean()
    future_reg['month'] = future_reg['ds'].dt.month
    future_reg['temp'] = future_reg.apply(lambda x: x['temp'] if pd.notnull(x['temp']) else monthly_avg[x['month']], axis=1)
    
    forecast_reg = m_reg.predict(future_reg)
    future_only = forecast_reg[forecast_reg['ds'] > df_p_reg['ds'].max()]

    # --- DASHBOARD RENDERING ---
    st.title(f"🛰️ SPEI Intelligence")
    st.caption("Discrete Monthly Analysis of Climate Data")
    
    # Added New Tab 2 for SPEI
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Water Dynamics", "📉 SPEI Tracker", "🌡️ Heatmap", "📊 Statistics", 
        "🤖 AI Brief", "📊 Full Timeline", "🔮 12-Month Forecast"
    ])

    with tab1:
        st.subheader("Monthly Hydrological Balance ($D = P - PET$)")
        
        # Define the shared brush
        brush = alt.selection_interval(encodings=['x']) 
        
        # Main Chart
        main_chart = alt.Chart(df).mark_bar(size=4, opacity=0.9).encode(
            x=alt.X('Combined_Date:T', title='Timeline', scale=alt.Scale(domain=brush)),
            y=alt.Y('D:Q', title='Water Balance (mm)', scale=alt.Scale(domain=zoom_range)),
            color=alt.Color('Status:N', 
                scale=alt.Scale(domain=['Surplus', 'Deficit'], range=['#00d4ff', '#ff4b4b']),
                legend=alt.Legend(title="Status", orient='right') 
            ),
            tooltip=[
                alt.Tooltip('Combined_Date:T', format='%B %Y', title='Month'), 
                alt.Tooltip('D:Q', format='.2f', title='Balance')
            ]
        ).properties(width='container', height=chart_height)
        
        zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='white', opacity=0.3).encode(y='y:Q')
        
        # Navigator
        nav_chart = alt.Chart(df).mark_area(opacity=0.4, color='#64748b').encode(
            x=alt.X('Combined_Date:T', title='← Drag to Zoom →'), 
            y=alt.Y('D:Q', axis=None)
        ).properties(width='container', height=80).add_params(brush)
        
        # Combine using vconcat for stricter layout control
        final_plot = alt.vconcat(
            (main_chart + zero_line), 
            nav_chart
        ).configure_view(
            stroke=None
        ).properties(
            autosize=alt.AutoSizeParams(type='fit-x', contains='padding')
        )
        
        st.altair_chart(final_plot, use_container_width=True)

    with tab2:
        st.subheader("📉 Standardized Precipitation Evapotranspiration Index (Proxy)")
        spei_chart = alt.Chart(df).mark_line(color='#00d4ff', size=2).encode(
            x=alt.X('Combined_Date:T', title='Timeline'),
            y=alt.Y('SPEI_Proxy:Q', title='Calculated SPEI Proxy'),
            tooltip=[alt.Tooltip('Combined_Date:T', format='%B %Y'), alt.Tooltip('SPEI_Proxy:Q', format='.2f')]
        ).properties(height=500).interactive()
        
        # Add a zero line for reference
        zero_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='white', strokeDash=[4, 4]).encode(y='y:Q')
        st.altair_chart((spei_chart + zero_rule), use_container_width=True)

    with tab3:
        st.subheader("🌡️ Historical Heatmap")
        df['Year'] = df['Combined_Date'].dt.year
        df['Month_Name'] = df['Combined_Date'].dt.month_name().str[:3]
        heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('Month_Name:N', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
            y=alt.Y('Year:O', axis=alt.Axis(values=list(range(df['Year'].min(), df['Year'].max(), 5)))),
            color=alt.Color('SPEI_Proxy:Q', scale=alt.Scale(scheme='redblue', domain=[-3, 3])),
            tooltip=['Year', 'Month_Name', 'SPEI_Proxy']
        ).properties(height=500).interactive()
        st.altair_chart(heatmap, use_container_width=True)

    with tab4:
        st.subheader("🧪 Mann-Kendall Trend Analysis")
        # Restored the metric widgets
        c1, c2, c3 = st.columns(3)
        c1.metric("$p$-value", f"{mk_res.p:.5f}")
        c2.metric("$z$-score", f"{mk_res.z:.4f}")
        c3.metric("Sen's Slope", f"{mk_res.slope:.6f}")
        
        is_significant = mk_res.p < 0.05
        is_decreasing = mk_res.slope < 0
        report_color = "#ff4b4b" if (is_significant and is_decreasing) else "#00c853"
        status_text = "SIGNIFICANT DRYING TREND" if (is_significant and is_decreasing) else "STABLE / INCREASING MOISTURE"

        st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background-color: {report_color}22; border: 2px solid {report_color}; margin-bottom: 20px;">
                <h3 style="color:{report_color}; margin:0;">{status_text}</h3>
                <p style="color:white; margin: 5px 0 0 0;">
                    The dataset shows a <b>{mk_res.trend}</b> trend at <b>{mk_res.slope:.6f}</b> units/month.
                </p>
            </div>
        """, unsafe_allow_html=True)

        peak_indices, _ = find_peaks(df['SPEI_Proxy'].values, distance=24, height=0.5)
        df_peaks = df.iloc[peak_indices].copy()
        top_3_peaks = df_peaks.nlargest(3, 'SPEI_Proxy')
        
        base = alt.Chart(df).encode(x='Combined_Date:T')
        raw = base.mark_line(color='#64748b', opacity=0.8).encode(y='SPEI_Proxy:Q')
        points = alt.Chart(df_peaks).mark_circle(color='#00d4ff', size=60).encode(
            x='Combined_Date:T', y='SPEI_Proxy:Q', tooltip=['Combined_Date', 'SPEI_Proxy']
        )
        
        # Restored the curve, changed to Gold/Yellow
        peak_trend_curve = alt.Chart(df_peaks).mark_line(color='#ffd700', size=4).transform_loess('Combined_Date', 'SPEI_Proxy').encode(x='Combined_Date:T', y='SPEI_Proxy:Q')

        peak_lines = alt.Chart(top_3_peaks).mark_rule(
            color='#ff4b4b', strokeDash=[5,5], size=1.5
        ).encode(
            y='SPEI_Proxy:Q',
            tooltip=[alt.Tooltip('SPEI_Proxy', title='Peak Height')]
        )

        st.altair_chart((raw + points + peak_trend_curve + peak_lines).properties(height=500).interactive(), use_container_width=True)

    with tab5:
        if st.button("Generate AI Assessment"):
            if st.session_state.saved_key:
                try:
                    genai.configure(api_key=st.session_state.saved_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    surplus_count = len(df[df['Status'] == 'Surplus'])
                    deficit_count = len(df[df['Status'] == 'Deficit'])
                    prompt = (
                        f"Analyze risk: Water Balance: {surplus_count} surplus vs {deficit_count} deficit. "
                        f"MK: {mk_res.trend}, Slope: {mk_res.slope:.6f}. Max Dry Streak: {max_duration} months. "
                        f"Forecast: Predicted avg SPEI {future_only['yhat'].mean():.2f}."
                    )
                    with st.spinner("Analyzing..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e: st.error(f"AI Error: {e}")
            else:
                st.warning("Please provide a Gemini API Key in the sidebar.")

    with tab6:
        st.subheader("📊 Full Historical & Predicted Timeline")
        df_hist_plot = df_p_reg.copy()
        df_hist_plot['Type'] = 'Historical SPEI'
        
        df_pred_plot = future_only[['ds', 'yhat']].rename(columns={'yhat': 'y'})
        df_pred_plot['Type'] = 'Predicted SPEI'
        
        combined_timeline = pd.concat([df_hist_plot[['ds', 'y', 'Type']], df_pred_plot])
        
        timeline_chart = alt.Chart(combined_timeline).mark_line().encode(
            x=alt.X('ds:T', title='Timeline'),
            y=alt.Y('y:Q', title='SPEI Value'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Historical SPEI', 'Predicted SPEI'], 
                range=['#64748b', '#00d4ff']
            )),
            tooltip=['ds', 'y', 'Type']
        ).properties(height=450).interactive()
        
        st.altair_chart(timeline_chart, use_container_width=True)

    with tab7:
        st.subheader("🔮 Prophet: 12-Month Future Projection")
        line = alt.Chart(future_only).mark_line(color='#00d4ff', size=3).encode(
            x=alt.X('ds:T', title='Timeline'),
            y=alt.Y('yhat:Q', title='Predicted SPEI'),
            tooltip=['ds', 'yhat']
        )
        band = alt.Chart(future_only).mark_area(opacity=0.2, color='#00d4ff').encode(
            x='ds:T', y='yhat_lower:Q', y2='yhat_upper:Q'
        )
        danger_1 = alt.Chart(pd.DataFrame({'y': [-1.0]})).mark_rule(color='#ff4b4b', strokeDash=[4, 4]).encode(y='y:Q')
        danger_2 = alt.Chart(pd.DataFrame({'y': [-1.5]})).mark_rule(color='red', strokeWidth=2).encode(y='y:Q')
        
        st.altair_chart((band + line + danger_1 + danger_2).interactive().properties(height=450), use_container_width=True)

else:
    st.info("👋 Please upload your climate data to begin.")
