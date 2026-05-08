import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import pymannkendall as mk
from prophet import Prophet
from scipy.signal import find_peaks
import os
import streamlit.components.v1 as components # Added for interactive HTML

# --- INITIALIZATION & STYLING ---
st.set_page_config(layout="wide", page_title="Drought Intelligence Pro", page_icon="🛰️")

os.environ['STREAMLIT_DATAFRAME_ENCODING'] = 'legacy'

# Inject Custom CSS for a "Premium" look
# Note: If you want seamless Light Mode, you might want to remove the hardcoded background-color here eventually!
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

# Refined helper function for parameter-specific styling
def make_detailed_chart(data, column, label, unit, color):
    return alt.Chart(data).mark_line(color=color, size=2).encode(
        x=alt.X('Combined_Date:T', title='Timeline (Year/Month)'),
        y=alt.Y(f'{column}:Q', title=f"{label} ({unit})", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip('Combined_Date:T', format='%B %Y', title='Date'),
            alt.Tooltip(f'{column}:Q', format='.2f', title=f"{label} ({unit})")
        ]
    ).properties(
        title=f"Historical {label} Trends (1970-2025)",
        height=chart_height
    ).interactive()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛰️ Drought Control")
    
    # Purely user-provided API key. "password" type hides the text.
    user_api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key at ai.google.dev")
    
    if st.button("🔍 Check API Connection"):
        if user_api_key:
            try:
                genai.configure(api_key=user_api_key)
                # Quick test to see if the key is valid
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                st.success("✅ Connection Successful!")
            except Exception as e:
                st.error(f"❌ Invalid Key: {e}")
        else:
            st.warning("Please enter an API Key first.")
            
    st.subheader("📊 Chart Controls")
    # ... keep your zoom_range and file uploader code exactly the same below this ...            
    st.subheader("📊 Chart Controls")
    
    # This slider controls the Y-Axis of Tab 1
    zoom_range = st.slider(
        "Focus on Deficit Range (Y-Axis)", 
        min_value=-150, max_value=500, value=(-5, 10)
    )
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

    # --- DATA PROCESSING ---
    df = df_raw.copy()
    
    # ERROR HANDLING ADDED HERE: Catches invalid data like Pivot Tables
    try:
        df['Combined_Date'] = pd.to_datetime(df[year_col].astype(str) + '-' + df[month_col].astype(str) + '-01')
        for col in [p_col, t_col, w_col, h_col]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Combined_Date', p_col, t_col]).sort_values('Combined_Date')
    except Exception:
        st.warning("⚠️ Invalid data format. This usually occurs if you've selected a Pivot Table or Summary Sheet. Please select a sheet containing raw, chronologically ordered climate data.")
        st.stop() # Prevents the rest of the code from running and causing a red error block

    # --- SPEI CALCULATIONS ---
    df['PET'] = df.apply(lambda r: calculate_simplified_pet(r[t_col], r[w_col], r[h_col]), axis=1)
    df['D'] = df[p_col] - df['PET']
    df['Status'] = np.where(df['D'] >= 0, 'Surplus', 'Deficit')
    df['SPEI_Proxy'] = (df['D'] - df['D'].mean()) / df['D'].std()
    df['SPEI_Rolling'] = df['SPEI_Proxy'].rolling(window=12, center=True).mean()
    
    extreme_count = len(df[df['SPEI_Proxy'] <= -2.0])
    max_duration = get_longest_duration(df['SPEI_Proxy'])
    mk_res = mk.original_test(df['SPEI_Proxy'])

    # --- PROPHET FORECAST (Pre-calculated for use in multiple tabs) ---
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
    
    # RENAME ds and yhat to simpler terms for UI
    forecast_reg = forecast_reg.rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted_SPEI',
        'yhat_lower': 'Lower_Confidence',
        'yhat_upper': 'Upper_Confidence'
    })
    
    future_only = forecast_reg[forecast_reg['Date'] > df_p_reg['ds'].max()]

    # --- DASHBOARD HEADER ---
    st.title(f"🛰️ SPEI Intelligence: {selected_sheet}")
    st.caption("Advanced Hydrological Analytics & Predictive Modeling")
    
    # --- METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trend Direction", mk_res.trend.upper(), delta=f"{mk_res.slope:.6f} units/mo")
    with m2: st.metric("Max Streak", f"{max_duration} Months") 
    with m3: st.metric("Extreme Events", extreme_count)
    with m4: st.metric("Current Status", "MODERATE" if df['SPEI_Proxy'].iloc[-1] < -1 else "STABLE")

    # --- TABS REORDERED (AI Brief moved to end) ---
    tab_t, tab_p, tab_w, tab_h, tab1, tab2, tab3, tab4, tab6, tab7, tab5 = st.tabs([
        "🌡️ Temperature", "🌧️ Precipitation", "💨 Wind Speed", "💧 Humidity",
        "📈 Water Dynamics", "📉 SPEI Tracker", "🌡️ Heatmap", "📊 Statistics", 
        "📊 Full Timeline", "🔮 12-Month Forecast", "🤖 AI Brief"
    ])

    with tab_t:
        st.subheader("🌡️ Temperature Analysis")
        chart_t = make_detailed_chart(df, t_col, "Temperature", "°C", "#FF4B4B") 
        st.altair_chart(chart_t, use_container_width=True)

    with tab_p:
        st.subheader("🌧️ Precipitation Analysis")
        chart_p = make_detailed_chart(df, p_col, "Precipitation", "mm", "#00D4FF")
        st.altair_chart(chart_p, use_container_width=True)

    with tab_w:
        st.subheader("💨 Wind Speed Analysis")
        chart_w = make_detailed_chart(df, w_col, "Wind Speed", "m/s", "#94A3B8") 
        st.altair_chart(chart_w, use_container_width=True)

    with tab_h:
        st.subheader("💧 Humidity Analysis")
        chart_h = make_detailed_chart(df, h_col, "Relative Humidity", "%", "#00C853")
        st.altair_chart(chart_h, use_container_width=True)

    danger_lines = alt.Chart(pd.DataFrame({'y': [-1.5, -2.0], 'label': ['Severe', 'Extreme']})).mark_rule(color='red', strokeDash=[5,5]).encode(y='y:Q', tooltip='label:N')

    with tab1:
        st.subheader("Monthly Hydrological Balance ($D = P - PET$)")
        
        brush = alt.selection_interval(encodings=['x']) 
        
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
        
        nav_chart = alt.Chart(df).mark_area(opacity=0.4, color='#64748b').encode(
            x=alt.X('Combined_Date:T', title='← Drag to Zoom →'), 
            y=alt.Y('D:Q', axis=None)
        ).properties(width='container', height=80).add_params(brush)
        
        final_plot = alt.vconcat((main_chart + zero_line), nav_chart).configure_view(
            stroke=None
        ).properties(autosize=alt.AutoSizeParams(type='fit-x', contains='padding'))
        
        st.altair_chart(final_plot, use_container_width=True)

    with tab2:
        st.subheader("📉 Standardized Precipitation Evapotranspiration Index (Proxy)")
        spei_chart = alt.Chart(df).mark_line(color='#00d4ff', size=2).encode(
            x=alt.X('Combined_Date:T', title='Timeline'),
            y=alt.Y('SPEI_Proxy:Q', title='Calculated SPEI Proxy'),
            tooltip=[alt.Tooltip('Combined_Date:T', format='%B %Y'), alt.Tooltip('SPEI_Proxy:Q', format='.2f')]
        ).properties(height=500).interactive()
        
        zero_rule = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='white', strokeDash=[4, 4]).encode(y='y:Q')
        st.altair_chart((spei_chart + zero_rule), use_container_width=True)

    with tab3:
        st.subheader("🌡️ Historical Heatmap")
        df['Year'] = df['Combined_Date'].dt.year
        df['Month_Name'] = df['Combined_Date'].dt.month_name().str[:3]
        heatmap = alt.Chart(df).mark_rect().encode(
            x=alt.X('Month_Name:N', title='Month', sort=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
            y=alt.Y('Year:O', title='Year', axis=alt.Axis(values=list(range(df['Year'].min(), df['Year'].max(), 5)))),
            color=alt.Color('SPEI_Proxy:Q', scale=alt.Scale(scheme='redblue', domain=[-3, 3])),
            tooltip=['Year', 'Month_Name', 'SPEI_Proxy']
        ).properties(height=500).interactive()
        st.altair_chart(heatmap, use_container_width=True)

    with tab4:
        st.subheader("🧪 Mann-Kendall Trend Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("$p$-value", f"{mk_res.p:.5f}")
        c2.metric("$z$-score", f"{mk_res.z:.4f}")
        c3.metric("Sen's Slope", f"{mk_res.slope:.6f}")
        
        is_significant = mk_res.p < 0.05
        is_decreasing = mk_res.slope < 0
        report_color = "#ff4b4b" if (is_significant and is_decreasing) else "#00c853"
        status_text = "SIGNIFICANT DRYING TREND" if (is_significant and is_decreasing) else "STABLE / INCREASING MOISTURE"

        # CHANGED: Replaced color:white with var(--text-color) so it adapts to the theme automatically
        st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background-color: {report_color}22; border: 2px solid {report_color}; margin-bottom: 20px;">
                <h3 style="color:{report_color}; margin:0;">{status_text}</h3>
                <p style="color: var(--text-color); margin: 5px 0 0 0;">
                    The dataset shows a <b>{mk_res.trend}</b> trend at <b>{mk_res.slope:.6f}</b> units/month.
                </p>
            </div>
        """, unsafe_allow_html=True)

        peak_indices, _ = find_peaks(df['SPEI_Proxy'].values, distance=24, height=0.5)
        df_peaks = df.iloc[peak_indices].copy()
        top_3_peaks = df_peaks.nlargest(3, 'SPEI_Proxy')
        
        base = alt.Chart(df).encode(x=alt.X('Combined_Date:T', title='Timeline'))
        raw = base.mark_line(color='#64748b', opacity=0.8).encode(y=alt.Y('SPEI_Proxy:Q', title='SPEI'))
        points = alt.Chart(df_peaks).mark_circle(color='#00d4ff', size=60).encode(
            x='Combined_Date:T', y='SPEI_Proxy:Q', tooltip=['Combined_Date', 'SPEI_Proxy']
        )
        
        peak_trend_curve = alt.Chart(df_peaks).mark_line(color='#ffd700', size=4).transform_loess('Combined_Date', 'SPEI_Proxy').encode(x='Combined_Date:T', y='SPEI_Proxy:Q')
        peak_lines = alt.Chart(top_3_peaks).mark_rule(color='#ff4b4b', strokeDash=[5,5], size=1.5).encode(
            y='SPEI_Proxy:Q', tooltip=[alt.Tooltip('SPEI_Proxy', title='Peak Height')]
        )

        st.altair_chart((raw + points + peak_trend_curve + peak_lines).properties(height=500).interactive(), use_container_width=True)

    with tab6:
        st.subheader("📊 Full Historical & Predicted Timeline")
        df_hist_plot = df_p_reg.copy().rename(columns={'ds': 'Date', 'y': 'SPEI_Value'})
        df_hist_plot['Type'] = 'Historical SPEI'
        
        df_pred_plot = future_only[['Date', 'Predicted_SPEI']].rename(columns={'Predicted_SPEI': 'SPEI_Value'})
        df_pred_plot['Type'] = 'Predicted SPEI'
        
        combined_timeline = pd.concat([df_hist_plot[['Date', 'SPEI_Value', 'Type']], df_pred_plot])
        
        timeline_chart = alt.Chart(combined_timeline).mark_line().encode(
            x=alt.X('Date:T', title='Timeline'),
            y=alt.Y('SPEI_Value:Q', title='SPEI Value'),
            color=alt.Color('Type:N', scale=alt.Scale(
                domain=['Historical SPEI', 'Predicted SPEI'], 
                range=['#64748b', '#00d4ff']
            )),
            tooltip=['Date', 'SPEI_Value', 'Type']
        ).properties(height=450).interactive()
        
        st.altair_chart(timeline_chart, use_container_width=True)

    with tab7:
        st.subheader("🔮 Prophet: 12-Month Future Projection")
        line = alt.Chart(future_only).mark_line(color='#00d4ff', size=3).encode(
            x=alt.X('Date:T', title='Timeline'),
            y=alt.Y('Predicted_SPEI:Q', title='Predicted SPEI'),
            tooltip=['Date', 'Predicted_SPEI']
        )
        band = alt.Chart(future_only).mark_area(opacity=0.2, color='#00d4ff').encode(
            x='Date:T', y='Lower_Confidence:Q', y2='Upper_Confidence:Q'
        )
        danger_1 = alt.Chart(pd.DataFrame({'y': [-1.0]})).mark_rule(color='#ff4b4b', strokeDash=[4, 4]).encode(y='y:Q')
        danger_2 = alt.Chart(pd.DataFrame({'y': [-1.5]})).mark_rule(color='red', strokeWidth=2).encode(y='y:Q')
        
        st.altair_chart((band + line + danger_1 + danger_2).interactive().properties(height=450), use_container_width=True)

    with tab5:
        st.subheader("🤖 AI Strategic Briefing")
        if st.button("Generate AI Assessment"):
            if user_api_key:
                try:
                    genai.configure(api_key=user_api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')

                    surplus_count = len(df[df['Status'] == 'Surplus'])
                    deficit_count = len(df[df['Status'] == 'Deficit'])
                    prompt = (
                        f"Analyze risk: Water Balance: {surplus_count} surplus vs {deficit_count} deficit. "
                        f"MK: {mk_res.trend}, Slope: {mk_res.slope:.6f}. Max Dry Streak: {max_duration} months. "
                        f"Forecast: Predicted avg SPEI {future_only['Predicted_SPEI'].mean():.2f}."
                    )
                    with st.spinner("Analyzing..."):
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e: 
                    st.error(f"AI Error: {e}")
            else:
                st.warning("Please provide a Gemini API Key in the sidebar.")

else:
    # CHANGED: Replaced the simple st.info with an interactive mouse-reactive canvas
    st.markdown("<h2 style='text-align: center;'>🛰️ Welcome to Drought Intelligence Pro</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Awaiting climate data upload via the sidebar...</p>", unsafe_allow_html=True)
    
    # Injecting a cool interactive network/particle animation
    interactive_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { margin: 0; overflow: hidden; background-color: transparent; }
        canvas { display: block; width: 100vw; height: 65vh; border-radius: 10px; }
    </style>
    </head>
    <body>
    <canvas id="particleCanvas"></canvas>
    <script>
        const canvas = document.getElementById('particleCanvas');
        const ctx = canvas.getContext('2d');
        
        let width = canvas.width = window.innerWidth;
        let height = canvas.height = window.innerHeight;
        
        let particles = [];
        let mouse = { x: width / 2, y: height / 2 };

        // Track mouse position
        window.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
        });

        // Handle resizing
        window.addEventListener('resize', () => {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        });

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 2;
                this.vy = (Math.random() - 0.5) * 2;
                this.radius = Math.random() * 2 + 1;
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                // Bounce off edges
                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
                
                // Mouse interaction line drawing
                let dx = mouse.x - this.x;
                let dy = mouse.y - this.y;
                let distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance < 120) {
                    ctx.beginPath();
                    ctx.strokeStyle = `rgba(0, 212, 255, ${1 - distance / 120})`;
                    ctx.lineWidth = 1;
                    ctx.moveTo(this.x, this.y);
                    ctx.lineTo(mouse.x, mouse.y);
                    ctx.stroke();
                }
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
                ctx.fill();
            }
        }

        // Initialize particles
        for (let i = 0; i < 100; i++) {
            particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            particles.forEach(p => {
                p.update();
                p.draw();
            });
            requestAnimationFrame(animate);
        }

        animate();
    </script>
    </body>
    </html>
    """
    
    components.html(interactive_html, height=500)