import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Black+Ops+One&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #04080f !important;
    color: #f0f0f0 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(255,165,0,0.08) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 80%, rgba(0,180,255,0.07) 0%, transparent 55%),
        #04080f !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 960px !important; }

/* ── Hero Title ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(255,130,0,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff8c00;
    border: 1px solid rgba(255,140,0,0.4);
    border-radius: 2px;
    padding: 0.25rem 0.75rem;
    margin-bottom: 1rem;
    animation: pulse-badge 2.5s ease-in-out infinite;
}
@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 6px rgba(255,140,0,0.3); }
    50% { box-shadow: 0 0 18px rgba(255,140,0,0.6); }
}
.hero-title {
    font-family: 'Black Ops One', cursive;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    line-height: 1;
    letter-spacing: 0.03em;
    background: linear-gradient(135deg, #ff8c00 0%, #ffd700 50%, #ff4500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(255,140,0,0.4));
    animation: flicker 6s ease-in-out infinite;
}
@keyframes flicker {
    0%, 94%, 100% { filter: drop-shadow(0 0 30px rgba(255,140,0,0.4)); }
    95% { filter: drop-shadow(0 0 60px rgba(255,200,0,0.8)); }
    97% { filter: drop-shadow(0 0 20px rgba(255,140,0,0.2)); }
}
.hero-sub {
    font-family: 'Barlow', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    color: rgba(240,240,240,0.5);
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,0,0.5), rgba(255,215,0,0.5), transparent);
    margin: 1.5rem 0;
    animation: shimmer-line 3s linear infinite;
    background-size: 200% 100%;
}
@keyframes shimmer-line {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,0,0.15);
    border-radius: 8px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,140,0,0.04), transparent);
    animation: card-sweep 4s ease-in-out infinite;
}
@keyframes card-sweep {
    0% { left: -100%; }
    50%, 100% { left: 100%; }
}
.card:hover {
    border-color: rgba(255,140,0,0.4);
    box-shadow: 0 0 25px rgba(255,140,0,0.1);
}
.card-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff8c00;
    margin-bottom: 0.75rem;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,140,0,0.25) !important;
    border-radius: 6px !important;
    color: #f0f0f0 !important;
    font-family: 'Barlow', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: rgba(255,140,0,0.7) !important;
    box-shadow: 0 0 12px rgba(255,140,0,0.2) !important;
}
.stSelectbox svg { fill: #ff8c00 !important; }

label[data-testid="stWidgetLabel"] p {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(240,240,240,0.6) !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #ff6a00 0%, #ffb300 100%) !important;
    color: #04080f !important;
    font-family: 'Black Ops One', cursive !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.9rem 2rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,0,0.35) !important;
}
.stButton > button::after {
    content: '';
    position: absolute;
    top: -50%; left: -60%;
    width: 40%; height: 200%;
    background: rgba(255,255,255,0.25);
    transform: skewX(-20deg);
    animation: btn-shine 3s ease-in-out infinite;
}
@keyframes btn-shine {
    0%, 70%, 100% { left: -60%; }
    40% { left: 120%; }
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(255,140,0,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Panel ── */
.result-container {
    margin-top: 2rem;
    animation: result-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes result-in {
    from { opacity: 0; transform: translateY(30px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    text-align: center;
    margin-bottom: 1.5rem;
}
.teams-row {
    display: flex;
    gap: 1rem;
    align-items: stretch;
    justify-content: center;
    flex-wrap: wrap;
}
.team-card {
    flex: 1;
    min-width: 200px;
    max-width: 340px;
    border-radius: 10px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.team-card.batting {
    background: linear-gradient(145deg, rgba(255,140,0,0.12), rgba(255,215,0,0.08));
    border: 1px solid rgba(255,140,0,0.4);
    box-shadow: 0 0 40px rgba(255,140,0,0.15);
}
.team-card.bowling {
    background: linear-gradient(145deg, rgba(0,150,255,0.1), rgba(0,80,200,0.07));
    border: 1px solid rgba(0,150,255,0.35);
    box-shadow: 0 0 40px rgba(0,150,255,0.12);
}
.team-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.batting .team-label { color: rgba(255,180,0,0.7); }
.bowling .team-label { color: rgba(80,180,255,0.7); }

.team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    letter-spacing: 0.04em;
    margin-bottom: 1.25rem;
    line-height: 1.2;
}
.batting .team-name { color: #ffd700; }
.bowling .team-name { color: #7ec8f7; }

.pct-number {
    font-family: 'Black Ops One', cursive;
    font-size: 4rem;
    line-height: 1;
    letter-spacing: -0.02em;
    display: block;
    animation: count-in 0.8s ease-out forwards;
}
.batting .pct-number {
    color: #ff8c00;
    filter: drop-shadow(0 0 20px rgba(255,140,0,0.6));
}
.bowling .pct-number {
    color: #00aaff;
    filter: drop-shadow(0 0 20px rgba(0,170,255,0.5));
}
@keyframes count-in {
    from { transform: scale(0.5); opacity: 0; }
    to   { transform: scale(1); opacity: 1; }
}

.pct-symbol {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 600;
    font-size: 1.5rem;
    vertical-align: super;
    margin-left: 2px;
}

/* ── Progress Bar ── */
.matchup-bar-wrap { margin: 1.5rem 0 0.5rem; }
.matchup-bar {
    height: 10px;
    border-radius: 99px;
    overflow: hidden;
    background: rgba(255,255,255,0.07);
    position: relative;
}
.matchup-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #ff6a00, #ffd700);
    transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}
.matchup-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 20px; height: 100%;
    background: rgba(255,255,255,0.4);
    filter: blur(4px);
    animation: bar-glow 1.5s ease-in-out infinite;
}
@keyframes bar-glow {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Barlow', sans-serif;
    font-size: 0.7rem;
    color: rgba(240,240,240,0.45);
    margin-top: 0.4rem;
    letter-spacing: 0.04em;
}

/* ── Stats Row ── */
.stats-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.stat-pill {
    flex: 1;
    min-width: 100px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.stat-pill-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(240,240,240,0.35);
    margin-bottom: 0.25rem;
}
.stat-pill-value {
    font-family: 'Black Ops One', cursive;
    font-size: 1.3rem;
    color: #ffd700;
}

/* ── Error ── */
.stAlert {
    background: rgba(255, 50, 50, 0.08) !important;
    border: 1px solid rgba(255,80,80,0.3) !important;
    border-radius: 6px !important;
    font-family: 'Barlow', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #04080f; }
::-webkit-scrollbar-thumb { background: rgba(255,140,0,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,140,0,0.6); }

/* ── Stagger animation for columns ── */
[data-testid="column"]:nth-child(1) { animation: slide-up 0.5s 0.1s ease-out both; }
[data-testid="column"]:nth-child(2) { animation: slide-up 0.5s 0.2s ease-out both; }
[data-testid="column"]:nth-child(3) { animation: slide-up 0.5s 0.3s ease-out both; }
@keyframes slide-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ─── DATA ─────────────────────────────────────────────────────────────────────
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Rajasthan Royals',
    'Delhi Capitals', 'Sunrisers Hyderabad'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# ─── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🏏 Live Match Analysis</div>
    <div class="hero-title">IPL WIN PREDICTOR</div>
    <div class="hero-sub">Real-time ML-powered probability engine</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open('pipe.pkl', 'rb'))

pipe = load_model()

# ─── TEAM SELECTION ───────────────────────────────────────────────────────────
st.markdown('<div class="card-title">⚔️ Select Teams</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Batting Team', sorted(teams), key='batting')
with col2:
    bowling_team = st.selectbox('🎯 Bowling Team', sorted(teams), key='bowling')

# ─── VENUE ────────────────────────────────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="card-title">🏟️ Venue</div>', unsafe_allow_html=True)
selected_city = st.selectbox('Host City', sorted(cities))

# ─── MATCH STATE ──────────────────────────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="card-title">📊 Match State</div>', unsafe_allow_html=True)

col_t, _ = st.columns([1, 2])
with col_t:
    target = st.number_input('🎯 Target Score', min_value=0, step=1)

st.markdown('<br>', unsafe_allow_html=True)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs Completed', min_value=0.0, max_value=20.0, step=0.1, format="%.1f")
with col5:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1)

# ─── BUTTON ───────────────────────────────────────────────────────────────────
st.markdown('<br>', unsafe_allow_html=True)
predict_btn = st.button('⚡ PREDICT WIN PROBABILITY')

# ─── PREDICTION LOGIC ─────────────────────────────────────────────────────────
if predict_btn:
    if batting_team == bowling_team:
        st.error("⚠️ Please select two different teams.")
    else:
        runs_left   = target - score
        balls_left  = 120 - (overs * 6)
        wickets_left = 10 - wickets

        if overs == 0 or wickets > 10:
            st.error("⚠️ Please check your inputs — overs must be > 0 and wickets ≤ 10.")

        elif score >= target:
            bat_pct, bowl_pct = 100, 0

            st.markdown(f"""
            <div class="result-container">
                <div class="result-title">Match Result</div>
                <div class="teams-row">
                    <div class="team-card batting">
                        <div class="team-label">🏏 Batting</div>
                        <div class="team-name">{batting_team}</div>
                        <span class="pct-number">{bat_pct}<span class="pct-symbol">%</span></span>
                    </div>
                    <div class="team-card bowling">
                        <div class="team-label">🎯 Bowling</div>
                        <div class="team-name">{bowling_team}</div>
                        <span class="pct-number">{bowl_pct}<span class="pct-symbol">%</span></span>
                    </div>
                </div>
                <div class="matchup-bar-wrap">
                    <div class="matchup-bar">
                        <div class="matchup-bar-fill" style="width:{bat_pct}%"></div>
                    </div>
                    <div class="bar-labels"><span>{batting_team}</span><span>{bowling_team}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif wickets >= 10 or (balls_left <= 0 and score < target):
            bat_pct, bowl_pct = 0, 100

            st.markdown(f"""
            <div class="result-container">
                <div class="result-title">Match Result</div>
                <div class="teams-row">
                    <div class="team-card batting">
                        <div class="team-label">🏏 Batting</div>
                        <div class="team-name">{batting_team}</div>
                        <span class="pct-number">{bat_pct}<span class="pct-symbol">%</span></span>
                    </div>
                    <div class="team-card bowling">
                        <div class="team-label">🎯 Bowling</div>
                        <div class="team-name">{bowling_team}</div>
                        <span class="pct-number">{bowl_pct}<span class="pct-symbol">%</span></span>
                    </div>
                </div>
                <div class="matchup-bar-wrap">
                    <div class="matchup-bar">
                        <div class="matchup-bar-fill" style="width:{bat_pct}%"></div>
                    </div>
                    <div class="bar-labels"><span>{batting_team}</span><span>{bowling_team}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            crr = round(score / overs, 2)
            rrr = round((runs_left * 6) / balls_left, 2) if balls_left > 0 else 99.99

            input_df = pd.DataFrame({
                'batting_team':  [batting_team],
                'bowling_team':  [bowling_team],
                'city':          [selected_city],
                'runs_left':     [int(runs_left)],
                'balls_left':    [int(balls_left)],
                'wickets':       [int(wickets_left)],
                'total_runs_x':  [int(target)],
                'crr':           [crr],
                'rrr':           [rrr],
            })

            result   = pipe.predict_proba(input_df)
            bat_pct  = round(result[0][1] * 100)
            bowl_pct = round(result[0][0] * 100)

            st.markdown(f"""
            <div class="result-container">
                <div class="result-title">Win Probability</div>
                <div class="teams-row">
                    <div class="team-card batting">
                        <div class="team-label">🏏 Batting</div>
                        <div class="team-name">{batting_team}</div>
                        <span class="pct-number">{bat_pct}<span class="pct-symbol">%</span></span>
                    </div>
                    <div class="team-card bowling">
                        <div class="team-label">🎯 Bowling</div>
                        <div class="team-name">{bowling_team}</div>
                        <span class="pct-number">{bowl_pct}<span class="pct-symbol">%</span></span>
                    </div>
                </div>
                <div class="matchup-bar-wrap">
                    <div class="matchup-bar">
                        <div class="matchup-bar-fill" style="width:{bat_pct}%"></div>
                    </div>
                    <div class="bar-labels"><span>{batting_team}</span><span>{bowling_team}</span></div>
                </div>
                <div class="stats-row">
                    <div class="stat-pill">
                        <div class="stat-pill-label">Runs Left</div>
                        <div class="stat-pill-value">{int(runs_left)}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">Balls Left</div>
                        <div class="stat-pill-value">{int(balls_left)}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">Wickets Left</div>
                        <div class="stat-pill-value">{int(wickets_left)}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">CRR</div>
                        <div class="stat-pill-value">{crr}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-pill-label">RRR</div>
                        <div class="stat-pill-value">{rrr}</div>
                    </div>
                </div>  
            </div>
            """, unsafe_allow_html=True)