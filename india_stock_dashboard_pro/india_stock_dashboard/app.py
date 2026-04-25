import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NSE Terminal | India Markets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

# ── Theme Definitions ────────────────────────────────────────────────────────
THEMES = {
    "🌑 Midnight": {
        "BG_DARK":"#0A0E1A","BG_CARD":"#111827","BG_CARD2":"#1A2235",
        "BORDER":"#1E2D45","BLUE":"#2196F3","BLUE_L":"#64B5F6",
        "GREEN":"#00E676","GREEN_D":"#00C853","RED":"#FF1744","RED_D":"#F44336",
        "AMBER":"#FFD600","BROWN":"#8D6E63","MUTED":"#546E7A",
        "TEXT":"#E0E0E0","TEXT_D":"#90A4AE",
    },
    "🌊 Ocean": {
        "BG_DARK":"#000D1A","BG_CARD":"#001A33","BG_CARD2":"#002347",
        "BORDER":"#00396E","BLUE":"#00B0FF","BLUE_L":"#80D8FF",
        "GREEN":"#1DE9B6","GREEN_D":"#00BFA5","RED":"#FF5252","RED_D":"#FF1744",
        "AMBER":"#FFD740","BROWN":"#80CBC4","MUTED":"#37474F",
        "TEXT":"#E1F5FE","TEXT_D":"#81D4FA",
    },
    "💜 Cyberpunk": {
        "BG_DARK":"#0D001A","BG_CARD":"#160028","BG_CARD2":"#200038",
        "BORDER":"#3D006B","BLUE":"#EA00FF","BLUE_L":"#F48FFF",
        "GREEN":"#00FFAB","GREEN_D":"#00E5FF","RED":"#FF1744","RED_D":"#FF4081",
        "AMBER":"#FFD600","BROWN":"#CE93D8","MUTED":"#6A1B9A",
        "TEXT":"#F3E5F5","TEXT_D":"#CE93D8",
    },
    "🌿 Forest": {
        "BG_DARK":"#0A1A0F","BG_CARD":"#0F2518","BG_CARD2":"#1A3A25",
        "BORDER":"#1E4530","BLUE":"#00C853","BLUE_L":"#69F0AE",
        "GREEN":"#76FF03","GREEN_D":"#64DD17","RED":"#FF5252","RED_D":"#FF1744",
        "AMBER":"#FFEA00","BROWN":"#8D6E63","MUTED":"#4E6B58",
        "TEXT":"#E8F5E9","TEXT_D":"#A5D6A7",
    },
    "🌅 Sunset": {
        "BG_DARK":"#1A0A00","BG_CARD":"#2A1200","BG_CARD2":"#3A1A05",
        "BORDER":"#5D2E08","BLUE":"#FF6D00","BLUE_L":"#FFD180",
        "GREEN":"#CCFF90","GREEN_D":"#B2FF59","RED":"#FF1744","RED_D":"#F44336",
        "AMBER":"#FFD740","BROWN":"#BCAAA4","MUTED":"#795548",
        "TEXT":"#FFF8F0","TEXT_D":"#FFCC80",
    },
    "🏛️ Classic Light": {
        "BG_DARK":"#F4F6F9","BG_CARD":"#FFFFFF","BG_CARD2":"#EEF1F5",
        "BORDER":"#CBD5E1","BLUE":"#1565C0","BLUE_L":"#42A5F5",
        "GREEN":"#2E7D32","GREEN_D":"#388E3C","RED":"#C62828","RED_D":"#D32F2F",
        "AMBER":"#F57F17","BROWN":"#5D4037","MUTED":"#90A4AE",
        "TEXT":"#1A1A2E","TEXT_D":"#546E7A",
    },
}

# ── Session state defaults ───────────────────────────────────────────────────
if "theme"  not in st.session_state: st.session_state.theme  = "🌑 Midnight"
if "menu"   not in st.session_state: st.session_state.menu   = "Market Overview"
if "stock"  not in st.session_state: st.session_state.stock  = "RELIANCE"
if "period" not in st.session_state: st.session_state.period = "1y"

T = THEMES[st.session_state.theme]
BG_DARK=T["BG_DARK"]; BG_CARD=T["BG_CARD"]; BG_CARD2=T["BG_CARD2"]
BORDER=T["BORDER"];   BLUE=T["BLUE"];        BLUE_L=T["BLUE_L"]
GREEN=T["GREEN"];     GREEN_D=T["GREEN_D"];  RED=T["RED"];   RED_D=T["RED_D"]
AMBER=T["AMBER"];     BROWN=T["BROWN"];      MUTED=T["MUTED"]
TEXT=T["TEXT"];       TEXT_D=T["TEXT_D"]
IS_LIGHT = BG_DARK in ("#F4F6F9",)

plt.rcParams.update({
    "figure.facecolor":BG_CARD,"axes.facecolor":BG_CARD,"axes.edgecolor":BORDER,
    "axes.labelcolor":TEXT_D,"axes.titlecolor":TEXT,"xtick.color":TEXT_D,"ytick.color":TEXT_D,
    "grid.color":BORDER,"grid.linewidth":0.6,"text.color":TEXT,
    "legend.facecolor":BG_CARD2,"legend.edgecolor":BORDER,"legend.labelcolor":TEXT_D,
    "font.family":"monospace","axes.spines.top":False,"axes.spines.right":False,
})

BTN_TEXT = "#111111" if IS_LIGHT else "#ffffff"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{{font-family:'JetBrains Mono',monospace;background:{BG_DARK};color:{TEXT};}}
.stApp{{background:{BG_DARK};}}
/* Hide sidebar completely */
#MainMenu,footer,header{{visibility:hidden;}}
section[data-testid="stSidebar"]{{display:none!important;}}
[data-testid="collapsedControl"]{{display:none!important;}}
.block-container{{padding-top:0.2rem!important;padding-bottom:1rem;max-width:100%!important;}}
/* Metrics */
div[data-testid="metric-container"]{{background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;padding:16px!important;}}
div[data-testid="metric-container"] label{{color:{TEXT_D}!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1px;}}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{{color:{TEXT}!important;font-size:22px!important;font-weight:700;}}
/* Tabs */
.stTabs [data-baseweb="tab-list"]{{background:{BG_CARD};border-radius:8px;gap:4px;padding:4px;border:1px solid {BORDER};}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{TEXT_D};border-radius:6px;font-size:12px;padding:8px 16px;}}
.stTabs [aria-selected="true"]{{background:{BLUE}!important;color:{BTN_TEXT}!important;}}
/* Misc */
.stDataFrame{{border-radius:10px;overflow:hidden;border:1px solid {BORDER};}}
.stButton>button{{background:{BLUE};color:{BTN_TEXT};border:none;border-radius:8px;
  font-family:'JetBrains Mono',monospace;font-weight:600;padding:10px 24px;transition:opacity .2s;}}
.stButton>button:hover{{opacity:0.8;}}
.stNumberInput input,.stSelectbox select,.stTextInput input{{background:{BG_CARD2}!important;
  border:1px solid {BORDER}!important;border-radius:6px!important;color:{TEXT}!important;}}
/* Nav active state */
.nav-active .stButton>button{{
  background:{BLUE}!important;color:{BTN_TEXT}!important;
  box-shadow:0 0 18px {BLUE}55!important;border:1px solid {BLUE}!important;
}}
.nav-inactive .stButton>button{{
  background:{BG_CARD}!important;color:{TEXT_D}!important;
  border:1px solid {BORDER}!important;
}}
.nav-inactive .stButton>button:hover{{color:{BLUE}!important;border-color:{BLUE}!important;}}
/* KPI cards */
.kpi-card{{background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;}}
.kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;}}
.kpi-card.blue::before{{background:{BLUE};}}
.kpi-card.green::before{{background:{GREEN};}}
.kpi-card.red::before{{background:{RED};}}
.kpi-card.amber::before{{background:{AMBER};}}
.kpi-label{{font-size:10px;color:{TEXT_D};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;}}
.kpi-value{{font-size:24px;font-weight:700;color:{TEXT};line-height:1;font-family:'Syne',sans-serif;}}
.kpi-change{{font-size:12px;margin-top:6px;font-weight:600;}}
.kpi-sub{{font-size:11px;color:{TEXT_D};margin-top:4px;}}
.up{{color:{GREEN};}} .down{{color:{RED};}}
.section-hdr{{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:{TEXT_D};
  text-transform:uppercase;letter-spacing:2px;padding:6px 0;border-bottom:1px solid {BORDER};margin:20px 0 14px;}}
.insight-row{{background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {BLUE};
  border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;font-size:13px;color:{TEXT};}}
.insight-row.bull{{border-left-color:{GREEN};}}
.insight-row.bear{{border-left-color:{RED};}}
.hm-cell{{border-radius:8px;padding:10px 6px;text-align:center;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;}}
::-webkit-scrollbar{{width:4px;height:4px;}}
::-webkit-scrollbar-track{{background:{BG_DARK};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:2px;}}
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
NIFTY50 = {
    "RELIANCE":"Reliance Industries","TCS":"Tata Consultancy","HDFCBANK":"HDFC Bank",
    "INFY":"Infosys","ICICIBANK":"ICICI Bank","HINDUNILVR":"Hindustan Unilever",
    "ITC":"ITC Ltd","SBIN":"State Bank of India","BHARTIARTL":"Bharti Airtel",
    "KOTAKBANK":"Kotak Mahindra","LT":"Larsen & Toubro","HCLTECH":"HCL Technologies",
    "AXISBANK":"Axis Bank","WIPRO":"Wipro","MARUTI":"Maruti Suzuki",
    "SUNPHARMA":"Sun Pharma","ULTRACEMCO":"UltraTech Cement","BAJFINANCE":"Bajaj Finance",
    "TATAMOTORS":"Tata Motors","TECHM":"Tech Mahindra","TITAN":"Titan Company",
    "NESTLEIND":"Nestle India","ASIANPAINT":"Asian Paints","POWERGRID":"Power Grid",
    "NTPC":"NTPC Ltd","ONGC":"ONGC","JSWSTEEL":"JSW Steel","TATASTEEL":"Tata Steel",
    "COALINDIA":"Coal India","DRREDDY":"Dr Reddy's","CIPLA":"Cipla",
    "DIVISLAB":"Divi's Labs","BAJAJFINSV":"Bajaj Finserv","EICHERMOT":"Eicher Motors",
    "HINDALCO":"Hindalco","ADANIPORTS":"Adani Ports","M&M":"Mahindra & Mahindra",
    "GRASIM":"Grasim Industries","BPCL":"Bharat Petroleum","INDUSINDBK":"IndusInd Bank",
}
SECTOR_MAP = {
    "RELIANCE":"Energy","ONGC":"Energy","BPCL":"Energy","COALINDIA":"Energy",
    "TCS":"IT","INFY":"IT","HCLTECH":"IT","WIPRO":"IT","TECHM":"IT",
    "HDFCBANK":"Banking","ICICIBANK":"Banking","SBIN":"Banking","KOTAKBANK":"Banking",
    "AXISBANK":"Banking","INDUSINDBK":"Banking","BAJFINANCE":"Banking","BAJAJFINSV":"Banking",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG",
    "BHARTIARTL":"Telecom","LT":"Infra","POWERGRID":"Infra","NTPC":"Infra","ADANIPORTS":"Infra",
    "MARUTI":"Auto","TATAMOTORS":"Auto","EICHERMOT":"Auto","M&M":"Auto",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "TITAN":"Consumer","ASIANPAINT":"Consumer","ULTRACEMCO":"Cement","GRASIM":"Cement",
    "JSWSTEEL":"Metals","TATASTEEL":"Metals","HINDALCO":"Metals",
}

@st.cache_data(ttl=300)
def fetch_ohlcv(sym, period="1y"):
    if YF_AVAILABLE:
        try:
            df = yf.download(sym+".NS", period=period, progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
                return df[["Open","High","Low","Close","Volume"]].dropna()
        except: pass
    return _synth(sym, period)

@st.cache_data(ttl=300)
def fetch_index(sym, period="6mo"):
    if YF_AVAILABLE:
        try:
            df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
                return df[["Open","High","Low","Close","Volume"]].dropna()
        except: pass
    return _synth(sym, period)

@st.cache_data(ttl=300)
def fetch_quote(sym):
    if YF_AVAILABLE:
        try:
            info = yf.Ticker(sym+".NS").fast_info
            return {"price":round(info.last_price,2),"prev":round(info.previous_close,2),
                    "high":round(info.day_high,2),"low":round(info.day_low,2),
                    "vol":int(info.three_month_average_volume or 0),"mktcap":info.market_cap}
        except: pass
    return _synth_quote(sym)

def _synth(sym, period):
    np.random.seed(abs(hash(sym))%9999)
    days = {"1mo":22,"3mo":66,"6mo":130,"1y":252,"2y":504}.get(period,252)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)
    base = np.random.uniform(400,4000)
    rets = np.random.normal(0.0003,0.013,days)
    c = base*np.cumprod(1+rets)
    h=c*(1+np.abs(np.random.normal(0,0.007,days)))
    l=c*(1-np.abs(np.random.normal(0,0.007,days)))
    o=c*(1+np.random.normal(0,0.004,days))
    v=np.random.randint(500000,8000000,days).astype(float)
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Volume":v},index=dates)

def _synth_quote(sym):
    np.random.seed(abs(hash(sym))%9999)
    p=round(np.random.uniform(200,4000),2); pv=round(p*np.random.uniform(0.97,1.03),2)
    return {"price":p,"prev":pv,"high":round(p*1.015,2),"low":round(p*0.985,2),
            "vol":int(np.random.randint(1e6,1e7)),"mktcap":p*1e9}

def indicators(df):
    df=df.copy(); c=df["Close"]
    df["SMA20"]=c.rolling(20).mean(); df["SMA50"]=c.rolling(50).mean(); df["SMA200"]=c.rolling(200).mean()
    df["EMA12"]=c.ewm(span=12,adjust=False).mean(); df["EMA26"]=c.ewm(span=26,adjust=False).mean()
    df["MACD"]=df["EMA12"]-df["EMA26"]
    df["Signal"]=df["MACD"].ewm(span=9,adjust=False).mean(); df["Hist"]=df["MACD"]-df["Signal"]
    d=c.diff(); g=d.clip(lower=0).rolling(14).mean(); ls=(-d.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+g/ls.replace(0,np.nan)))
    df["BB_M"]=c.rolling(20).mean(); std=c.rolling(20).std()
    df["BB_U"]=df["BB_M"]+2*std; df["BB_L"]=df["BB_M"]-2*std
    hl=df["High"]-df["Low"]; hpc=(df["High"]-c.shift()).abs(); lpc=(df["Low"]-c.shift()).abs()
    df["ATR"]=pd.concat([hl,hpc,lpc],axis=1).max(axis=1).rolling(14).mean()
    df["OBV"]=(np.sign(c.diff())*df["Volume"]).fillna(0).cumsum()
    df["VolSMA"]=df["Volume"].rolling(20).mean()
    df["Ret"]=c.pct_change(); df["Vol20"]=df["Ret"].rolling(20).std()*np.sqrt(252)
    for lag in [1,2,3,5,10]: df[f"L{lag}"]=c.shift(lag)
    df["Range"]=df["High"]-df["Low"]; df["PxChg"]=c-df["Open"]
    df["W52H"]=c.rolling(min(252,len(c))).max(); df["W52L"]=c.rolling(min(252,len(c))).min()
    return df

def signal_engine(df):
    row=df.dropna(subset=["RSI","MACD","Signal","SMA20","SMA50"]).iloc[-1]
    score=0; sigs=[]
    rsi=row["RSI"]
    if rsi<30:   score+=2; sigs.append(("RSI Oversold","BUY",f"{rsi:.1f}"))
    elif rsi>70: score-=2; sigs.append(("RSI Overbought","SELL",f"{rsi:.1f}"))
    else:        sigs.append(("RSI Neutral","HOLD",f"{rsi:.1f}"))
    if row["MACD"]>row["Signal"]: score+=1; sigs.append(("MACD Crossover","BUY",f"{row['MACD']:.2f}"))
    else:                         score-=1; sigs.append(("MACD Crossover","SELL",f"{row['MACD']:.2f}"))
    if row["Close"]>row["SMA50"]: score+=1; sigs.append(("50 SMA Trend","BUY","Above"))
    else:                         score-=1; sigs.append(("50 SMA Trend","SELL","Below"))
    if row["Close"]>row["BB_M"]:  score+=0.5; sigs.append(("Bollinger Band","BUY","Above mid"))
    else:                         score-=0.5; sigs.append(("Bollinger Band","SELL","Below mid"))
    overall=("STRONG BUY" if score>=2 else "BUY" if score>=1 else
             "STRONG SELL" if score<=-2 else "SELL" if score<=-1 else "HOLD")
    return {"sigs":sigs,"score":score,"overall":overall,"rsi":rsi,
            "macd":row["MACD"],"close":row["Close"],"sma20":row["SMA20"],"sma50":row["SMA50"]}

def run_ml(df):
    if not SK_AVAILABLE: return None,None,None,None
    feats=["SMA20","SMA50","EMA12","EMA26","RSI","MACD","ATR","BB_U","BB_L",
           "Volume","L1","L2","L3","L5","Range","PxChg","Ret","Vol20"]
    feats=[f for f in feats if f in df.columns]
    data=df[feats+["Close"]].dropna()
    if len(data)<80: return None,None,None,None
    X,y=data[feats].values,data["Close"].values
    sc=MinMaxScaler(); Xs=sc.fit_transform(X)
    sp=int(len(Xs)*0.8); Xtr,Xte,ytr,yte=Xs[:sp],Xs[sp:],y[:sp],y[sp:]
    models={"Linear Regression":LinearRegression(),
            "Random Forest":RandomForestRegressor(n_estimators=150,random_state=42),
            "Gradient Boosting":GradientBoostingRegressor(n_estimators=150,random_state=42)}
    res={}
    for name,mdl in models.items():
        mdl.fit(Xtr,ytr); p=mdl.predict(Xte)
        res[name]={"model":mdl,"preds":p,"rmse":round(np.sqrt(mean_squared_error(yte,p)),2),
                   "r2":round(r2_score(yte,p),4),"yte":yte}
    best=min(res,key=lambda k:res[k]["rmse"])
    ri=feats.index("L1") if "L1" in feats else 0
    cur=sc.transform(X[-1:]); fc=[]
    for _ in range(30):
        p=res[best]["model"].predict(cur)[0]; fc.append(p); cur[0,ri]=p
    return res,best,fc,feats

# ════════════════════════════════════════════════════════════════════════════
# TOP HEADER
# ════════════════════════════════════════════════════════════════════════════
q_main = fetch_quote(st.session_state.stock)
chg_m  = ((q_main["price"]-q_main["prev"])/q_main["prev"])*100

st.markdown(f"""
<div style='background:{BG_CARD};border-bottom:1px solid {BORDER};padding:14px 32px;
     display:flex;align-items:center;justify-content:space-between;'>
  <div>
    <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                color:{TEXT};letter-spacing:1px;'>🇮🇳 NSE TERMINAL</div>
    <div style='font-size:9px;color:{TEXT_D};letter-spacing:2.5px;margin-top:2px;'>
      NSE · BSE · REAL-TIME · ML-POWERED PREDICTIONS</div>
  </div>
  <div style='display:flex;align-items:center;gap:32px;'>
    <div style='text-align:right;'>
      <div style='font-size:10px;color:{TEXT_D};letter-spacing:1px;'>
        {st.session_state.stock} &nbsp;·&nbsp; {NIFTY50.get(st.session_state.stock,"")[:20]}</div>
      <div style='font-size:22px;font-weight:800;color:{TEXT};font-family:Syne,sans-serif;line-height:1.15;'>
        ₹{q_main["price"]:,.2f}
        <span style='font-size:13px;font-weight:600;color:{"#00E676" if chg_m>=0 else "#FF1744"};margin-left:8px;'>
          {"▲" if chg_m>=0 else "▼"} {abs(chg_m):.2f}% today</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# CONTROL BAR  ─  Nav tabs | Stock | Period | Theme
# ════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("⬛ Overview",    "Market Overview"),
    ("📊 Analysis",   "Stock Analysis"),
    ("🤖 ML Predict", "ML Prediction"),
    ("🔥 Heatmap",    "Market Heatmap"),
    ("💡 Signals",    "Signals"),
    ("💰 Portfolio",  "Portfolio"),
]

st.markdown(f"""
<div style='background:{BG_CARD2};border-bottom:1px solid {BORDER};
     padding:8px 32px 8px;'>
""", unsafe_allow_html=True)

# One row: 6 nav buttons | gap | stock selector | period | theme
col_widths = [1.05, 1.05, 1.15, 1.05, 1.05, 1.05,  0.2,  1.7,  0.7,  2.0]
ctrl_cols  = st.columns(col_widths)

for i, (label, key) in enumerate(NAV_ITEMS):
    is_active = st.session_state.menu == key
    wrapper_class = "nav-active" if is_active else "nav-inactive"
    with ctrl_cols[i]:
        st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.menu = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# col index 6 = spacer (empty)

with ctrl_cols[7]:   # Stock selector
    stock_keys = list(NIFTY50.keys())
    new_stock = st.selectbox(
        "", stock_keys,
        index=stock_keys.index(st.session_state.stock),
        format_func=lambda x: f"📌 {x}  {NIFTY50[x][:13]}",
        label_visibility="collapsed", key="stock_sel"
    )
    if new_stock != st.session_state.stock:
        st.session_state.stock = new_stock; st.rerun()

with ctrl_cols[8]:   # Period
    periods = ["1mo","3mo","6mo","1y","2y"]
    new_period = st.selectbox(
        "", periods,
        index=periods.index(st.session_state.period),
        label_visibility="collapsed", key="period_sel"
    )
    if new_period != st.session_state.period:
        st.session_state.period = new_period; st.rerun()

with ctrl_cols[9]:   # Theme
    new_theme = st.selectbox(
        "", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed", key="theme_sel"
    )
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme; st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

menu   = st.session_state.menu
stock  = st.session_state.stock
period = st.session_state.period

# ════════════════════════════════════════════════════════════════════════════
# MARKET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if menu == "Market Overview":
    st.markdown('<div class="section-hdr">Key Indices</div>', unsafe_allow_html=True)
    idx_data = {"^NSEI":"NIFTY 50","^BSESN":"SENSEX","^NSEBANK":"BANK NIFTY","^CNXIT":"NIFTY IT"}
    cols = st.columns(4)
    card_clrs = ["blue","green","red","amber"]
    for i,(sym,name) in enumerate(idx_data.items()):
        df_i = fetch_index(sym,"5d")
        if len(df_i)>=2:
            cur=df_i["Close"].iloc[-1]; prv=df_i["Close"].iloc[-2]
            chg=((cur-prv)/prv)*100
            cols[i].markdown(f"""
            <div class="kpi-card {card_clrs[i]}">
              <div class="kpi-label">{name}</div>
              <div class="kpi-value">{cur:,.2f}</div>
              <div class="kpi-change {'up' if chg>=0 else 'down'}">
                {"▲" if chg>=0 else "▼"} {abs(chg):.2f}%</div>
              <div class="kpi-sub">{"↑ Advancing" if chg>=0 else "↓ Declining"}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="section-hdr">Nifty 50 — {period} Performance</div>', unsafe_allow_html=True)
    df_n = fetch_index("^NSEI", period)
    if not df_n.empty:
        fig,axes=plt.subplots(3,1,figsize=(14,9),gridspec_kw={"height_ratios":[4,1.5,1.5]},sharex=True)
        fig.patch.set_facecolor(BG_CARD)
        ax=axes[0]
        ax.plot(df_n.index,df_n["Close"],color=BLUE,lw=1.8,zorder=3)
        ax.fill_between(df_n.index,df_n["Close"],df_n["Close"].min()*0.998,alpha=0.12,color=BLUE)
        ax.set_ylabel("Price (₹)",fontsize=11); ax.grid(True,alpha=0.2)
        ax.set_title("NIFTY 50 — Price · Volume · Daily Returns",fontsize=13,fontweight="bold",pad=12)
        vc=[GREEN if r>=0 else RED for r in df_n["Close"].pct_change().fillna(0)]
        axes[1].bar(df_n.index,df_n["Volume"]/1e6,color=vc,alpha=0.7,width=0.8)
        axes[1].set_ylabel("Vol (M)",fontsize=10); axes[1].grid(True,alpha=0.15)
        rets=df_n["Close"].pct_change()*100
        axes[2].bar(df_n.index,rets,color=[GREEN if r>=0 else RED for r in rets.fillna(0)],alpha=0.8,width=0.8)
        axes[2].axhline(0,color=MUTED,lw=0.8,ls="--")
        axes[2].set_ylabel("Ret %",fontsize=10); axes[2].grid(True,alpha=0.15)
        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Live Snapshot — Top 20 Nifty Stocks</div>', unsafe_allow_html=True)
    rows=[]
    for sym in list(NIFTY50.keys())[:20]:
        q=fetch_quote(sym); chg=((q["price"]-q["prev"])/q["prev"])*100
        rows.append({"Symbol":sym,"Company":NIFTY50[sym],"Sector":SECTOR_MAP.get(sym,"—"),
                     "LTP (₹)":f"{q['price']:,.2f}",
                     "Change":f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%",
                     "High":f"{q['high']:,.2f}","Low":f"{q['low']:,.2f}"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Stock Analysis":
    st.markdown(f'<div class="section-hdr">{stock} · {NIFTY50[stock]} · Full Technical View</div>',
                unsafe_allow_html=True)
    df=fetch_ohlcv(stock,period); df=indicators(df); q=fetch_quote(stock)
    chg=((q["price"]-q["prev"])/q["prev"])*100
    c1,c2,c3,c4,c5=st.columns(5)
    for col,lbl,val,dlt in [
        (c1,"LTP (₹)",f"{q['price']:,.2f}",f"Prev ₹{q['prev']:,.2f}"),
        (c2,"Change",f"{'▲' if chg>=0 else '▼'} {abs(chg):.2f}%","Today"),
        (c3,"Day High",f"₹{q['high']:,.2f}","Intraday"),
        (c4,"Day Low",f"₹{q['low']:,.2f}","Intraday"),
        (c5,"Sector",SECTOR_MAP.get(stock,"—"),"Classification"),
    ]: col.metric(lbl,val,dlt)

    st.markdown('<div class="section-hdr">Price + Technical Indicators</div>', unsafe_allow_html=True)
    df_p=df.loc[df.index>=df.index[-1]-pd.Timedelta(days=120)] if len(df)>120 else df
    fig,axes=plt.subplots(4,1,figsize=(14,14),gridspec_kw={"height_ratios":[4,1.8,1.8,1.5]},sharex=True)
    fig.patch.set_facecolor(BG_CARD)
    ax=axes[0]
    ax.plot(df_p.index,df_p["Close"],color=BLUE,lw=2,label="Close",zorder=4)
    ax.plot(df_p.index,df_p["SMA20"],color=AMBER,lw=1.2,ls="--",alpha=0.85,label="SMA 20")
    ax.plot(df_p.index,df_p["SMA50"],color=BROWN,lw=1.2,ls="--",alpha=0.85,label="SMA 50")
    ax.fill_between(df_p.index,df_p["BB_U"],df_p["BB_L"],alpha=0.07,color=BLUE,label="BB Bands")
    ax.plot(df_p.index,df_p["BB_U"],color=BLUE,lw=0.6,alpha=0.4)
    ax.plot(df_p.index,df_p["BB_L"],color=BLUE,lw=0.6,alpha=0.4)
    ax.fill_between(df_p.index,df_p["Close"],df_p["Close"].min()*0.998,alpha=0.08,color=BLUE)
    ax.legend(loc="upper left",fontsize=9,framealpha=0.3)
    ax.set_ylabel("Price (₹)",fontsize=11)
    ax.set_title(f"{stock} — Technical Dashboard",fontsize=13,fontweight="bold",pad=12)
    ax.grid(True,alpha=0.15)
    ax2=axes[1]
    colors_h=[GREEN if v>=0 else RED for v in df_p["Hist"].fillna(0)]
    ax2.bar(df_p.index,df_p["Hist"],color=colors_h,alpha=0.75,width=0.7)
    ax2.plot(df_p.index,df_p["MACD"],color=BLUE,lw=1.3,label="MACD")
    ax2.plot(df_p.index,df_p["Signal"],color=RED,lw=1.3,ls="--",label="Signal")
    ax2.axhline(0,color=MUTED,lw=0.8); ax2.legend(fontsize=9,framealpha=0.3)
    ax2.set_ylabel("MACD",fontsize=10); ax2.grid(True,alpha=0.15)
    ax3=axes[2]
    ax3.plot(df_p.index,df_p["RSI"],color="#CE93D8",lw=1.5)
    ax3.fill_between(df_p.index,df_p["RSI"],30,where=df_p["RSI"]<30,alpha=0.3,color=GREEN)
    ax3.fill_between(df_p.index,df_p["RSI"],70,where=df_p["RSI"]>70,alpha=0.3,color=RED)
    ax3.axhline(70,color=RED,lw=1,ls=":",alpha=0.6); ax3.axhline(30,color=GREEN,lw=1,ls=":",alpha=0.6)
    ax3.axhline(50,color=MUTED,lw=0.6,ls="--",alpha=0.5)
    ax3.set_ylim(0,100); ax3.set_ylabel("RSI",fontsize=10); ax3.grid(True,alpha=0.15)
    ax4=axes[3]
    vc=[GREEN if c>=o else RED for c,o in zip(df_p["Close"],df_p["Open"])]
    ax4.bar(df_p.index,df_p["Volume"]/1e6,color=vc,alpha=0.7,width=0.7)
    ax4.plot(df_p.index,df_p["VolSMA"]/1e6,color=BLUE,lw=1.2,ls="--",alpha=0.8,label="Vol SMA")
    ax4.set_ylabel("Vol (M)",fontsize=10); ax4.legend(fontsize=9,framealpha=0.3)
    ax4.grid(True,alpha=0.15)
    plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Key Statistics</div>', unsafe_allow_html=True)
    row=df.dropna().iloc[-1]
    def safe(v,fmt="₹{:,.2f}"): return fmt.format(v) if not (isinstance(v,float) and np.isnan(v)) else "N/A"
    stats=[("RSI (14)",f"{row['RSI']:.1f}"),("MACD",f"{row['MACD']:.2f}"),
           ("SMA 20",safe(row['SMA20'])),("SMA 50",safe(row['SMA50'])),("SMA 200",safe(row['SMA200'])),
           ("BB Upper",safe(row['BB_U'])),("BB Lower",safe(row['BB_L'])),("ATR (14)",safe(row['ATR'])),
           ("52W High",safe(row['W52H'])),("52W Low",safe(row['W52L'])),
           ("Ann. Volatility",f"{row['Vol20']*100:.1f}%"),("OBV",f"{row['OBV']/1e6:.1f}M")]
    sc1,sc2=st.columns(2); h=len(stats)//2
    sc1.dataframe(pd.DataFrame(stats[:h],columns=["Indicator","Value"]).set_index("Indicator"),use_container_width=True)
    sc2.dataframe(pd.DataFrame(stats[h:],columns=["Indicator","Value"]).set_index("Indicator"),use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif menu == "ML Prediction":
    if not SK_AVAILABLE: st.error("pip install scikit-learn"); st.stop()
    st.markdown(f'<div class="section-hdr">ML Price Prediction — {stock}</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-row">📌 Trained on 15+ technical indicators. Educational use only — not financial advice.</div>',
                unsafe_allow_html=True)
    with st.spinner("Training models on NSE data…"):
        df=fetch_ohlcv(stock,"2y"); df=indicators(df)
        res,best,fc,feats=run_ml(df)
    if res is None: st.warning("Insufficient data."); st.stop()

    st.markdown('<div class="section-hdr">Model Leaderboard</div>', unsafe_allow_html=True)
    mc=st.columns(3)
    clr_map={"Linear Regression":BLUE,"Random Forest":GREEN,"Gradient Boosting":AMBER}
    for col,(name,r) in zip(mc,res.items()):
        col.markdown(f"""
        <div class="kpi-card {'green' if name==best else 'blue'}">
          <div class="kpi-label">{name} {"★ BEST" if name==best else ""}</div>
          <div class="kpi-value" style="color:{clr_map[name]};font-size:20px;">R² {r['r2']:.4f}</div>
          <div class="kpi-sub">RMSE: ₹{r['rmse']:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Actual vs Predicted — Test Set</div>', unsafe_allow_html=True)
    br=res[best]
    fig,ax=plt.subplots(figsize=(14,5)); fig.patch.set_facecolor(BG_CARD)
    ax.plot(br["yte"],color=BLUE_L,lw=2,label="Actual",alpha=0.9)
    ax.plot(br["preds"],color=GREEN,lw=2,ls="--",label=f"Predicted ({best})",alpha=0.9)
    ax.fill_between(range(len(br["yte"])),br["yte"],br["preds"],alpha=0.1,color=AMBER)
    ax.set_title(f"Model Performance — {best}",fontsize=13,fontweight="bold",pad=12)
    ax.set_ylabel("Price (₹)",fontsize=11); ax.legend(fontsize=10,framealpha=0.3); ax.grid(True,alpha=0.15)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">30-Day Price Forecast</div>', unsafe_allow_html=True)
    last=df["Close"].iloc[-1]
    fdates=pd.bdate_range(start=df.index[-1]+pd.Timedelta(days=1),periods=30)
    hist=df["Close"].loc[df.index>=df.index[-1]-pd.Timedelta(days=60)]
    fig,ax=plt.subplots(figsize=(14,5)); fig.patch.set_facecolor(BG_CARD)
    ax.plot(hist.index,hist.values,color=BLUE,lw=2,label="Historical")
    ax.plot(fdates,fc,color=GREEN,lw=2,ls="--",label="Forecast",zorder=4)
    ax.fill_between(fdates,[p*0.97 for p in fc],[p*1.03 for p in fc],alpha=0.18,color=GREEN,label="±3% band")
    ax.axvline(df.index[-1],color=RED,lw=1.5,ls=":",alpha=0.7,label="Today")
    ax.set_title(f"30-Day Forecast — {stock}  |  Model: {best}",fontsize=13,fontweight="bold",pad=12)
    ax.set_ylabel("Price (₹)",fontsize=11); ax.legend(fontsize=10,framealpha=0.3); ax.grid(True,alpha=0.15)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    f_chg=((fc[-1]-last)/last)*100
    st.markdown(f'<div class="insight-row {"bull" if f_chg>0 else "bear"}">{"📈" if f_chg>0 else "📉"} <b>Forecast:</b> {stock} projected at <b>₹{fc[-1]:,.2f}</b> in 30 days ({("+" if f_chg>=0 else "")}{f_chg:.2f}% from ₹{last:,.2f})</div>',
                unsafe_allow_html=True)

    if best in ("Random Forest","Gradient Boosting"):
        st.markdown('<div class="section-hdr">Feature Importance</div>', unsafe_allow_html=True)
        fi=pd.Series(res[best]["model"].feature_importances_,index=feats).sort_values().tail(12)
        bar_clrs=[BLUE if v<fi.quantile(0.66) else (AMBER if v<fi.quantile(0.85) else GREEN) for v in fi.values]
        fig,ax=plt.subplots(figsize=(11,5)); fig.patch.set_facecolor(BG_CARD)
        bars=ax.barh(fi.index,fi.values,color=bar_clrs,edgecolor=BG_CARD,height=0.65)
        for bar,val in zip(bars,fi.values):
            ax.text(val+0.001,bar.get_y()+bar.get_height()/2,f"{val:.3f}",va="center",fontsize=9,color=TEXT_D)
        ax.set_xlabel("Importance Score",fontsize=11)
        ax.set_title(f"Feature Importances — {best}",fontsize=13,fontweight="bold",pad=12)
        ax.grid(axis="x",alpha=0.15); ax.spines["left"].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# HEATMAP
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Market Heatmap":
    st.markdown('<div class="section-hdr">Nifty 50 — Live Market Heatmap</div>', unsafe_allow_html=True)
    with st.spinner("Loading market data…"):
        syms=list(NIFTY50.keys())[:36]; hdata=[]
        for sym in syms:
            q=fetch_quote(sym); chg=((q["price"]-q["prev"])/q["prev"])*100
            hdata.append({"sym":sym,"chg":round(chg,2),"sector":SECTOR_MAP.get(sym,"Other")})
    hdf=pd.DataFrame(hdata)
    NCOLS=6; rows_=[syms[i:i+NCOLS] for i in range(0,len(syms),NCOLS)]
    for row_ in rows_:
        cols=st.columns(NCOLS)
        for col,sym in zip(cols,row_):
            chg=hdf.loc[hdf["sym"]==sym,"chg"].values[0]
            if   chg>=2:   bg,fg="#003300",GREEN
            elif chg>=0.5: bg,fg="#004D00","#81C784"
            elif chg>=0:   bg,fg="#0A1F0A","#A5D6A7"
            elif chg>=-0.5:bg,fg="#1F0A0A","#EF9A9A"
            elif chg>=-2:  bg,fg="#4D0000","#EF5350"
            else:          bg,fg="#330000",RED
            col.markdown(f"""
            <div class="hm-cell" style="background:{bg};border:1px solid {BORDER};">
              <div style="color:{TEXT_D};font-size:10px;">{sym}</div>
              <div style="color:{fg};font-size:13px;margin-top:4px;">{"+" if chg>=0 else ""}{chg:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Sector Performance</div>', unsafe_allow_html=True)
    sp=hdf.groupby("sector")["chg"].mean().sort_values()
    fig,axes=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(BG_CARD)
    sc_=[GREEN if v>=0 else RED for v in sp.values]
    axes[0].barh(sp.index,sp.values,color=sc_,height=0.6,edgecolor=BG_CARD)
    axes[0].axvline(0,color=MUTED,lw=0.8,ls="--")
    for i,(v,nm) in enumerate(zip(sp.values,sp.index)):
        axes[0].text(v+(0.04 if v>=0 else -0.04),i,f"{v:+.2f}%",va="center",
                     ha="left" if v>=0 else "right",fontsize=10,color=GREEN if v>=0 else RED)
    axes[0].set_title("Avg Sector % Change",fontsize=13,fontweight="bold",pad=12)
    axes[0].set_xlabel("% Change",fontsize=11); axes[0].grid(axis="x",alpha=0.15)
    axes[0].spines["left"].set_visible(False)
    sc2=hdf["sector"].value_counts()
    pie_clrs=[BLUE,GREEN,RED,AMBER,BROWN,"#7B1FA2","#00838F","#558B2F","#E65100"]
    wedges,_,ats=axes[1].pie(sc2.values,labels=sc2.index,colors=pie_clrs[:len(sc2)],
                               autopct="%1.0f%%",startangle=140,textprops={"fontsize":10,"color":TEXT})
    for at in ats: at.set_color("#111" if IS_LIGHT else BG_DARK); at.set_fontweight("bold")
    axes[1].set_title("Sector Composition",fontsize=13,fontweight="bold",pad=12)
    plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Signals":
    st.markdown(f'<div class="section-hdr">Trading Signal — {stock}</div>', unsafe_allow_html=True)
    with st.spinner("Analysing indicators…"):
        df=fetch_ohlcv(stock,period); df=indicators(df); sig=signal_engine(df)
    clr_map={"STRONG BUY":GREEN,"BUY":GREEN_D,"HOLD":AMBER,"SELL":RED_D,"STRONG SELL":RED}
    sc=sig["overall"]; sc_clr=clr_map.get(sc,BLUE)
    st.markdown(f"""
    <div style='background:{BG_CARD};border:2px solid {sc_clr};border-radius:14px;
         padding:28px;text-align:center;margin:12px 0;'>
      <div style='font-size:11px;color:{TEXT_D};letter-spacing:2px;'>OVERALL SIGNAL</div>
      <div style='font-size:48px;font-weight:800;color:{sc_clr};font-family:Syne,sans-serif;
                  letter-spacing:2px;margin:8px 0;'>{sc}</div>
      <div style='font-size:13px;color:{TEXT_D};'>Composite Score:
        <span style='color:{sc_clr};font-weight:700;'>{sig["score"]:.1f}</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Indicator Breakdown</div>', unsafe_allow_html=True)
    for label,action,value in sig["sigs"]:
        bc=GREEN if action=="BUY" else (RED if action=="SELL" else AMBER)
        bg_a=("rgba(0,230,118,0.15)" if action=="BUY" else
              ("rgba(255,23,68,0.15)" if action=="SELL" else "rgba(255,214,0,0.15)"))
        st.markdown(f"""
        <div style='background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {bc};
             border-radius:0 10px 10px 0;padding:12px 18px;margin:6px 0;
             display:flex;justify-content:space-between;align-items:center;'>
          <div>
            <div style='font-size:13px;color:{TEXT};font-weight:500;'>{label}</div>
            <div style='font-size:11px;color:{TEXT_D};margin-top:2px;'>Value: {value}</div>
          </div>
          <span style='background:{bg_a};color:{bc};border:1px solid {bc};
                       padding:4px 14px;border-radius:20px;font-size:11px;
                       font-weight:700;letter-spacing:1px;'>{action}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">RSI Gauge</div>', unsafe_allow_html=True)
    fig,ax=plt.subplots(figsize=(10,2)); fig.patch.set_facecolor(BG_CARD)
    ax.barh(0,100,color=BG_CARD2,height=0.5)
    rc=GREEN if sig["rsi"]<30 else (RED if sig["rsi"]>70 else BLUE)
    ax.barh(0,sig["rsi"],color=rc,height=0.5,alpha=0.85)
    ax.axvline(30,color=GREEN,lw=2,ls="--",alpha=0.7)
    ax.axvline(70,color=RED,lw=2,ls="--",alpha=0.7)
    ax.axvline(50,color=MUTED,lw=1,ls=":",alpha=0.5)
    ax.text(15,0.45,"Oversold",ha="center",fontsize=9,color=GREEN,va="bottom")
    ax.text(85,0.45,"Overbought",ha="center",fontsize=9,color=RED,va="bottom")
    ax.text(sig["rsi"],-0.38,f"RSI: {sig['rsi']:.1f}",ha="center",fontsize=11,fontweight="bold",color=rc)
    ax.set_xlim(0,100); ax.set_ylim(-0.6,0.7); ax.axis("off")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-hdr">Signal Scan — Top 12 Stocks</div>', unsafe_allow_html=True)
    scan=[]
    with st.spinner("Scanning market…"):
        for sym in list(NIFTY50.keys())[:12]:
            try:
                ds=fetch_ohlcv(sym,"3mo"); ds=indicators(ds); s=signal_engine(ds)
                scan.append({"Symbol":sym,"Company":NIFTY50[sym][:20],
                              "Signal":s["overall"],"RSI":f"{s['rsi']:.1f}",
                              "MACD":f"{s['macd']:.2f}","Score":f"{s['score']:.1f}"})
            except: pass
    st.dataframe(pd.DataFrame(scan),use_container_width=True,hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TOOLS
# ════════════════════════════════════════════════════════════════════════════
elif menu == "Portfolio":
    st.markdown('<div class="section-hdr">Portfolio & Investment Tools</div>', unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["📅  SIP Calculator","💵  Lumpsum","📂  Portfolio Builder"])

    with tab1:
        c1,c2,c3=st.columns(3)
        amt=c1.number_input("Monthly SIP (₹)",500,500000,10000,500)
        rate=c2.slider("Expected Return (% p.a.)",6.0,30.0,12.0,0.5)
        yrs=c3.slider("Duration (Years)",1,40,10)
        r=rate/100/12; n=yrs*12
        corpus=amt*((pow(1+r,n)-1)/r)*(1+r); inv=amt*n; profit=corpus-inv
        m1,m2,m3=st.columns(3)
        m1.metric("Total Invested",f"₹{inv:,.0f}")
        m2.metric("Estimated Corpus",f"₹{corpus:,.0f}")
        m3.metric("Wealth Gained",f"₹{profit:,.0f}",f"+{profit/inv*100:.0f}%")
        ya=list(range(1,yrs+1))
        ia=[amt*y*12 for y in ya]; ca=[amt*((pow(1+r,y*12)-1)/r)*(1+r) for y in ya]
        fig,ax=plt.subplots(figsize=(12,4)); fig.patch.set_facecolor(BG_CARD)
        ax.fill_between(ya,ia,alpha=0.5,color=BLUE,label="Invested")
        ax.fill_between(ya,ca,alpha=0.4,color=GREEN,label="Corpus")
        ax.plot(ya,ca,color=GREEN,lw=2); ax.plot(ya,ia,color=BLUE,lw=1.5,ls="--")
        ax.set_xlabel("Years",fontsize=11); ax.set_ylabel("Amount (₹)",fontsize=11)
        ax.set_title("SIP Growth Projection",fontsize=13,fontweight="bold",pad=12)
        ax.legend(fontsize=10,framealpha=0.3); ax.grid(True,alpha=0.15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1e5:.1f}L"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        c1,c2,c3=st.columns(3)
        ls_amt=c1.number_input("Investment (₹)",1000,10000000,100000,1000)
        ls_rate=c2.slider("Return (% p.a.)",6.0,30.0,14.0,0.5,key="ls_r")
        ls_yrs=c3.slider("Years",1,40,10,key="ls_y")
        val=ls_amt*pow(1+ls_rate/100,ls_yrs)
        m1,m2,m3=st.columns(3)
        m1.metric("Principal",f"₹{ls_amt:,.0f}")
        m2.metric("Maturity",f"₹{val:,.0f}")
        m3.metric("Profit",f"₹{val-ls_amt:,.0f}",f"+{(val-ls_amt)/ls_amt*100:.0f}%")
        ya=list(range(1,ls_yrs+1)); va=[ls_amt*pow(1+ls_rate/100,y) for y in ya]
        fig,ax=plt.subplots(figsize=(12,4)); fig.patch.set_facecolor(BG_CARD)
        ax.plot(ya,va,color=GREEN,lw=2.5); ax.fill_between(ya,ls_amt,va,alpha=0.2,color=GREEN)
        ax.axhline(ls_amt,color=RED,lw=1.2,ls="--",label="Principal")
        ax.set_xlabel("Years",fontsize=11); ax.set_ylabel("Value (₹)",fontsize=11)
        ax.set_title("Lumpsum Growth",fontsize=13,fontweight="bold",pad=12)
        ax.legend(fontsize=10,framealpha=0.3); ax.grid(True,alpha=0.15)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f"₹{x/1e5:.1f}L"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        sel=st.multiselect("Select Stocks",list(NIFTY50.keys()),
                            default=["RELIANCE","TCS","HDFCBANK","INFY","SBIN"])
        if sel:
            wts={}; cols=st.columns(len(sel))
            for col,sym in zip(cols,sel):
                wts[sym]=col.slider(sym,0,100,100//len(sel),key=f"w_{sym}")
            total=sum(wts.values())
            st.markdown(f"**Total: {total}%** {'✅' if total==100 else '⚠️ Adjust to 100%'}")
            if st.button("Analyse Portfolio"):
                with st.spinner("Building portfolio…"):
                    pdf=pd.DataFrame()
                    for sym in sel:
                        d=fetch_ohlcv(sym,"1y")
                        if not d.empty: pdf[sym]=d["Close"]
                    if not pdf.empty:
                        pdf=pdf.dropna(); rets=pdf.pct_change().dropna()
                        wa=np.array([wts[s]/100 for s in sel])
                        pr=(rets*wa).sum(axis=1); cr=(1+pr).cumprod()
                        fig,axes=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(BG_CARD)
                        axes[0].plot(cr.index,cr.values,color=BLUE,lw=2)
                        axes[0].fill_between(cr.index,1,cr.values,where=cr.values>=1,alpha=0.15,color=GREEN)
                        axes[0].fill_between(cr.index,1,cr.values,where=cr.values<1,alpha=0.15,color=RED)
                        axes[0].axhline(1,color=MUTED,lw=1,ls="--")
                        axes[0].set_title("Portfolio Cumulative Return",fontsize=13,fontweight="bold",pad=12)
                        axes[0].set_ylabel("Growth of ₹1",fontsize=11); axes[0].grid(True,alpha=0.15)
                        pc=[BLUE,GREEN,RED,AMBER,BROWN,"#7B1FA2","#00838F"]
                        wedges,_,ats=axes[1].pie([wts[s] for s in sel],labels=sel,
                                                  colors=pc[:len(sel)],autopct="%1.0f%%",startangle=140,
                                                  textprops={"fontsize":11,"color":TEXT})
                        for at in ats: at.set_color(BG_DARK); at.set_fontweight("bold")
                        axes[1].set_title("Allocation",fontsize=13,fontweight="bold",pad=12)
                        plt.tight_layout(pad=1.5); st.pyplot(fig); plt.close()
                        ar=pr.mean()*252*100; av=pr.std()*np.sqrt(252)*100
                        sr=ar/av if av>0 else 0; md=((cr-cr.cummax())/cr.cummax()).min()*100
                        r1,r2,r3,r4=st.columns(4)
                        r1.metric("Annual Return",f"{ar:.2f}%")
                        r2.metric("Volatility",f"{av:.2f}%")
                        r3.metric("Sharpe Ratio",f"{sr:.2f}")
                        r4.metric("Max Drawdown",f"{md:.2f}%")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='text-align:center;color:{TEXT_D};font-size:11px;padding:8px 0;letter-spacing:0.5px;'>
  NSE TERMINAL · AI/ML Course Project · Data: Yahoo Finance (NSE) · Not financial advice ·
  Python · Streamlit · scikit-learn · matplotlib
</div>""", unsafe_allow_html=True)
