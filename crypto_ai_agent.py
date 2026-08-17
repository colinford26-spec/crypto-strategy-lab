import io
from datetime import date, timedelta
import numpy as np, pandas as pd, requests, plotly.graph_objects as go, streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Crypto AI Research Agent", page_icon="🤖", layout="wide")
st.title("🤖 Crypto AI Research Agent")
st.caption("An AI-assisted research and paper-backtesting tool using daily market data. It does not place live trades or guarantee outcomes.")
COINS={"Bitcoin":"bitcoin","Ethereum":"ethereum","BNB":"binancecoin","XRP":"ripple","Solana":"solana","Cardano":"cardano","Dogecoin":"dogecoin","TRON":"tron","Avalanche":"avalanche-2","Chainlink":"chainlink"}

@st.cache_data(ttl=900,show_spinner=False)
def load_coin(coin_id,start,end):
    r=requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range",params={"vs_currency":"usd","from":int(pd.Timestamp(start,tz="UTC").timestamp()),"to":int(pd.Timestamp(end,tz="UTC").timestamp())},timeout=30); r.raise_for_status(); x=r.json()
    if not x.get("prices"): raise ValueError("CoinGecko returned no prices")
    d=pd.DataFrame(x["prices"],columns=["ms","close"]); d["timestamp"]=pd.to_datetime(d.ms,unit="ms",utc=True).dt.floor("D"); d=d.groupby("timestamp",as_index=False).close.last(); d["open"]=d.close.shift(1).fillna(d.close); d["high"]=d[["open","close"]].max(axis=1); d["low"]=d[["open","close"]].min(axis=1); return d.dropna()

def features(df):
    d=df.copy(); ret=d.close.pct_change(); d["ret1"]=ret; d["ret7"]=d.close.pct_change(7); d["ret14"]=d.close.pct_change(14); d["sma20"]=d.close.rolling(20).mean(); d["sma50"]=d.close.rolling(50).mean(); d["sma20_gap"]=d.close/d.sma20-1; d["sma50_gap"]=d.close/d.sma50-1; delta=d.close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean(); d["rsi"]=100-(100/(1+gain/loss)); d.loc[(loss==0)&(gain>0),"rsi"]=100; d["volatility"]=ret.rolling(14).std(); d["volume_change"]=d.volume.pct_change() if "volume" in d else 0; d["target"]=(d.close.shift(-1)>d.close).astype(int); return d

def run_agent(d,capital,fee,slip,size,threshold):
    cols=["ret1","ret7","ret14","sma20_gap","sma50_gap","rsi","volatility","volume_change"]; z=d.replace([np.inf,-np.inf],np.nan).dropna(subset=cols+['target']).copy(); split=max(int(len(z)*.7),1)
    if split>=len(z): raise ValueError("Not enough data after feature calculation")
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=1000,class_weight="balanced"))]); model.fit(z.iloc[:split][cols],z.iloc[:split].target); z["probability"]=np.nan; z.iloc[split:,z.columns.get_loc("probability")]=model.predict_proba(z.iloc[split:][cols])[:,1]; z["signal"]=np.where(z.probability>=threshold,1,np.where(z.probability<=1-threshold,-1,0)); cash=capital; qty=0.; entry=0.; trades=[]; equity=[]
    for _,r in z.iloc[split:].iterrows():
        price=float(r.close); action=int(r.signal)
        if action==1 and not qty: exe=price*(1+slip/100); spend=cash*size/100; qty=spend/exe; cash-=spend*(1+fee/100); entry=exe; trades.append([r.timestamp,"BUY",exe,qty,0])
        elif action==-1 and qty: exe=price*(1-slip/100); proceeds=qty*exe*(1-fee/100); pnl=proceeds-qty*entry; cash+=proceeds; trades.append([r.timestamp,"SELL",exe,qty,pnl]); qty=0; entry=0
        equity.append([r.timestamp,cash+qty*price])
    if qty: price=float(z.iloc[-1].close); proceeds=qty*price*(1-fee/100); cash+=proceeds; trades.append([z.iloc[-1].timestamp,"FINAL EXIT",price,qty,proceeds-qty*entry])
    eq=pd.DataFrame(equity,columns=["timestamp","equity"]); tr=pd.DataFrame(trades,columns=["timestamp","side","price","quantity","pnl"]); y=z.iloc[split:].target; prob=z.iloc[split:].probability; auc=roc_auc_score(y,prob) if y.nunique()>1 else np.nan; return z,eq,tr,cash,accuracy_score(y,(prob>=.5).astype(int)),auc

st.sidebar.header("1. Data")
coin=st.sidebar.selectbox("Cryptocurrency",list(COINS)); end=date.today(); start=st.sidebar.date_input("Start date",end-timedelta(days=365),min_value=end-timedelta(days=365),max_value=end); load=st.sidebar.button("Load data",type="primary")
if load:
    try: st.session_state.df=load_coin(COINS[coin],start,end+timedelta(days=1)); st.success("Data loaded")
    except Exception as e: st.error(str(e))
df=st.session_state.get("df")
if df is None: st.info("Select a coin and click Load data."); st.stop()
st.sidebar.header("2. Agent controls"); threshold=st.sidebar.slider("Minimum AI probability",.51,.90,.60,.01); capital=st.sidebar.number_input("Starting balance",100.,1000000.,1000.,100.); size=st.sidebar.slider("Position size (%)",1,100,100); fee=st.sidebar.number_input("Fee per side (%)",0.,5.,.1,.01); slip=st.sidebar.number_input("Slippage (%)",0.,5.,.05,.01)
if st.sidebar.button("Run AI agent",type="primary"):
    try: st.session_state.result=run_agent(df,capital,fee,slip,size,threshold)
    except Exception as e: st.error(str(e))
if "result" not in st.session_state: st.info("Set the controls and click Run AI agent."); st.stop()
z,eq,tr,final,acc,auc=st.session_state.result; ret=(final/capital-1)*100; dd=(eq.equity/eq.equity.cummax()-1)*100
c=st.columns(6); c[0].metric("Agent signal", "BUY" if z.signal.iloc[-1]==1 else "SELL" if z.signal.iloc[-1]==-1 else "HOLD"); c[1].metric("Latest probability",f"{z.probability.iloc[-1]*100:.1f}%" if pd.notna(z.probability.iloc[-1]) else "n/a"); c[2].metric("Final balance",f"${final:,.2f}"); c[3].metric("Return",f"{ret:.2f}%"); c[4].metric("Trades",len(tr)); c[5].metric("Test AUC",f"{auc:.3f}" if pd.notna(auc) else "n/a")
t1,t2,t3=st.tabs(["Agent view","Backtest","Trades"])
with t1:
    st.subheader("Prediction probability"); pf=go.Figure(go.Scatter(x=z.timestamp,y=z.probability*100,name="Probability")); pf.add_hline(y=threshold*100,line_dash="dash",line_color="green"); pf.add_hline(y=(1-threshold)*100,line_dash="dash",line_color="red"); pf.update_yaxes(range=[0,100],title="Probability of positive next day (%)"); st.plotly_chart(pf,width="stretch"); st.write(f"Out-of-sample accuracy: {acc:.1%}. The agent trains on the first 70% and evaluates on the later 30%.")
with t2:
    ef=go.Figure(go.Scatter(x=eq.timestamp,y=eq.equity,name="AI equity")); ef.update_layout(height=430); st.plotly_chart(ef,width="stretch"); st.download_button("Download predictions",z.to_csv(index=False),"predictions.csv","text/csv")
with t3:
    st.download_button("Download trades",tr.to_csv(index=False),"trades.csv","text/csv"); st.dataframe(tr,width="stretch")
