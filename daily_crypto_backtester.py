import io, requests
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Daily Crypto Backtester", page_icon="📈", layout="wide")
st.title("📈 Daily Crypto Backtester")
st.caption("Daily CoinGecko market data with simple strategy backtesting. Results are simulated and not financial advice.")

COINS={"Bitcoin":"bitcoin","Ethereum":"ethereum","BNB":"binancecoin","XRP":"ripple","Solana":"solana","Cardano":"cardano","Dogecoin":"dogecoin","TRON":"tron","Avalanche":"avalanche-2","Chainlink":"chainlink"}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_coingecko(coin_id, start, end):
    url=f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
    params={"vs_currency":"usd","from":int(pd.Timestamp(start,tz="UTC").timestamp()),"to":int(pd.Timestamp(end,tz="UTC").timestamp())}
    r=requests.get(url,params=params,timeout=30)
    if r.status_code==401: raise ValueError("CoinGecko requires an API key for this request. Add it in Streamlit Secrets as COINGECKO_API_KEY.")
    if r.status_code==429: raise ValueError("CoinGecko rate limit reached. Wait a few minutes and try again.")
    r.raise_for_status(); payload=r.json()
    if not payload.get("prices"): raise ValueError("CoinGecko returned no price data.")
    prices=pd.DataFrame(payload["prices"],columns=["ms","close"])
    caps=pd.DataFrame(payload.get("market_caps",[]),columns=["ms","market_cap"])
    vols=pd.DataFrame(payload.get("total_volumes",[]),columns=["ms","volume"])
    df=prices.merge(caps,on="ms",how="left").merge(vols,on="ms",how="left")
    df["timestamp"]=pd.to_datetime(df.ms,unit="ms",utc=True).dt.floor("D")
    df=df.groupby("timestamp",as_index=False).agg({"close":"last","market_cap":"last","volume":"last"})
    df["open"]=df.close.shift(1).fillna(df.close); df["high"]=df[["open","close"]].max(axis=1); df["low"]=df[["open","close"]].min(axis=1)
    return df.dropna(subset=["close"])[["timestamp","open","high","low","close","volume","market_cap"]]

def parse_upload(raw):
    try: df=pd.read_csv(io.BytesIO(raw))
    except Exception: df=pd.read_excel(io.BytesIO(raw))
    df.columns=[str(c).strip().lower().replace(" ","_") for c in df.columns]
    aliases={"date":"timestamp","datetime":"timestamp","time":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"}
    df=df.rename(columns={c:aliases.get(c,c) for c in df.columns})
    missing=[c for c in ["timestamp","open","high","low","close"] if c not in df]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    df.timestamp=pd.to_datetime(df.timestamp,utc=True,errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.dropna(subset=["timestamp","open","high","low","close"]).sort_values("timestamp").drop_duplicates("timestamp")

def indicators(df,strategy,fast,slow,rsi_period):
    d=df.copy(); d["fast_sma"]=d.close.rolling(fast).mean(); d["slow_sma"]=d.close.rolling(slow).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0).rolling(rsi_period).mean(); loss=(-delta.clip(upper=0)).rolling(rsi_period).mean(); d["rsi"]=100-(100/(1+gain/loss.replace(0,np.nan)))
    d["signal"]=np.where(d.fast_sma>d.slow_sma,1,0) if strategy=="SMA crossover" else np.where(d.rsi<30,1,np.where(d.rsi>70,-1,0))
    return d

def backtest(d,capital,fee,slip,size,stop,take):
    cash=capital; qty=0.; entry=0.; trades=[]; equity=[]
    for _,r in d.iterrows():
        price=float(r.close); action=0
        if qty and ((stop > 0 and price <= entry * (1 - stop / 100)) or (take > 0 and price >= entry * (1 + take / 100))):
            action = -1
        elif not qty and r.signal==1: action=1
        elif qty and r.signal==-1: action=-1
        if action==1:
            execution=price*(1+slip/100); spend=cash*size/100; qty=spend/execution; cash-=spend*(1+fee/100); entry=execution; trades.append([r.timestamp,"BUY",execution,qty,0.0])
        elif action==-1 and qty:
            execution=price*(1-slip/100); proceeds=qty*execution*(1-fee/100); pnl=proceeds-qty*entry; cash+=proceeds; trades.append([r.timestamp,"SELL",execution,qty,pnl]); qty=0.; entry=0.
        equity.append([r.timestamp,cash+qty*price])
    if qty:
        price=float(d.iloc[-1].close); proceeds=qty*price*(1-fee/100); trades.append([d.iloc[-1].timestamp,"FINAL EXIT",price,qty,proceeds-qty*entry]); cash+=proceeds
    return pd.DataFrame(equity,columns=["timestamp","equity"]),pd.DataFrame(trades,columns=["timestamp","side","price","quantity","pnl"]),cash

st.sidebar.header("Data source")
source=st.sidebar.radio("Choose data",["CoinGecko daily data","Upload CSV/XLSX"])
if source=="CoinGecko daily data":
    coin_name=st.sidebar.selectbox("Cryptocurrency",list(COINS)); coin_id=COINS[coin_name]
    end=date.today(); start=st.sidebar.date_input("Start date",end-timedelta(days=365),min_value=end-timedelta(days=365),max_value=end)
    st.sidebar.caption("Free historical range limited to the last 365 days.")
    if st.sidebar.button("Load CoinGecko data",type="primary"):
        try: st.session_state.df=fetch_coingecko(coin_id,start,end+timedelta(days=1)); st.session_state.dataset=coin_name; st.success(f"Loaded {len(st.session_state.df):,} daily records")
        except Exception as e: st.error(str(e))
else:
    files=st.sidebar.file_uploader("CSV or Excel candle files",type=["csv","xlsx"],accept_multiple_files=False)
    if files:
        try: st.session_state.df=parse_upload(files.getvalue()); st.session_state.dataset=files.name
        except Exception as e: st.sidebar.error(str(e))

df=st.session_state.get("df")
if df is None:
    st.info("Choose a data source and load data from the sidebar.")
    st.stop()

st.sidebar.header("Strategy")
strategy=st.sidebar.selectbox("Strategy",["SMA crossover","RSI reversal"]); fast=st.sidebar.number_input("Fast SMA",2,500,20) if strategy=="SMA crossover" else 20; slow=st.sidebar.number_input("Slow SMA",3,1000,50) if strategy=="SMA crossover" else 50; rsi_period=st.sidebar.number_input("RSI period",2,100,14)
st.sidebar.header("Simulation")
capital=st.sidebar.number_input("Starting balance",10.0,1000000.0,1000.0,step=100.0); size=st.sidebar.slider("Position size (%)",1,100,100); fee=st.sidebar.number_input("Fee per side (%)",0.0,5.0,0.1,step=0.01); slip=st.sidebar.number_input("Slippage (%)",0.0,5.0,0.05,step=0.01); stop=st.sidebar.number_input("Stop loss (%)",0.0,100.0,0.0,step=0.5); take=st.sidebar.number_input("Take profit (%)",0.0,500.0,0.0,step=0.5)
d=indicators(df,strategy,int(fast),int(slow),int(rsi_period)); eq,tr,final=backtest(d,capital,fee,slip,size,stop,take); dd=(eq.equity/eq.equity.cummax()-1)*100
c=st.columns(5); c[0].metric("Final balance",f"${final:,.2f}"); c[1].metric("Return",f"{(final/capital-1)*100:.2f}%"); c[2].metric("Trades",len(tr)); c[3].metric("Max drawdown",f"{dd.min():.2f}%"); c[4].metric("Daily records",f"{len(d):,}")
t1,t2,t3=st.tabs(["Equity","Price","Trades"])
with t1:
    st.plotly_chart(go.Figure(go.Scatter(x=eq.timestamp,y=eq.equity,name="Equity")).update_layout(height=430),use_container_width=True); st.download_button("Download equity CSV",eq.to_csv(index=False),"equity.csv","text/csv")
with t2:
    fig=go.Figure(go.Candlestick(x=d.timestamp,open=d.open,high=d.high,low=d.low,close=d.close,name="Price"));
    if strategy=="SMA crossover": fig.add_trace(go.Scatter(x=d.timestamp,y=d.fast_sma,name="Fast SMA")); fig.add_trace(go.Scatter(x=d.timestamp,y=d.slow_sma,name="Slow SMA"))
    st.plotly_chart(fig.update_layout(height=600,xaxis_rangeslider_visible=False),use_container_width=True)
with t3:
    st.download_button("Download trades CSV",tr.to_csv(index=False),"trades.csv","text/csv"); st.dataframe(tr,use_container_width=True)
