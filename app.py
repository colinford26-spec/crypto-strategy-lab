import json, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ccxt, numpy as np, pandas as pd, plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Crypto Strategy Lab", page_icon="📈", layout="wide")
DATA=Path("data"); DATA.mkdir(exist_ok=True)

@st.cache_data(ttl=3600, show_spinner=False)
def markets(exchange_id):
    ex=getattr(ccxt, exchange_id)({"enableRateLimit":True})
    ex.load_markets()
    return sorted([s for s,v in ex.markets.items() if v.get("spot") and s.endswith("/USDT")])

def file_for(exchange,symbol,tf):
    return DATA/(exchange+"_"+symbol.replace("/","-")+"_"+tf+".csv")

def download(exchange_id,symbol,tf,start,end):
    ex=getattr(ccxt,exchange_id)({"enableRateLimit":True})
    ex.load_markets()
    since=int(pd.Timestamp(start,tz="UTC").timestamp()*1000); end_ms=int(pd.Timestamp(end,tz="UTC").timestamp()*1000)
    rows=[]
    while since < end_ms:
        batch=ex.fetch_ohlcv(symbol,tf,since=since,limit=1000)
        if not batch: break
        rows += [x for x in batch if x[0] < end_ms]
        nxt=batch[-1][0]+1
        if nxt<=since: break
        since=nxt; time.sleep(max(ex.rateLimit,50)/1000)
        if len(batch)<2: break
    df=pd.DataFrame(rows,columns=["timestamp","open","high","low","close","volume"])
    if df.empty: raise ValueError("No candles returned by this exchange.")
    df=df.drop_duplicates("timestamp").sort_values("timestamp")
    df["timestamp"]=pd.to_datetime(df.timestamp,unit="ms",utc=True)
    df.to_csv(file_for(exchange_id,symbol,tf),index=False)
    return df

def indicators(df,kind,fast,slow,rsi_period):
    d=df.copy(); d["fast"] = d.close.rolling(fast).mean(); d["slow"] = d.close.rolling(slow).mean()
    delta=d.close.diff(); gain=delta.clip(lower=0).rolling(rsi_period).mean(); loss=(-delta.clip(upper=0)).rolling(rsi_period).mean()
    d["rsi"]=100-(100/(1+gain/loss.replace(0,np.nan)))
    if kind=="SMA crossover": d["signal"]=np.where(d.fast>d.slow,1,0)
    elif kind=="RSI reversal": d["signal"]=np.where(d.rsi<30,1,np.where(d.rsi>70,-1,0))
    return d

def backtest(d,capital,fee,slip,size,stop,take):
    cash=capital; qty=0.; entry=0.; trades=[]; equity=[]
    for _,r in d.iterrows():
        p=float(r.close); action=0
        if qty and ((stop and p<=entry*(1-stop/100)) or (take and p>=entry*(1+take/100))): action=-1
        elif r.signal==1 and not qty: action=1
        elif r.signal==-1 and qty: action=-1
        if action==1:
            spend=cash*size/100; execution=p*(1+slip/100); qty=spend/execution; cash-=spend*(1+fee/100); entry=execution
        elif action==-1 and qty:
            execution=p*(1-slip/100); proceeds=qty*execution*(1-fee/100); pnl=proceeds-qty*entry; cash+=proceeds
            trades.append([r.timestamp,"SELL",execution,qty,pnl]); qty=0; entry=0
        equity.append([r.timestamp,cash+qty*p])
    if qty:
        p=float(d.iloc[-1].close); proceeds=qty*p*(1-fee/100); trades.append([d.iloc[-1].timestamp,"FINAL EXIT",p,qty,proceeds-qty*entry]); cash+=proceeds; qty=0
    eq=pd.DataFrame(equity,columns=["timestamp","equity"]); tr=pd.DataFrame(trades,columns=["timestamp","side","price","quantity","pnl"])
    return eq,tr,cash

st.title("📈 Crypto Strategy Lab")
st.caption("Download public exchange candles, configure a simple strategy, and inspect a simulated result.")
with st.sidebar:
    st.header("1. Market data")
    exchange=st.selectbox("Exchange",["kraken","binance","coinbase","bybit"])
    try: symbols=markets(exchange)
    except Exception as e: symbols=["BTC/USDT","ETH/USDT"]; st.warning("Could not load markets; showing defaults.")
    symbol=st.selectbox("Market",symbols,index=0 if "BTC/USDT" not in symbols else symbols.index("BTC/USDT"))
    tf=st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=3)
    default_start=datetime.now(timezone.utc)-timedelta(days=180)
    start=st.date_input("Start date",default_start.date()); end=st.date_input("End date",datetime.now(timezone.utc).date())
    if st.button("Download / refresh data",type="primary"):
        try: st.session_state.df=download(exchange,symbol,tf,start,end); st.success(f"Loaded {len(st.session_state.df):,} candles")
        except Exception as e: st.error(str(e))
    st.header("2. Strategy")
    strategy=st.selectbox("Strategy",["SMA crossover","RSI reversal"])
    fast=st.number_input("Fast SMA",2,500,20) if strategy=="SMA crossover" else 20
    slow=st.number_input("Slow SMA",3,1000,50) if strategy=="SMA crossover" else 50
    rsi_period=st.number_input("RSI period",2,100,14)
    st.header("3. Simulation")
    capital=st.number_input("Starting balance",10.0,1000000.0,1000.0,step=100.0)
    size=st.slider("Position size (%)",1,100,100)
    fee=st.number_input("Fee (%) per side",0.0,5.0,0.1,step=0.01)
    slip=st.number_input("Slippage (%)",0.0,5.0,0.05,step=0.01)
    stop=st.number_input("Stop loss (%)",0.0,100.0,0.0,step=0.5)
    take=st.number_input("Take profit (%)",0.0,500.0,0.0,step=0.5)

df=st.session_state.get("df")
if df is None:
    p=file_for(exchange,symbol,tf)
    if p.exists(): df=pd.read_csv(p,parse_dates=["timestamp"])
if df is None:
    st.info("Choose the market settings in the sidebar and click Download / refresh data.")
    st.stop()
if "timestamp" not in df: st.error("Invalid data file."); st.stop()
d=indicators(df,strategy,int(fast),int(slow),int(rsi_period)); eq,tr,balance=backtest(d,capital,fee,slip,size,stop,take)
first=float(eq.equity.iloc[0]); final=float(eq.equity.iloc[-1]); ret=(final/capital-1)*100; peak=eq.equity.cummax(); dd=(eq.equity/peak-1)*100; maxdd=float(dd.min())
cols=st.columns(5); cols[0].metric("Final balance",f"${final:,.2f}"); cols[1].metric("Return",f"{ret:.2f}%"); cols[2].metric("Trades",len(tr)); cols[3].metric("Max drawdown",f"{maxdd:.2f}%"); cols[4].metric("Candles",f"{len(d):,}")
tab1,tab2,tab3=st.tabs(["Equity curve","Price & signals","Trades / data"])
with tab1:
    fig=go.Figure(); fig.add_trace(go.Scatter(x=eq.timestamp,y=eq.equity,name="Equity")); fig.update_layout(height=430,margin=dict(l=10,r=10,t=30,b=10)); st.plotly_chart(fig,use_container_width=True)
    ddfig=go.Figure(go.Scatter(x=eq.timestamp,y=dd,fill="tozeroy",name="Drawdown")); ddfig.update_layout(height=250,margin=dict(l=10,r=10,t=30,b=10),yaxis_title="% from peak"); st.plotly_chart(ddfig,use_container_width=True)
with tab2:
    fig=go.Figure(go.Candlestick(x=d.timestamp,open=d.open,high=d.high,low=d.low,close=d.close,name=symbol));
    if strategy=="SMA crossover": fig.add_trace(go.Scatter(x=d.timestamp,y=d.fast,name="Fast SMA")); fig.add_trace(go.Scatter(x=d.timestamp,y=d.slow,name="Slow SMA"))
    fig.update_layout(height=600,xaxis_rangeslider_visible=False); st.plotly_chart(fig,use_container_width=True)
with tab3:
    st.download_button("Download trades CSV",tr.to_csv(index=False),"trades.csv","text/csv"); st.download_button("Download candles CSV",d.to_csv(index=False),"candles.csv","text/csv"); st.dataframe(tr,use_container_width=True)
