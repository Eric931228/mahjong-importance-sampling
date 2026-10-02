from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
CODE=ROOT/"code"
if str(CODE) not in sys.path: sys.path.insert(0,str(CODE))

from generate_hand.proposal import PRESET_KOKUSHI,PRESET_NORMAL,PRESET_SEVEN_PAIRS,WEIGHT_NAMES,normalize_weights
from judger.hand_judge import EVENT_KOKUSHI,EVENT_NORMAL,EVENT_SEVEN_PAIRS,EVENT_TOTAL
from simulation.benchmarks import TENHOU_BENCHMARKS
from simulation.tenhou import simulate_tenhou_batch
from simulation.chiho import simulate_chiho_batch
from table.results import cumulative_result_table,format_result_table,ceil_to_step



st.set_page_config(page_title="Riichi Mahjong Rare-Win IS",layout="wide")
EVENT_OPTIONS={"普通型":EVENT_NORMAL,"七對子":EVENT_SEVEN_PAIRS,"國士無雙":EVENT_KOKUSHI,"總事件":EVENT_TOTAL}
SEAT_OPTIONS={"南家":1,"西家":2,"北家":3}
PRESETS={"普通型":PRESET_NORMAL,"七對子":PRESET_SEVEN_PAIRS,"國士無雙":PRESET_KOKUSHI,"總事件":PRESET_NORMAL}

def set_weights(values):
    for n, v in zip(WEIGHT_NAMES, values):
        st.session_state[n] = float(v)

for n, v in zip(WEIGHT_NAMES, PRESET_NORMAL):
    if n not in st.session_state:
        st.session_state[n] = float(v)

st.title("日本麻將稀有和牌事件：Mixture Importance Sampling")
st.caption("Tenhou benchmark calibration + Chiihou first-round extension")

left,right=st.columns([1.05,1.0],gap="large")
with left:
    st.subheader("模擬設定")
    simulation_type=st.radio("模擬",["東家天和","地和"],horizontal=True)
    event_label=st.radio("事件",list(EVENT_OPTIONS),horizontal=True)
    event_code=EVENT_OPTIONS[event_label]
    if st.session_state.get("_last_event")!=event_label:
        set_weights(PRESETS[event_label]); st.session_state["_last_event"]=event_label
    if simulation_type=="地和":
        selected_seats=st.multiselect("目標座位（可複選）",list(SEAT_OPTIONS),default=["南家"])
        p_chi=st.select_slider("合法吃牌機會下的吃牌條件機率",options=[0.2,0.5,0.8],value=0.5)
    else:
        selected_seats=[]; p_chi=0.5
    c1,c2=st.columns(2)
    with c1:
        batch_size=int(st.number_input("batch size",min_value=1_000,max_value=5_000_000,value=100_000,step=10_000))
    with c2:
        requested_batches=int(st.number_input("模擬批次數 M",min_value=1,max_value=500,value=10,step=5,help="實際 M 自動無條件進位至 5 的倍數。"))
    n_batches=ceil_to_step(requested_batches,5)
    with st.expander("進階設定"):
        seed=int(st.number_input("Random seed",min_value=0,max_value=2_147_483_000,value=42,step=1))

with right:
    st.subheader("Mixture proposal 權重")
    b1,b2,b3=st.columns(3)
    if b1.button("普通型建議權重",use_container_width=True): set_weights(PRESET_NORMAL); st.rerun()
    if b2.button("七對子建議權重",use_container_width=True): set_weights(PRESET_SEVEN_PAIRS); st.rerun()
    if b3.button("國士建議權重",use_container_width=True): set_weights(PRESET_KOKUSHI); st.rerun()
    labels={"w_base":r"$w_{base}$","w_pair":r"$w_{pair}$","w_d1":r"$w_{d1}$","w_d2":r"$w_{d2}$","w_yaochu":r"$w_{yaochu}$","w_missing":r"$w_{missing}$","w_kpair":r"$w_{kpair}$"}
    cols=st.columns(4); raw=[]
    for i,n in enumerate(WEIGHT_NAMES):
        with cols[i%4]: raw.append(st.number_input(labels[n],min_value=0.0,max_value=100.0,step=0.01,format="%.4f",key=n))
    try:
        weights=normalize_weights(raw)
        st.caption("正規化權重：("+", ".join(f"{x:.4f}" for x in weights)+")")
        if weights[0]<=0: st.warning("w_base = 0 可能破壞 support condition。")
    except ValueError as e:
        weights=None; st.error(str(e))


st.divider()
run_disabled=weights is None or (simulation_type=="地和" and not selected_seats)
if simulation_type=="地和" and not selected_seats: st.info("地和至少選一個座位。")
run=st.button("開始模擬",type="primary",disabled=run_disabled,use_container_width=True)

if run:
    weights=weights.astype(np.float64)
    total_jobs=n_batches if simulation_type=="東家天和" else n_batches*len(selected_seats)
    progress=st.progress(0.0,text="準備 Numba 核心與模擬批次…")
    done=0
    if simulation_type=="東家天和":
        batch_est=[]; batch_hits=[]
        for m in range(n_batches):
            est,hits=simulate_tenhou_batch(batch_size,event_code,weights,seed+1009*(m+1))
            batch_est.append(est); batch_hits.append(hits); done+=1
            progress.progress(done/total_jobs,text=f"完成 batch {m+1}/{n_batches}")
        benchmark=TENHOU_BENCHMARKS[event_code]
        result=cumulative_result_table(batch_est,benchmark=benchmark,step=5)
        display=format_result_table(result,has_benchmark=True)
        st.subheader(f"東家天和 — {event_label}")
        st.dataframe(display,use_container_width=True,hide_index=True)
        st.caption("Bias = Estimate − Benchmark；Absolute Relative Error = |Estimate − Benchmark| / Benchmark。")
        with st.expander("模擬診斷"):
            st.write(f"Benchmark: `{benchmark:.9e}`")
            st.write(f"總 proposal 命中數: `{sum(batch_hits):,}`")
            st.write(f"總模擬手牌數: `{batch_size*n_batches:,}`")
    else:
        st.subheader(f"地和 — {event_label}")
        st.caption(f"各座位分開估計。地和沒有封閉解，因此不列 Bias / Absolute Relative Error。")
        for seat_label in selected_seats:
            seat=SEAT_OPTIONS[seat_label]; batch_est=[]; diags=[]; raws=[]
            for m in range(n_batches):
                weighted,rawc=simulate_chiho_batch(batch_size,seat,event_code,weights,float(p_chi),seed+seat*1_000_003+1009*(m+1))
                batch_est.append(float(weighted[0])); diags.append(weighted); raws.append(rawc); done+=1
                progress.progress(done/total_jobs,text=f"{seat_label}: batch {m+1}/{n_batches}")
            result=cumulative_result_table(batch_est,benchmark=None,step=5)
            display=format_result_table(result,has_benchmark=False)
            st.markdown(f"#### {seat_label}")
            st.dataframe(display,use_container_width=True,hide_index=True)
            d=np.vstack(diags).mean(axis=0); r=np.vstack(raws).sum(axis=0); total=batch_size*n_batches
            dd=pd.DataFrame({"診斷量":["地和事件 Estimate","取得第一次自摸機會","東家天和中斷","前置玩家自摸中斷","榮和中斷","鳴牌中斷"],"加權估計":d,"Raw rate":r/total})
            dd["加權估計"]=dd["加權估計"].map(lambda x:f"{x:.6e}")
            dd["Raw rate"]=dd["Raw rate"].map(lambda x:f"{100*x:.4f}%")
            with st.expander(f"{seat_label} 中斷診斷"): st.dataframe(dd,use_container_width=True,hide_index=True)
    progress.progress(1.0,text="模擬完成")

st.divider()
st.caption("所有玩家需要棄牌時，目前採手牌均勻隨機棄一張；此為模擬的假設。")
