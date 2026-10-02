from __future__ import annotations
import math
import numpy as np
import pandas as pd

M_STEP=5

def ceil_to_step(value:int, step:int=M_STEP):
    """
    若總 batch 模擬數不為5的倍數，則向上取最接近的5的倍數的數
    """
    return max(step, int(math.ceil(value/step)*step))

def cumulative_result_table(batch_estimates, benchmark=None, step:int=M_STEP):
    """
    輸出表格結果
    """
    estimates=np.asarray(batch_estimates,dtype=np.float64)
    rows=[]
    # 以每5個 batch 顯示模擬結果
    for m in range(step, len(estimates)+1, step):
        cur=estimates[:m]
        est=float(cur.mean())
        mc_se=float(cur.std(ddof=1)/np.sqrt(m)) if m>=2 else np.nan
        row={"M":m,"Estimate":est,"MC s.e.":mc_se}
        if benchmark is not None:
            signed_bias=est-benchmark
            are=abs(signed_bias)/benchmark
            row["Bias"]=signed_bias
            row["Absolute Relative Error"]=are
        rows.append(row)
    return pd.DataFrame(rows)

def format_result_table(df, has_benchmark:bool):
    """
    調整輸出格式
    """
    out=df.copy()
    out["Estimate"]=out["Estimate"].map(lambda x:f"{x:.6e}")
    out["MC s.e."]=out["MC s.e."].map(lambda x:"NA" if pd.isna(x) else f"{x:.6e}")
    if has_benchmark:
        out["Bias"]=out["Bias"].map(lambda x:f"{x:.6e}")
        out["Absolute Relative Error"]=out["Absolute Relative Error"].map(lambda x:f"{100*x:.4f}%")
    return out
