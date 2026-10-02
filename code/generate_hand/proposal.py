from __future__ import annotations
import numpy as np
from numba import njit
from generate_hand.tiles import YAOCHU_MASK

WEIGHT_NAMES=("w_base","w_pair","w_d1","w_d2","w_yaochu","w_missing","w_kpair")

# 預設權重組合
PRESET_NORMAL=np.array([0.50,0.15,0.15,0.15,0.05,0,0],dtype=np.float64)
PRESET_SEVEN_PAIRS=np.array([0.50,0.35,0.10,0.03,0.02,0,0],dtype=np.float64)
PRESET_KOKUSHI=np.array([0.15,0,0,0,0.10,0.65,0.10],dtype=np.float64)

def normalize_weights(weights):
    """
    確認權重是否合法並正規化
    """
    a=np.asarray(weights,dtype=np.float64)
    if a.shape!=(7,): raise ValueError("mixture proposal 必須恰好有 7 個權重。")
    if np.any(~np.isfinite(a)) or np.any(a<0): raise ValueError("所有權重必須是有限非負數。")
    s=float(a.sum())
    if s<=0: raise ValueError("權重總和必須大於 0。")
    return a/s

@njit(cache=True)
def _add_or_base(q,counts,base,eligible,weight):
    """
    將原本的機率分配加上 proposal probability，
    如果已經沒有特定種類的手牌則退回原本的機率分布
    """
    if weight==0.0: return
    d=0

    # 計算 proposal 的手牌張數總和
    for i in range(34):     
        if eligible[i] and counts[i]>0: d+=counts[i]

    # 如果 proposal 的剩餘張數為0，退回原本的機率分布
    if d<=0:
        for i in range(34): q[i]+=weight*base[i]

    # 否則對所有 proposal 增加機率
    else:
        for i in range(34):
            if eligible[i] and counts[i]>0: q[i]+=weight*counts[i]/d

@njit(cache=True)
def proposal_probabilities(counts,hand,weights):
    """
    計算 mixture importance sampling 的 proposal probability
    """
    # 計算目前牌牆剩餘張數
    total=0.0  
    for i in range(34): total+=counts[i]
    base=np.zeros(34,dtype=np.float64)

    # 計算 base probability
    for i in range(34):
        if counts[i]>0: base[i]=counts[i]/total

    # 將 base component 放進 proposal
    q=weights[0]*base

    # 建立候選 mixture component
    e=np.zeros(34,dtype=np.bool_)

    # 1. 成對
    for i in range(34): e[i]=hand[i]>0
    _add_or_base(q,counts,base,e,weights[1])
    for i in range(34): e[i]=False

    # 2. 距離1
    for i in range(27):
        if hand[i]>0:
            s=(i//9)*9; r=i%9
            if r>0: e[s+r-1]=True
            if r<8: e[s+r+1]=True
    _add_or_base(q,counts,base,e,weights[2])
    for i in range(34): e[i]=False

    # 3. 距離2
    for i in range(27):
        if hand[i]>0:
            s=(i//9)*9; r=i%9
            if r>=2: e[s+r-2]=True
            if r<=6: e[s+r+2]=True
    _add_or_base(q,counts,base,e,weights[3])
    for i in range(34): e[i]=False

    # 4. 么九牌
    for i in range(34): e[i]=YAOCHU_MASK[i]
    _add_or_base(q,counts,base,e,weights[4])
    for i in range(34): e[i]=False

    # 5. 未出現么九牌（國士無雙用）
    for i in range(34): e[i]=YAOCHU_MASK[i] and hand[i]==0
    _add_or_base(q,counts,base,e,weights[5])
    for i in range(34): e[i]=False

    # 6. 成對么九牌（國士無雙用）
    for i in range(34): e[i]=YAOCHU_MASK[i] and hand[i]>0
    _add_or_base(q,counts,base,e,weights[6])

    # 將機率正規化使其總合為1
    s=0.0
    for i in range(34): s+=q[i]
    for i in range(34): q[i]/=s
    return q

@njit(cache=True)
def draw_from_distribution(prob):
    """
    從 proposal probability 中選取一張牌。
    透過累加機率直到超過一開始從 Uniform(0,1) 中抽到的數字，視為選到該牌。
    """
    u=np.random.random(); c=0.0; last=0
    for i in range(34):
        if prob[i]>0: last=i
        c+=prob[i]
        if u<=c: return i
    return last

@njit(cache=True)
def draw_fair(counts):
    """
    實際抽取選擇的牌，並且將牌牆中的該種類牌的張數減一
    """
    # 計算牌牆張數
    total=0
    for i in range(34): total+=counts[i]
    u=np.random.random()*total; c=0.0; last=0

    # 抽取選擇的牌並將張數減一
    for i in range(34):
        if counts[i]>0:
            last=i; c+=counts[i]
            if u<c:
                counts[i]-=1; return i
    counts[last]-=1
    return last

@njit(cache=True)
def draw_proposal(counts,hand,weights):
    """
    從 proposal probability 中抽取
    """
    total=0.0
    for i in range(34): total+=counts[i]
    q=proposal_probabilities(counts,hand,weights)
    tile=draw_from_distribution(q)
    log_ratio=np.log(counts[tile]/total)-np.log(q[tile])
    counts[tile]-=1; hand[tile]+=1
    return tile,log_ratio
