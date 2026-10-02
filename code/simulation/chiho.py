from __future__ import annotations
import numpy as np
from numba import njit
from generate_hand.proposal import draw_fair,draw_proposal
from judger.hand_judge import is_any_complete_hand,matches_event
from simulation.common import call_interrupts,discard_causes_ron,discard_uniform

D_ESTIMATE=0; D_OPPORTUNITY=1; D_EAST_TENHOU=2; D_PRETARGET_TSUMO=3; D_RON=4; D_CALL=5

@njit(cache=True)
def _deal_initial_hands(target_seat,weights):
    """
    地和模擬：首先東家先依照正常機率抽取14張牌，
    接下來除目標座位外，其他玩家也是依照正常機率抽取13張牌，
    目標座位依照 proposal probability 抽13張牌
    """
    wall=np.full(34,4,dtype=np.int16)
    hands=np.zeros((4,34),dtype=np.int16)
    lw=0.0
    for _ in range(14):
        t=draw_fair(wall); hands[0,t]+=1
    for seat in range(1,4):
        for _ in range(13):
            if seat==target_seat:
                _,lr=draw_proposal(wall,hands[seat],weights); lw+=lr
            else:
                t=draw_fair(wall); hands[seat,t]+=1
    return wall,hands,lw

@njit(cache=True)
def simulate_chiho_batch(n,target_seat,event_code,weights,p_chi,seed):
    """
    給定模擬次數、目標座位、目標偵測事件、權重組合、種子碼，執行地和模擬
    """
    np.random.seed(seed)
    weighted=np.zeros(6,dtype=np.float64)
    raw=np.zeros(6,dtype=np.int64)
    for _ in range(n):
        wall,hands,lw0=_deal_initial_hands(target_seat,weights)
        w0=np.exp(lw0)

        # 判斷東家是否滿足天和條件
        if is_any_complete_hand(hands[0]):
            weighted[D_EAST_TENHOU]+=w0; raw[D_EAST_TENHOU]+=1; continue
        interrupted=False

        # 模擬目標座位以前的玩家
        # 1. 是否能夠宣告和牌
        # 2. 隨機棄1張牌
        # 3. 是否要宣告副露
        for actor in range(0,target_seat):
            if actor!=0:
                t=draw_fair(wall); hands[actor,t]+=1
                if is_any_complete_hand(hands[actor]):
                    weighted[D_PRETARGET_TSUMO]+=w0; raw[D_PRETARGET_TSUMO]+=1; interrupted=True; break
            disc=discard_uniform(hands[actor])
            if discard_causes_ron(hands,actor,disc):
                weighted[D_RON]+=w0; raw[D_RON]+=1; interrupted=True; break
            if call_interrupts(hands,actor,target_seat,disc,p_chi):
                weighted[D_CALL]+=w0; raw[D_CALL]+=1; interrupted=True; break

        if interrupted: continue
        weighted[D_OPPORTUNITY]+=w0; raw[D_OPPORTUNITY]+=1
        _,lr=draw_proposal(wall,hands[target_seat],weights)
        wf=np.exp(lw0+lr)
        if matches_event(hands[target_seat],event_code):
            weighted[D_ESTIMATE]+=wf; raw[D_ESTIMATE]+=1
    weighted/=n
    return weighted,raw
