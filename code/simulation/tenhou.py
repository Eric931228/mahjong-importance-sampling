from __future__ import annotations
import numpy as np
from numba import njit
from generate_hand.proposal import draw_proposal
from judger.hand_judge import matches_event

@njit(cache=True)
def simulate_tenhou_batch(n,event_code,weights,seed):
    """
    給定模擬次數、目標偵測事件、權重組合、種子碼，執行天和模擬
    """
    np.random.seed(seed)
    total=0.0; hits=0
    for _ in range(n):
        wall=np.full(34,4,dtype=np.int16)
        hand=np.zeros(34,dtype=np.int16)
        lw=0.0
        for _i in range(14):
            _,lr=draw_proposal(wall,hand,weights); lw+=lr
        if matches_event(hand,event_code):
            total+=np.exp(lw); hits+=1
    return total/n,hits
