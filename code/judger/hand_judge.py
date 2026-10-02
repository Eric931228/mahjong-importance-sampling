from __future__ import annotations
import numpy as np
from numba import njit
from generate_hand.tiles import YAOCHU_INDICES

# 將天和事件分為普通型、七對子、國士無雙
EVENT_NORMAL=0
EVENT_SEVEN_PAIRS=1
EVENT_KOKUSHI=2
EVENT_TOTAL=3
EVENT_LABELS_ZH={EVENT_NORMAL:"普通型",EVENT_SEVEN_PAIRS:"七對子",EVENT_KOKUSHI:"國士無雙",EVENT_TOTAL:"總事件"}

@njit(cache=True)
def is_seven_pairs(counts):
    """
    判別完成手牌是否為七對子
    """
    if counts.sum()!=14: return False
    n=0
    for i in range(34):
        if counts[i]==2: n+=1
        elif counts[i]!=0: return False
    return n==7

@njit(cache=True)
def is_kokushi(counts):
    """
    判別完成手牌是否為國士無雙
    """
    if counts.sum()!=14: return False
    dup=0
    for j in range(13):
        c=counts[YAOCHU_INDICES[j]]
        if c==0: return False
        if c>=2: dup+=1
    if dup!=1: return False
    for i in range(34):
        found=False
        for j in range(13):
            if YAOCHU_INDICES[j]==i:
                found=True; break
        if not found and counts[i]!=0: return False
    return True

@njit(cache=True)
def _can_form_melds(counts):
    """
    判斷除了雀頭(眼睛)外，其餘12張牌是否能夠組成4組面子
    """
    
    # 先從一萬開始找出第一個還有牌的牌種，如果能夠組成面子的話就將該面子暫時扣除
    first=-1
    for i in range(34):
        if counts[i]>0:
            first=i; break
    if first==-1: return True

    # 如果有三張相同的牌，嘗試組成刻子，並將三張相同的牌暫時扣除
    if counts[first]>=3:
        counts[first]-=3
        # 遞迴
        if _can_form_melds(counts):
            counts[first]+=3
            return True
        counts[first]+=3

    # 如果非字牌，嘗試組成順子，先判斷數字為1到7，並且數字+1和數字+2皆有牌，將這三張牌暫時扣除
    if first<27:
        r=first%9
        if r<=6 and counts[first+1]>0 and counts[first+2]>0:
            counts[first]-=1; counts[first+1]-=1; counts[first+2]-=1
            # 遞迴
            if _can_form_melds(counts):
                counts[first]+=1; counts[first+1]+=1; counts[first+2]+=1
                return True
            counts[first]+=1; counts[first+1]+=1; counts[first+2]+=1
    return False

@njit(cache=True)
def is_standard_hand(counts):
    """
    判斷完成手牌是否為普通型
    將任意兩張相同的牌暫時扣除後，呼叫_can_form_melds，
    如果為真，則為普通型，如果為假，則選擇下一種有兩張相同的牌，繼續呼叫_can_form_melds
    """
    if counts.sum()!=14: return False
    work=counts.copy()
    for p in range(34):
        if work[p]>=2:
            work[p]-=2
            ok=_can_form_melds(work)
            work[p]+=2
            if ok: return True
    return False

@njit(cache=True)
def is_normal_hand(counts):
    """
    由於將七對子跟普通型當作互斥事件，所以先判斷是否不為七對子，再判斷是否為普通型
    """
    if is_seven_pairs(counts) or is_kokushi(counts): return False
    return is_standard_hand(counts)

@njit(cache=True)
def is_any_complete_hand(counts):
    """
    判斷手牌是否能宣告和牌
    """
    return is_normal_hand(counts) or is_seven_pairs(counts) or is_kokushi(counts)

@njit(cache=True)
def matches_event(counts,event_code):
    """
    判斷完成手牌牌型
    """
    if event_code==EVENT_NORMAL: return is_normal_hand(counts)
    if event_code==EVENT_SEVEN_PAIRS: return is_seven_pairs(counts)
    if event_code==EVENT_KOKUSHI: return is_kokushi(counts)
    return is_any_complete_hand(counts)
