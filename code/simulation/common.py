from __future__ import annotations
import numpy as np
from numba import njit
from judger.hand_judge import is_any_complete_hand

@njit(cache=True)
def discard_uniform(hand):
    """
    以相等機率隨機捨棄一張手牌
    """
    total=0
    for i in range(34): total+=hand[i]

    # 將手牌編號從1到總手牌數，透過 Unif(0,1) 乘以總手牌數決定棄牌的位置，將該牌的數量減1
    u=np.random.random()*total; c=0.0; last=0
    for i in range(34):
        if hand[i]>0:
            last=i; c+=hand[i]
            if u<c:
                hand[i]-=1; return i
    hand[last]-=1; return last

@njit(cache=True)
def discard_causes_ron(hands,discarder,tile):
    """
    判斷某一玩家的捨牌是否為其他玩家的銃牌
    """
    # 從棄牌者除外開始判斷，如果其他玩家的手牌加上捨牌可以宣告和牌的話視為榮和
    for seat in range(4):
        if seat==discarder: continue
        if hands[seat].sum()!=13: continue
        hands[seat,tile]+=1
        won=is_any_complete_hand(hands[seat])
        hands[seat,tile]-=1
        if won: return True
    return False

@njit(cache=True)
def is_yakuhai_for_seat(tile,seat):
    """
    判斷字牌是否為役牌（場風：東、自風、三元：白,發,中）
    """
    if tile>=31: return True
    if tile==27: return True
    return tile==27+seat

@njit(cache=True)
def can_chi(hand, tile):
    """
    判斷是否可以宣告吃牌
    """
    # 字牌不能吃
    if tile >= 27:
        return False

    rank = tile % 9
    if rank >= 2:
        if hand[tile-2] > 0 and hand[tile-1] > 0:
            return True
    if 1 <= rank <= 7:
        if hand[tile-1] > 0 and hand[tile+1] > 0:
            return True
    if rank <= 6:
        if hand[tile+1] > 0 and hand[tile+2] > 0:
            return True

    return False

@njit(cache=True)
def call_interrupts(hands, discarder, target_seat, tile, p_chi):
    """
    1. 若非目標玩家可以用棄牌碰來形成役牌，則必定鳴牌
    2. 否則檢查棄牌者下家是否可以吃牌
    3. 若可以吃牌，依 p_chi 決定是否實際宣告吃牌
    """

    # 役牌碰 / 大明槓：必定鳴牌
    for seat in range(4):
        if seat == discarder or seat == target_seat:
            continue

        if (
            hands[seat, tile] >= 2
            and is_yakuhai_for_seat(tile, seat)
        ):
            return True

    # 吃牌：只有棄牌者下家可以吃
    chi_seat = (discarder + 1) % 4

    # target player 維持地和保護
    if chi_seat == target_seat:
        return False

    if can_chi(hands[chi_seat], tile):
        return np.random.random() < p_chi

    return False
