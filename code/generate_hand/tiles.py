from __future__ import annotations
import numpy as np

N_TILE_TYPES = 34
COPIES_PER_TILE = 4
N_TILES = 136
TILE_NAMES = [*(f"{r}m" for r in range(1,10)), *(f"{r}p" for r in range(1,10)), *(f"{r}s" for r in range(1,10)), "East","South","West","North","White","Green","Red"]
YAOCHU_INDICES = np.array([0,8,9,17,18,26,27,28,29,30,31,32,33], dtype=np.int64)
# yaochu_tiles: 1,9m, 1,9p, 1,9s, 東, 南, 西, 北, 白, 發, 中
YAOCHU_MASK = np.zeros(34, dtype=np.bool_)
YAOCHU_MASK[YAOCHU_INDICES] = True
SEAT_NAMES_ZH = {0:"東家",1:"南家",2:"西家",3:"北家"}

def fresh_wall_counts():
    return np.full(34, 4, dtype=np.int16)
