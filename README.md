# Riichi Mahjong Rare Event Importance Sampling

以 **Mixture Importance Sampling** 估計日本麻將稀有和牌事件，先以已知的天和理論機率進行校準，再延伸至南家、西家、北家的第一巡地和模擬。

## 功能

- 天和：普通型、七對子、國士無雙與總事件。
- 地和：南家、西家、北家分開估計，納入第一巡座位順序、榮和、自摸與鳴牌中斷。
- Proposal：`base`、`pair`、`d1`、`d2`、`yaochu`、`missing`、`kpair` 七個 mixture components。
- Streamlit：調整事件、batch size、批次數、座位與 proposal weights。

## 專案結構

```text
mahjong-importance-sampling/
├── app.py
├── main.ipynb
├── README.md
├── requirements.txt
├── code/
│   ├── generate_hand/
│   ├── judger/
│   ├── simulation/
│   └── table/
└── report/
    └── figures/
```

## 本機執行

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

啟動後開啟 Streamlit 顯示的 Local URL，通常是：

```text
http://localhost:8501
```

## 報告

完整研究動機、方法、公式、天和 benchmark calibration、地和模擬與最新結果請見：

- [`slide/slide.pdf`](slide/slide.pdf)

README 僅保留專案入口與使用方式；完整技術說明以 `slide/` 內的報告為準。
