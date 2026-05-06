# 基於臺中市交通事故資料之事故嚴重程度預測與決策支援分析（Dashboard Demo）

本 Demo 以 **臺中市** 為主，提供可即時展示的風險地圖與決策建議頁面。即使目前還沒有組員整理好的圖表與最終資料，也可先用內建假資料完成展示。

## 1) 怎麼跑（本機）

```bash
cd /workspace/MDS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard_app.py
```

啟動後瀏覽器會自動開啟，若未開啟可手動進入終端顯示網址（通常是 `http://localhost:8501`）。

## 2) 目前頁面與功能

### A. 決策總覽
- KPI：事故總數 / 高風險事件 / 平均風險 / 死傷示意
- **臺中市地圖（主頁）且可開關圖層**：
  - 歷史事故點
  - 模型高風險預測點
  - 建議巡邏區域
- 決策建議清單（Top 15）
- 高風險時段分析
- 臺中市各行政區風險比較

### B. 模型效能頁
- 模型指標表（Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC）
- Recall 比較圖
- PR-AUC 比較圖
- Confusion Matrix（示意）

### C. 關鍵因子頁
- 特徵重要性圖（Demo，可替換為 SHAP）
- 解讀文字區（可改為 SHAP summary / dependence / local explanation）

## 3) 上傳真實資料（CSV）

至少需要欄位：

- `pred_prob`（模型預測機率）
- `hour`（事故時）
- `lat`（緯度）
- `lon`（經度）

> 若你們目前尚未產出 `prob_pred / pred_prob`，可先提供 `Y_事故嚴重度`（0/1）；系統會暫時轉為示意風險機率（0→0.2、1→0.8）以便展示流程。

### 目前也支援你們這組欄位命名

- `GPS座標X` 自動對應為 `lon`
- `GPS座標Y` 自動對應為 `lat`
- 若沒有 `district`，會嘗試從 one-hot 欄位（例如 `區_北屯區`、`區_西屯區`）還原行政區

建議欄位（有會更完整）：

- `month`, `district`, `accident_type`, `deaths`, `injuries`, `city`

> 若 CSV 含 `city` 欄位，系統會自動篩出「台中/臺中」資料，符合題目範圍。

## 4) 題目對應

- 題目：**基於臺中市交通事故資料之事故嚴重程度預測與決策支援分析**
- 資料期間：**114 年 10 月 ~ 12 月**（Demo 內建資料也對應此期間）
- 決策規則：
  - 高風險（P ≥ 0.7）：優先派遣警力 / 加強巡邏
  - 中風險（0.4 ≤ P < 0.7）：定期巡查
  - 低風險（P < 0.4）：維持現狀
