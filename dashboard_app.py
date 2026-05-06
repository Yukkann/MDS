import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Taichung Traffic Risk DSS Demo", layout="wide")

TAICHUNG_CENTER = {"lat": 24.1477, "lon": 120.6736}
TAICHUNG_DISTRICTS = [
    "中區", "東區", "南區", "西區", "北區", "北屯區", "西屯區", "南屯區",
    "太平區", "大里區", "霧峰區", "烏日區", "豐原區", "后里區", "石岡區", "東勢區",
    "新社區", "潭子區", "大雅區", "神岡區", "大肚區", "沙鹿區", "龍井區", "梧棲區",
    "清水區", "大甲區", "外埔區", "大安區", "和平區"
]


@st.cache_data
def generate_demo_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weather = ["晴", "陰", "雨"]
    road_category = ["市區道路", "省道", "區道"]
    accident_types = ["車與車", "車與行人", "單一車輛"]

    months = rng.choice([10, 11, 12], size=n, p=[0.33, 0.34, 0.33])
    hours = rng.integers(0, 24, size=n)
    districts = rng.choice(TAICHUNG_DISTRICTS, size=n)
    weather_col = rng.choice(weather, size=n, p=[0.5, 0.3, 0.2])

    base = (
        0.18
        + (hours >= 22) * 0.22
        + (hours <= 5) * 0.18
        + (weather_col == "雨") * 0.14
        + rng.normal(0, 0.08, n)
    )
    pred_prob = np.clip(base, 0.01, 0.99)
    severity_bin = (pred_prob > 0.45).astype(int)

    lats = TAICHUNG_CENTER["lat"] + rng.normal(0, 0.06, size=n)
    lons = TAICHUNG_CENTER["lon"] + rng.normal(0, 0.06, size=n)

    df = pd.DataFrame(
        {
            "year": 2025,
            "month": months,
            "day": rng.integers(1, 29, size=n),
            "hour": hours,
            "city": "臺中市",
            "district": districts,
            "weather": weather_col,
            "road_category": rng.choice(road_category, size=n),
            "accident_type": rng.choice(accident_types, size=n),
            "pred_prob": pred_prob,
            "severity_bin": severity_bin,
            "deaths": (pred_prob > 0.8).astype(int),
            "injuries": np.where(severity_bin == 1, rng.integers(1, 4, size=n), rng.integers(0, 2, size=n)),
            "lat": lats,
            "lon": lons,
        }
    )
    return df


@st.cache_data
def generate_demo_model_scores(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest", "AdaBoost", "XGBoost"],
            "Accuracy": [0.74, 0.81, 0.79, 0.83],
            "Precision": [0.62, 0.73, 0.70, 0.75],
            "Recall": [0.71, 0.82, 0.80, 0.85],
            "F1": [0.66, 0.77, 0.75, 0.79],
            "ROC_AUC": [0.79, 0.87, 0.85, 0.89],
            "PR_AUC": [0.58, 0.72, 0.69, 0.75],
        }
    )


@st.cache_data
def generate_demo_feature_importance(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = ["天候", "路口型態", "照明", "速限", "事故型態", "肇因", "是否酒駕", "時間", "車種", "道路類別"]
    vals = np.sort(rng.uniform(0.02, 0.24, size=len(features)))[::-1]
    return pd.DataFrame({"feature": features, "importance": vals})




def normalize_uploaded_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "GPS座標X": "lon",
        "GPS座標Y": "lat",
        "區": "district",
        "道路速限": "speed_limit",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}).copy()

    if "pred_prob" not in df.columns:
        if "prob_pred" in df.columns:
            df["pred_prob"] = pd.to_numeric(df["prob_pred"], errors="coerce")
        elif "Y_事故嚴重度" in df.columns:
            sev = pd.to_numeric(df["Y_事故嚴重度"], errors="coerce")
            df["pred_prob"] = sev.map({0: 0.2, 1: 0.8})

    if "district" not in df.columns:
        district_cols = [col for col in df.columns if col.startswith("區_")]
        if district_cols:
            dummies = df[district_cols].fillna(0)
            max_idx = dummies.idxmax(axis=1)
            has_any = dummies.max(axis=1) > 0
            df["district"] = np.where(has_any, max_idx.str.replace("區_", "", regex=False), "未提供")

    return df

def strategy_by_risk(risk: str) -> str:
    mapping = {
        "High": "優先派遣警力 / 加強巡邏",
        "Medium": "定期巡查",
        "Low": "維持現狀",
    }
    return mapping.get(risk, "維持現狀")


def build_layered_map(view: pd.DataFrame, show_hist: bool, show_high: bool, show_patrol: bool) -> go.Figure:
    fig = go.Figure()

    if show_hist:
        fig.add_trace(
            go.Scattermap(
                lat=view["lat"],
                lon=view["lon"],
                mode="markers",
                name="歷史事故點",
                marker=dict(size=6, color="#4C78A8", opacity=0.45),
                text=view["district"],
                hovertemplate="歷史事故點<br>行政區: %{text}<extra></extra>",
            )
        )

    high = view[view["risk_level"] == "High"]
    if show_high and not high.empty:
        fig.add_trace(
            go.Scattermap(
                lat=high["lat"],
                lon=high["lon"],
                mode="markers",
                name="模型高風險預測點",
                marker=dict(size=10, color="red", opacity=0.8),
                text=(high["district"] + " | P=" + high["pred_prob"].round(2).astype(str)),
                hovertemplate="模型高風險預測點<br>%{text}<extra></extra>",
            )
        )

    if show_patrol:
        patrol = (
            high.groupby("district", as_index=False)
            .agg(lat=("lat", "mean"), lon=("lon", "mean"), cnt=("district", "count"), p=("pred_prob", "mean"))
            .sort_values("cnt", ascending=False)
            .head(8)
        )
        if not patrol.empty:
            fig.add_trace(
                go.Scattermap(
                    lat=patrol["lat"],
                    lon=patrol["lon"],
                    mode="markers+text",
                    name="建議巡邏區域",
                    marker=dict(size=16, color="#F39C12", symbol="star"),
                    text=patrol["district"],
                    textposition="top right",
                    hovertemplate="建議巡邏區域<br>%{text}<br>高風險點數: %{customdata[0]}<br>平均P: %{customdata[1]:.2f}<extra></extra>",
                    customdata=np.stack([patrol["cnt"], patrol["p"]], axis=1),
                )
            )

    fig.update_layout(
        map=dict(style="open-street-map", center=TAICHUNG_CENTER, zoom=11),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


st.title("🚦 基於臺中市交通事故資料之事故嚴重程度預測與決策支援分析（Demo）")
st.caption("資料範圍：114 年 10~12 月。可先用內建假資料展示流程，再替換為臺中市警察局實際資料。")

with st.sidebar:
    st.header("控制面板")
    uploaded = st.file_uploader("上傳臺中市事故資料（CSV）", type=["csv"])
    threshold_high = st.slider("高風險門檻 (P ≥)", 0.5, 0.9, 0.7, 0.01)
    threshold_mid = st.slider("中風險下限 (P ≥)", 0.2, 0.6, 0.4, 0.01)
    st.markdown("---")
    st.caption("地圖圖層開關已移到主畫面地圖上方，避免找不到。")
    st.markdown("---")
    st.write("若未上傳資料，系統使用臺中市範圍之內建 Demo 資料。")

if uploaded is not None:
    df = normalize_uploaded_df(pd.read_csv(uploaded))
    required_cols = {"pred_prob", "hour", "lat", "lon"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV 缺少必要欄位: {required_cols}")
        st.stop()
    if "city" in df.columns:
        df = df[df["city"].astype(str).str.contains("台中|臺中", na=False)].copy()
else:
    df = generate_demo_data()

if df.empty:
    st.error("目前篩選後無臺中市資料，請確認上傳資料內容。")
    st.stop()

if threshold_mid >= threshold_high:
    st.error("中風險下限必須小於高風險門檻")
    st.stop()

df["risk_level"] = pd.cut(df["pred_prob"], bins=[-0.01, threshold_mid, threshold_high, 1.0], labels=["Low", "Medium", "High"])
df["strategy"] = df["risk_level"].astype(str).map(strategy_by_risk)
if "month" not in df.columns:
    df["month"] = 10
if "district" not in df.columns:
    df["district"] = "未提供"
if "accident_type" not in df.columns:
    df["accident_type"] = "未提供"

sel_month = st.multiselect("篩選月份", sorted(df["month"].dropna().unique().tolist()), default=sorted(df["month"].dropna().unique().tolist()))
sel_district = st.multiselect("篩選行政區", sorted(df["district"].dropna().unique().tolist()), default=sorted(df["district"].dropna().unique().tolist()))
view = df[df["month"].isin(sel_month) & df["district"].isin(sel_district)].copy()
if view.empty:
    st.warning("目前篩選條件下沒有資料。")
    st.stop()

tab1, tab2, tab3 = st.tabs(["決策總覽", "模型效能頁", "關鍵因子頁"])

with tab1:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("事故總數", f"{len(view):,}")
    k2.metric("高風險事件", f"{(view['risk_level'] == 'High').sum():,}")
    k3.metric("平均預測風險", f"{view['pred_prob'].mean():.2f}")
    k4.metric("死亡數(示意)", f"{view.get('deaths', pd.Series([0]*len(view))).sum():,}")
    k5.metric("受傷數(示意)", f"{view.get('injuries', pd.Series([0]*len(view))).sum():,}")
    k6.metric("更新時間", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    left, right = st.columns([2, 1])
    with left:
        st.subheader("臺中市事故風險地圖（可開關圖層）")
        layer_labels = {
            "歷史事故點": "show_hist",
            "模型高風險預測點": "show_high",
            "建議巡邏區域": "show_patrol",
        }
        selected_layers = st.multiselect(
            "選擇要顯示的地圖圖層",
            options=list(layer_labels.keys()),
            default=list(layer_labels.keys()),
            help="可同時勾選多個圖層。",
        )
        show_hist = "歷史事故點" in selected_layers
        show_high = "模型高風險預測點" in selected_layers
        show_patrol = "建議巡邏區域" in selected_layers
        st.plotly_chart(build_layered_map(view, show_hist, show_high, show_patrol), use_container_width=True)
    with right:
        st.subheader("決策建議清單（Top 15）")
        rec = (
            view.sort_values("pred_prob", ascending=False)
            .loc[:, ["district", "hour", "pred_prob", "risk_level", "strategy"]]
            .head(15)
        )
        st.dataframe(rec, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("高風險時段分析")
        hourly = view.assign(is_high=(view["risk_level"] == "High").astype(int)).groupby("hour", as_index=False)["is_high"].sum()
        st.plotly_chart(px.bar(hourly, x="hour", y="is_high", labels={"is_high": "高風險件數", "hour": "小時"}), use_container_width=True)
    with c2:
        st.subheader("臺中市各行政區風險比較")
        area = view.groupby("district", as_index=False)["pred_prob"].mean().sort_values("pred_prob", ascending=False)
        st.plotly_chart(px.bar(area, x="district", y="pred_prob", labels={"pred_prob": "平均風險機率", "district": "行政區"}), use_container_width=True)

with tab2:
    st.subheader("模型效能頁")
    st.caption("目前為 Demo 佔位資料；可替換為你們實際訓練結果。")

    model_scores = generate_demo_model_scores()
    st.dataframe(model_scores, use_container_width=True, hide_index=True)

    m1, m2 = st.columns(2)
    with m1:
        fig_recall = px.bar(model_scores, x="Model", y="Recall", color="Model", title="Recall 比較（重點指標）")
        st.plotly_chart(fig_recall, use_container_width=True)
    with m2:
        fig_pr = px.bar(model_scores, x="Model", y="PR_AUC", color="Model", title="PR-AUC 比較")
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("**Confusion Matrix（示意）**")
    cm = np.array([[410, 66], [34, 190]])
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual", color="Count"), x=["Non-severe", "Severe"], y=["Non-severe", "Severe"])
    st.plotly_chart(cm_fig, use_container_width=True)

with tab3:
    st.subheader("關鍵因子頁")
    st.caption("目前為 Demo 佔位資料；可替換為 SHAP 或特徵重要性結果。")

    fi = generate_demo_feature_importance()
    fig_fi = px.bar(fi.sort_values("importance", ascending=True), x="importance", y="feature", orientation="h", title="特徵重要性（Demo）")
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("**解讀建議（可替換為 SHAP 文案）**")
    st.write("- 天候、路口型態、照明、速限等因子在示意模型中影響較大。")
    st.write("- 可在此頁放入 SHAP summary plot、dependence plot、單筆個案解釋。")
