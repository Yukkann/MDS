import numpy as np
import pandas as pd
import plotly.express as px
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
    """Generate Taichung-only synthetic records for October~December 2025 (ROC 114)."""
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


def strategy_by_risk(risk: str) -> str:
    mapping = {
        "High": "優先派遣警力 / 加強巡邏",
        "Medium": "定期巡查",
        "Low": "維持現狀",
    }
    return mapping.get(risk, "維持現狀")


st.title("🚦 基於臺中市交通事故資料之事故嚴重程度預測與決策支援分析（Demo）")
st.caption("資料範圍：114 年 10~12 月。可先用內建假資料展示流程，再替換為臺中市警察局實際資料。")

with st.sidebar:
    st.header("控制面板")
    uploaded = st.file_uploader("上傳臺中市事故資料（CSV）", type=["csv"])
    threshold_high = st.slider("高風險門檻 (P ≥)", 0.5, 0.9, 0.7, 0.01)
    threshold_mid = st.slider("中風險下限 (P ≥)", 0.2, 0.6, 0.4, 0.01)
    st.markdown("---")
    st.write("若未上傳資料，系統使用臺中市範圍之內建 Demo 資料。")

if uploaded is not None:
    df = pd.read_csv(uploaded)
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

bins = [-0.01, threshold_mid, threshold_high, 1.0]
labels = ["Low", "Medium", "High"]
if threshold_mid >= threshold_high:
    st.error("中風險下限必須小於高風險門檻")
    st.stop()

df["risk_level"] = pd.cut(df["pred_prob"], bins=bins, labels=labels)
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

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("事故總數", f"{len(view):,}")
k2.metric("高風險事件", f"{(view['risk_level'] == 'High').sum():,}")
k3.metric("平均預測風險", f"{view['pred_prob'].mean():.2f}")
k4.metric("死亡數(示意)", f"{view.get('deaths', pd.Series([0]*len(view))).sum():,}")
k5.metric("受傷數(示意)", f"{view.get('injuries', pd.Series([0]*len(view))).sum():,}")
k6.metric("更新時間", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

left, right = st.columns([2, 1])
with left:
    st.subheader("臺中市事故風險地圖")
    fig_map = px.scatter_map(
        view,
        lat="lat",
        lon="lon",
        color="risk_level",
        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
        hover_data=["district", "hour", "pred_prob", "strategy", "accident_type"],
        zoom=11,
        center=TAICHUNG_CENTER,
        height=580,
    )
    fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

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
    hourly = (
        view.assign(is_high=(view["risk_level"] == "High").astype(int))
        .groupby("hour", as_index=False)["is_high"]
        .sum()
    )
    st.plotly_chart(px.bar(hourly, x="hour", y="is_high", labels={"is_high": "高風險件數", "hour": "小時"}), use_container_width=True)

with c2:
    st.subheader("臺中市各行政區風險比較")
    area = (
        view.groupby("district", as_index=False)["pred_prob"]
        .mean()
        .sort_values("pred_prob", ascending=False)
    )
    st.plotly_chart(px.bar(area, x="district", y="pred_prob", labels={"pred_prob": "平均風險機率", "district": "行政區"}), use_container_width=True)

st.markdown("---")
st.subheader("模型說明（Demo 佔位）")
st.write("- 後續可接入 Random Forest / AdaBoost / XGBoost / Logistic Regression 結果。")
st.write("- 可加入 Confusion Matrix、ROC-AUC、PR-AUC、SHAP 圖表。")
