import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Traffic Risk DSS Demo", layout="wide")


@st.cache_data
def generate_demo_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = ["Taipei", "New Taipei", "Taoyuan", "Taichung", "Tainan", "Kaohsiung"]
    districts = ["A", "B", "C", "D", "E"]
    weather = ["Sunny", "Cloudy", "Rainy"]
    road_category = ["Urban", "Provincial", "County"]
    accident_types = ["Vehicle-Vehicle", "Vehicle-Pedestrian", "Single Vehicle"]

    months = rng.choice([10, 11, 12], size=n, p=[0.33, 0.34, 0.33])
    hours = rng.integers(0, 24, size=n)
    cities_col = rng.choice(cities, size=n)
    district_col = rng.choice(districts, size=n)
    weather_col = rng.choice(weather, size=n, p=[0.5, 0.3, 0.2])

    base = (
        0.15
        + (hours >= 22) * 0.25
        + (hours <= 5) * 0.20
        + (weather_col == "Rainy") * 0.15
        + rng.normal(0, 0.08, n)
    )
    pred_prob = np.clip(base, 0.01, 0.99)
    severity_bin = (pred_prob > 0.45).astype(int)

    risk_level = pd.cut(
        pred_prob,
        bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
    )

    lat_base = {
        "Taipei": 25.04,
        "New Taipei": 25.01,
        "Taoyuan": 24.99,
        "Taichung": 24.15,
        "Tainan": 22.99,
        "Kaohsiung": 22.63,
    }
    lon_base = {
        "Taipei": 121.56,
        "New Taipei": 121.46,
        "Taoyuan": 121.30,
        "Taichung": 120.67,
        "Tainan": 120.20,
        "Kaohsiung": 120.30,
    }

    lats = [lat_base[c] + rng.normal(0, 0.03) for c in cities_col]
    lons = [lon_base[c] + rng.normal(0, 0.03) for c in cities_col]

    df = pd.DataFrame(
        {
            "year": 2025,
            "month": months,
            "day": rng.integers(1, 29, size=n),
            "hour": hours,
            "city": cities_col,
            "district": district_col,
            "weather": weather_col,
            "road_category": rng.choice(road_category, size=n),
            "accident_type": rng.choice(accident_types, size=n),
            "pred_prob": pred_prob,
            "risk_level": risk_level,
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


st.title("🚦 交通事故風險決策支援 Dashboard（Demo 版）")
st.caption("可先用內建假資料展示流程，之後再替換成真實資料。")

with st.sidebar:
    st.header("控制面板")
    uploaded = st.file_uploader("上傳資料（CSV）", type=["csv"])
    threshold_high = st.slider("高風險門檻 (P ≥)", 0.5, 0.9, 0.7, 0.01)
    threshold_mid = st.slider("中風險下限 (P ≥)", 0.2, 0.6, 0.4, 0.01)
    st.markdown("---")
    st.write("若未上傳資料，系統會使用內建 Demo 資料。")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    required_cols = {"pred_prob", "city", "hour", "lat", "lon"}
    if not required_cols.issubset(df.columns):
        st.error(f"CSV 缺少必要欄位: {required_cols}")
        st.stop()
else:
    df = generate_demo_data()

bins = [-0.01, threshold_mid, threshold_high, 1.0]
labels = ["Low", "Medium", "High"]
if threshold_mid >= threshold_high:
    st.error("中風險下限必須小於高風險門檻")
    st.stop()

df["risk_level"] = pd.cut(df["pred_prob"], bins=bins, labels=labels)
df["strategy"] = df["risk_level"].astype(str).map(strategy_by_risk)

cities = sorted(df["city"].dropna().unique().tolist())
sel_city = st.multiselect("篩選城市", cities, default=cities)
sel_month = st.multiselect("篩選月份", sorted(df["month"].unique().tolist()), default=sorted(df["month"].unique().tolist()))

view = df[df["city"].isin(sel_city) & df["month"].isin(sel_month)].copy()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("事故總數", f"{len(view):,}")
k2.metric("高風險事件", f"{(view['risk_level'] == 'High').sum():,}")
k3.metric("平均預測風險", f"{view['pred_prob'].mean():.2f}")
k4.metric("死亡數(示意)", f"{view.get('deaths', pd.Series([0]*len(view))).sum():,}")
k5.metric("受傷數(示意)", f"{view.get('injuries', pd.Series([0]*len(view))).sum():,}")
k6.metric("更新時間", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

left, right = st.columns([2, 1])
with left:
    st.subheader("事故風險地圖")
    fig_map = px.scatter_map(
        view,
        lat="lat",
        lon="lon",
        color="risk_level",
        color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"},
        hover_data=["city", "hour", "pred_prob", "strategy", "accident_type"],
        zoom=6,
        height=550,
    )
    fig_map.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with right:
    st.subheader("決策建議清單（Top 15）")
    rec = (
        view.sort_values("pred_prob", ascending=False)
        .loc[:, ["city", "district", "hour", "pred_prob", "risk_level", "strategy"]]
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
    fig_hour = px.bar(hourly, x="hour", y="is_high", labels={"is_high": "高風險件數", "hour": "小時"})
    st.plotly_chart(fig_hour, use_container_width=True)

with c2:
    st.subheader("區域風險比較")
    area = (
        view.groupby("city", as_index=False)["pred_prob"]
        .mean()
        .sort_values("pred_prob", ascending=False)
    )
    fig_area = px.bar(area, x="city", y="pred_prob", labels={"pred_prob": "平均風險機率", "city": "城市"})
    st.plotly_chart(fig_area, use_container_width=True)

st.markdown("---")
st.subheader("模型說明（Demo 佔位）")
st.write("- 後續可接入 Random Forest / AdaBoost / XGBoost / Logistic Regression 結果。")
st.write("- 可加入 Confusion Matrix、ROC-AUC、PR-AUC、SHAP 圖表。")
