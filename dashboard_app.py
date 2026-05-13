import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Taichung Traffic Risk DSS Demo", layout="wide")

TAICHUNG_CENTER = {"lat": 24.1477, "lon": 120.6736}
TAICHUNG_DISTRICTS = [
    "中區", "東區", "南區", "西區", "北區", "北屯區", "西屯區", "南屯區",
    "太平區", "大里區", "霧峰區", "烏日區", "豐原區", "后里區", "石岡區", "東勢區",
    "新社區", "潭子區", "大雅區", "神岡區", "大肚區", "沙鹿區", "龍井區", "梧棲區",
    "清水區", "大甲區", "外埔區", "大安區", "和平區"
]

PROCESSED_DATA_PATH = Path("MDS_model/MDS_model/processed_data.csv")
SHAP_ARTIFACT_PATH = Path("MDS_model/MDS_model/shap_dashboard_data.npz")
CONTINUOUS_COLS = ["GPS座標X", "GPS座標Y", "hour", "month", "weekday", "道路速限"]


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
def load_precomputed_shap_dashboard_data(artifact_path: str) -> dict:
    path = Path(artifact_path)
    if not path.exists():
        return {
            "error": (
                f"找不到 SHAP 預計算檔：{artifact_path}。"
                "請先執行 `py -3.9 precompute_shap.py`。"
            )
        }

    artifact = np.load(path, allow_pickle=True)
    return {
        "Logistic Regression": {
            "features": artifact["lr_features"].tolist(),
            "values": artifact["lr_values"],
            "data": pd.DataFrame(artifact["lr_data"], columns=artifact["lr_features"].tolist()),
            "base_value": float(artifact["lr_base_value"]),
            "waterfall_idx": int(artifact["waterfall_idx"]),
        },
        "Random Forest": {
            "features": artifact["rf_features"].tolist(),
            "values": artifact["rf_values"],
            "data": pd.DataFrame(artifact["rf_data"], columns=artifact["rf_features"].tolist()),
            "base_value": float(artifact["rf_base_value"]),
            "waterfall_idx": int(artifact["waterfall_idx"]),
        },
    }


def make_shap_summary_bar(shap_data: dict, title: str, top_n: int = 15) -> go.Figure:
    values = shap_data["values"]
    features = np.asarray(shap_data["features"])
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[-top_n:]
    plot_df = pd.DataFrame(
        {
            "feature": features[order],
            "mean_abs_shap": mean_abs[order],
        }
    ).sort_values("mean_abs_shap")
    fig = px.bar(
        plot_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        labels={"mean_abs_shap": "平均 |SHAP value|", "feature": "特徵"},
        title=title,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def make_shap_beeswarm(shap_data: dict, title: str, top_n: int = 15, seed: int = 42) -> go.Figure:
    values = shap_data["values"]
    data = shap_data["data"]
    features = np.asarray(shap_data["features"])
    mean_abs = np.abs(values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    rng = np.random.default_rng(seed)

    rows = []
    for rank, feature_idx in enumerate(top_idx):
        feature = features[feature_idx]
        feature_values = data.iloc[:, feature_idx].to_numpy(dtype=float)
        min_val = np.nanmin(feature_values)
        max_val = np.nanmax(feature_values)
        if np.isclose(min_val, max_val):
            color_values = np.full(len(feature_values), 0.5)
        else:
            color_values = (feature_values - min_val) / (max_val - min_val)
        y_base = len(top_idx) - rank - 1
        jitter = rng.normal(0, 0.08, size=len(feature_values))
        rows.append(
            pd.DataFrame(
                {
                    "shap_value": values[:, feature_idx],
                    "feature": feature,
                    "feature_value": feature_values,
                    "color_value": color_values,
                    "y": y_base + jitter,
                }
            )
        )
    plot_df = pd.concat(rows, ignore_index=True)

    fig = go.Figure(
        go.Scattergl(
            x=plot_df["shap_value"],
            y=plot_df["y"],
            mode="markers",
            marker=dict(
                size=6,
                color=plot_df["color_value"],
                colorscale="RdBu_r",
                colorbar=dict(title="特徵值<br>低 → 高"),
                opacity=0.72,
            ),
            text=plot_df["feature"],
            customdata=np.stack([plot_df["feature_value"]], axis=1),
            hovertemplate="特徵: %{text}<br>SHAP: %{x:.4f}<br>特徵值: %{customdata[0]:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#666")
    fig.update_layout(
        title=title,
        xaxis_title="SHAP value（往右代表推高致命事故預測）",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_idx))),
            ticktext=list(features[top_idx][::-1]),
        ),
        height=620,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def make_shap_waterfall(shap_data: dict, title: str, top_n: int = 12) -> go.Figure:
    values = shap_data["values"]
    data = shap_data["data"]
    features = np.asarray(shap_data["features"])
    idx = shap_data["waterfall_idx"]
    row_values = values[idx]
    order = np.argsort(np.abs(row_values))[::-1]
    top_idx = order[:top_n]
    other_idx = order[top_n:]

    x_labels = [
        f"{features[i]} = {data.iloc[idx, i]:.3g}"
        for i in top_idx
    ]
    y_values = [float(row_values[i]) for i in top_idx]
    if len(other_idx):
        x_labels.append(f"其他 {len(other_idx)} 個特徵")
        y_values.append(float(row_values[other_idx].sum()))

    total_value = shap_data["base_value"] + float(row_values.sum())
    fig = go.Figure(
        go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(y_values) + ["total"],
            x=x_labels + ["模型輸出"],
            y=y_values + [total_value],
            base=shap_data["base_value"],
            connector={"line": {"color": "#888"}},
            increasing={"marker": {"color": "#D62728"}},
            decreasing={"marker": {"color": "#1F77B4"}},
            totals={"marker": {"color": "#333333"}},
            hovertemplate="%{x}<br>貢獻: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="模型輸出值",
        height=560,
        margin=dict(l=10, r=10, t=50, b=80),
    )
    return fig


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
            go.Densitymap(
                lat=view["lat"],
                lon=view["lon"],
                z=np.ones(len(view)),
                radius=22,
                name="歷史事故熱區",
                colorscale="Blues",
                opacity=0.55,
                colorbar=dict(title="事故密度"),
                text=view["district"],
                hovertemplate="歷史事故熱區<br>行政區: %{text}<extra></extra>",
            )
        )

    high = view[view["risk_level"] == "High"]
    if show_high and not high.empty:
        fig.add_trace(
            go.Densitymap(
                lat=high["lat"],
                lon=high["lon"],
                z=high["pred_prob"],
                radius=28,
                name="模型高風險熱區",
                colorscale="YlOrRd",
                opacity=0.72,
                colorbar=dict(title="風險強度", x=1.08),
                text=(high["district"] + " | P=" + high["pred_prob"].round(2).astype(str)),
                hovertemplate="模型高風險熱區<br>%{text}<extra></extra>",
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
                    marker=dict(size=16, color="#F34312", symbol="star"),
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


st.title("臺中市交通事故資料之事故嚴重程度預測與決策支援分析Dashboard（Demo）")
st.caption("資料範圍：114 年 10~12 月")

with st.sidebar:
    st.header("控制面板")
    uploaded = st.file_uploader("上傳臺中市事故資料（CSV）", type=["csv"])
    threshold_high = st.slider("高風險門檻 (P ≥)", 0.5, 0.9, 0.7, 0.01)
    threshold_mid = st.slider("中風險下限 (P ≥)", 0.2, 0.6, 0.4, 0.01)
    st.markdown("---")
    st.subheader("地圖圖層開關")
    show_hist = st.checkbox("歷史事故熱區", value=True)
    show_high = st.checkbox("模型高風險熱區", value=True)
    show_patrol = st.checkbox("建議巡邏區域", value=True)
    st.markdown("---")

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

tab1, tab2, tab3 = st.tabs(["決策總覽", "模型效能", "關鍵因子"])

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
        st.subheader("臺中市事故風險熱區地圖（可開關圖層）")
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

    model_scores = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest", "AdaBoost", "XGBoost"],
            "Accuracy": [0.5638, 0.5674, 0.5750, 0.5531],
            "Precision": [0.6299, 0.6212, 0.6050, 0.5986],
            "Recall": [0.5354, 0.5829, 0.6943, 0.6123],
            "F1": [0.5788, 0.6014, 0.6466, 0.6054],
            "ROC_AUC": [0.5947, 0.5917, 0.5703, 0.5712],
            "PR_AUC": [0.6341, 0.6380, 0.6027, 0.6186],
        }
    )
    st.dataframe(model_scores, use_container_width=True, hide_index=True)

    m1, m2 = st.columns(2)
    with m1:
        fig_recall = px.bar(model_scores, x="Model", y="Recall", color="Model", title="Recall 比較")
        st.plotly_chart(fig_recall, use_container_width=True)
    with m2:
        fig_pr = px.bar(model_scores, x="Model", y="PR_AUC", color="Model", title="PR-AUC 比較")
        st.plotly_chart(fig_pr, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix(Logistic Regression)**")
        cm = np.array([[1012, 675], [997, 1149]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, use_container_width=True)
    with c2:
        st.markdown("**Confusion Matrix(Random Forest)**")
        cm = np.array([[924, 763], [895, 1251]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Confusion Matrix(AdaBoost)**")
        cm = np.array([[714, 973], [656, 1490]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, use_container_width=True)
    with c4:
        st.markdown("**Confusion Matrix(XGBoost)**")
        cm = np.array([[806, 881], [832, 1314]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, use_container_width=True)

with tab3:
    st.subheader("關鍵因子頁")

    st.markdown("**SHAP 模型解釋圖**")
    shap_result = load_precomputed_shap_dashboard_data(str(SHAP_ARTIFACT_PATH))
    if "error" in shap_result:
        st.warning(shap_result["error"])
    else:
        lr_tab, rf_tab = st.tabs(["Logistic Regression", "Random Forest"])
        for model_tab, model_name in [(lr_tab, "Logistic Regression"), (rf_tab, "Random Forest")]:
            with model_tab:
                model_shap = shap_result[model_name]
                st.plotly_chart(
                    make_shap_summary_bar(model_shap, f"{model_name} - Summary Bar Plot"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    make_shap_beeswarm(model_shap, f"{model_name} - Beeswarm Plot"),
                    use_container_width=True,
                )
                st.plotly_chart(
                    make_shap_waterfall(model_shap, f"{model_name} - Waterfall Plot"),
                    use_container_width=True,
                )

    st.markdown("**解讀建議:**")
    st.write("- Summary Bar Plot 用來比較整體特徵重要性。")
    st.write("- Beeswarm Plot 用來觀察特徵值高低如何推升或降低致命事故預測。")
    st.write("- Waterfall Plot 用來解釋單一事故案例的預測來源。")
