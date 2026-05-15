import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from datetime import datetime
from pathlib import Path
from typing import Optional

st.set_page_config(page_title="Taichung Traffic Risk DSS Demo", layout="wide")

TAICHUNG_CENTER = {"lat": 24.1477, "lon": 120.6736}
TAICHUNG_DISTRICTS = [
    "中區", "東區", "南區", "西區", "北區", "北屯區", "西屯區", "南屯區",
    "太平區", "大里區", "霧峰區", "烏日區", "豐原區", "后里區", "石岡區", "東勢區",
    "新社區", "潭子區", "大雅區", "神岡區", "大肚區", "沙鹿區", "龍井區", "梧棲區",
    "清水區", "大甲區", "外埔區", "大安區", "和平區"
]

PROCESSED_DATA_PATH = Path("MDS_model/MDS_model/processed_data.csv")
DEFAULT_XGB_DATA_PATH = Path("MDS_model/MDS_model/selected_data_6m_0514.csv")
LR_PROB_PATH = Path("MDS_model/MDS_model/lr_prob.csv")
SHAP_ARTIFACT_PATH = Path("MDS_model/MDS_model/shap_dashboard_data.npz")
CONTINUOUS_COLS = ["GPS座標X", "GPS座標Y", "hour", "month", "weekday", "道路速限"]
TARGET_COL = "Y_事故嚴重度"
MONTH_SEQUENCE = [7, 8, 9, 10, 11, 12]


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


def infer_months_by_order(n_rows: int) -> np.ndarray:
    month_idx = np.floor(np.arange(n_rows) * len(MONTH_SEQUENCE) / max(n_rows, 1)).astype(int)
    month_idx = np.clip(month_idx, 0, len(MONTH_SEQUENCE) - 1)
    return np.asarray(MONTH_SEQUENCE, dtype=int)[month_idx]


def infer_hour(row: pd.Series) -> int:
    if pd.notna(row.get("hour", np.nan)):
        return int(row.get("hour", 0))
    if pd.notna(row.get("時刻(小時)", np.nan)):
        return int(row.get("時刻(小時)", 0))
    if row.get("time_period_morning_rush", 0) == 1 or row.get("is_morning_rush", 0) == 1:
        return 8
    if row.get("time_period_evening_rush", 0) == 1 or row.get("is_evening_rush", 0) == 1:
        return 17
    if row.get("is_night", 0) == 1:
        return 22
    if row.get("time_period_daytime", 0) == 1:
        return 13
    return 0


def infer_districts(df: pd.DataFrame) -> pd.Series:
    district_cols = [col for col in df.columns if col.startswith("區_")]
    if not district_cols:
        district_cols = [col for col in TAICHUNG_DISTRICTS if col in df.columns]
    if not district_cols:
        return pd.Series("未提供", index=df.index)

    active = df[district_cols].fillna(0).astype(float)
    district = active.idxmax(axis=1).str.replace("區_", "", regex=False)
    has_district = active.max(axis=1) > 0
    return district.where(has_district, "未提供")


def prepare_dashboard_columns(df: pd.DataFrame, actual_severity: Optional[pd.Series] = None) -> pd.DataFrame:
    out = df.copy()
    if not {"GPS座標X", "GPS座標Y"}.issubset(out.columns):
        raise ValueError("CSV 缺少 GPS座標X / GPS座標Y，無法繪製熱區地圖。")

    out["lat"] = pd.to_numeric(out["GPS座標Y"], errors="coerce")
    out["lon"] = pd.to_numeric(out["GPS座標X"], errors="coerce")
    out["month"] = pd.to_numeric(out.get("month", out.get("月", pd.Series(np.nan, index=out.index))), errors="coerce")
    if out["month"].isna().all():
        out["month"] = infer_months_by_order(len(out))
    out["month"] = out["month"].fillna(0).astype(int)
    out["hour"] = out.apply(infer_hour, axis=1)
    out["district"] = infer_districts(out)
    out["accident_type"] = "未提供"
    out["case_id"] = out["序號"] if "序號" in out.columns else np.arange(1, len(out) + 1)
    if actual_severity is not None:
        out["actual_severity"] = actual_severity.to_numpy()
    return out.dropna(subset=["lat", "lon"]).copy()


@st.cache_data(show_spinner="正在以 XGBoost 產生 7-12 月風險結果...")
def load_xgboost_dashboard_data(csv_source) -> pd.DataFrame:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("尚未安裝 xgboost，請先安裝套件後再啟動 dashboard。") from exc

    if isinstance(csv_source, bytes):
        raw = pd.read_csv(io.BytesIO(csv_source))
    else:
        raw = pd.read_csv(csv_source)
    if TARGET_COL not in raw.columns:
        raise ValueError(f"CSV 缺少目標欄位：{TARGET_COL}")
    model_df = raw.drop(columns=["序號"], errors="ignore").copy()
    X = model_df.drop(columns=[TARGET_COL]).fillna(0)
    y = model_df[TARGET_COL].astype(int)

    if y.nunique() < 2:
        pred_prob = np.repeat(float(y.iloc[0]), len(y))
    else:
        scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.2,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X, y)
        pred_prob = model.predict_proba(X)[:, 1]

    df = raw.copy()
    df["pred_prob"] = pred_prob
    return prepare_dashboard_columns(df, actual_severity=y)


@st.cache_data(show_spinner="正在載入 Logistic Regression 預測結果...")
def load_lr_dashboard_data(source_path: str, prob_path: str) -> pd.DataFrame:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError("尚未安裝 scikit-learn，無法重建 lr_prob.csv 對應的測試集。") from exc

    source = pd.read_csv(source_path)
    prob = pd.read_csv(prob_path)
    if "y_prob" not in prob.columns:
        raise ValueError("lr_prob.csv 缺少 y_prob 欄位。")
    if "事故嚴重度" not in source.columns:
        raise ValueError("processed_data.csv 缺少 事故嚴重度 欄位，無法重建測試集。")

    y = source["事故嚴重度"].astype(int).reset_index(drop=True)
    all_idx = np.arange(len(source))
    _, test_idx = train_test_split(
        all_idx,
        test_size=len(prob),
        random_state=42,
        stratify=y,
    )
    actual = y.iloc[test_idx].reset_index(drop=True)
    if "y_true" in prob.columns and not actual.equals(prob["y_true"].astype(int).reset_index(drop=True)):
        raise ValueError(
            "lr_prob.csv 無法用 processed_data.csv 的 random_state=42 stratified test split 對回。"
            "請重新輸出包含序號的預測檔。"
        )

    df = source.iloc[test_idx].reset_index(drop=True).copy()
    df["pred_prob"] = pd.to_numeric(prob["y_prob"], errors="coerce")
    if "prediction" in prob.columns:
        df["model_prediction"] = prob["prediction"]
    return prepare_dashboard_columns(df, actual_severity=actual)


def make_monthly_risk_timeline(df: pd.DataFrame, selected_month: int) -> go.Figure:
    monthly = (
        df.assign(is_high=(df["risk_level"] == "High").astype(int))
        .groupby("month", as_index=False)
        .agg(
            avg_risk=("pred_prob", "mean"),
            high_count=("is_high", "sum"),
            total=("pred_prob", "size"),
        )
        .sort_values("month")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["avg_risk"],
            mode="lines+markers",
            name="平均 XGBoost 風險",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=9),
            customdata=np.stack([monthly["high_count"], monthly["total"]], axis=1),
            hovertemplate="%{x}月<br>平均風險: %{y:.3f}<br>高風險件數: %{customdata[0]:,}<br>事故數: %{customdata[1]:,}<extra></extra>",
        )
    )
    fig.add_vline(x=selected_month, line_width=2, line_dash="dash", line_color="#D62728")
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="月份", tickmode="array", tickvals=MONTH_SEQUENCE),
        yaxis=dict(title="平均預測風險", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig



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

with st.sidebar:
    st.header("控制面板")
    model_source = st.selectbox(
        "總攬頁模型來源",
        ["Logistic Regression (lr_prob.csv)", "XGBoost 即時計算"],
    )
    uploaded = st.file_uploader("上傳臺中市事故資料（CSV）", type=["csv"])
    threshold_high = st.slider("高風險門檻 (P ≥)", 0.5, 0.9, 0.7, 0.01)
    threshold_mid = st.slider("中風險下限 (P ≥)", 0.2, 0.6, 0.4, 0.01)
    st.markdown("---")
    st.subheader("地圖圖層開關")
    show_hist = st.checkbox("歷史事故熱區", value=True)
    show_high = st.checkbox("模型高風險熱區", value=True)
    show_patrol = st.checkbox("建議巡邏區域", value=True)
    st.markdown("---")

model_label = "Logistic Regression" if model_source.startswith("Logistic") else "XGBoost"
st.caption(f"資料範圍：7~12 月，總攬頁使用 {model_label} 預測風險")

if model_source.startswith("Logistic"):
    if uploaded is not None:
        st.warning("Logistic Regression 模式會使用 processed_data.csv + lr_prob.csv 對回測試集；上傳檔只適用於 XGBoost 即時計算模式。")
    try:
        df = load_lr_dashboard_data(str(PROCESSED_DATA_PATH), str(LR_PROB_PATH))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
        st.stop()
elif uploaded is not None:
    try:
        df = load_xgboost_dashboard_data(uploaded.getvalue())
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
else:
    try:
        df = load_xgboost_dashboard_data(str(DEFAULT_XGB_DATA_PATH))
    except (RuntimeError, ValueError) as exc:
        st.warning(f"{exc} 目前改用示範資料。")
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
    df["month"] = infer_months_by_order(len(df))
if "district" not in df.columns:
    df["district"] = "未提供"
if "accident_type" not in df.columns:
    df["accident_type"] = "未提供"

month_options = [month for month in MONTH_SEQUENCE if month in set(df["month"].dropna().astype(int))]
if not month_options:
    month_options = sorted(df["month"].dropna().astype(int).unique().tolist())
sel_month = st.select_slider(
    "7~12 月連續時間軸",
    options=month_options,
    value=month_options[0],
    format_func=lambda month: f"{int(month)}月",
)
sel_district = st.multiselect("篩選行政區", sorted(df["district"].dropna().unique().tolist()), default=sorted(df["district"].dropna().unique().tolist()))
trend_view = df[df["district"].isin(sel_district)].copy()
view = trend_view[trend_view["month"].eq(sel_month)].copy()
if view.empty:
    st.warning("目前篩選條件下沒有資料。")
    st.stop()

tab1, tab2, tab3 = st.tabs(["決策總覽", "模型效能", "關鍵因子"])

with tab1:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("事故總數", f"{len(view):,}")
    k2.metric("高風險事件", f"{(view['risk_level'] == 'High').sum():,}")
    k3.metric("平均預測風險", f"{view['pred_prob'].mean():.2f}")
    k4.metric("實際嚴重事故", f"{view.get('actual_severity', pd.Series([0]*len(view))).sum():,}")
    k5.metric("時間軸月份", f"{int(sel_month)}月")
    k6.metric("更新時間", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

    st.plotly_chart(make_monthly_risk_timeline(trend_view, int(sel_month)), width="stretch")

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"臺中市事故風險熱區地圖（{int(sel_month)}月）")
        st.plotly_chart(build_layered_map(view, show_hist, show_high, show_patrol), width="stretch")
    with right:
        st.subheader(f"決策建議清單（{int(sel_month)}月 Top 15）")
        decision_cols = ["case_id", "district", "hour", "pred_prob", "risk_level", "strategy"]
        decision_cols = [col for col in decision_cols if col in view.columns]
        rec = (
            view.sort_values("pred_prob", ascending=False)
            .loc[:, decision_cols]
            .head(15)
        )
        rec = rec.rename(
            columns={
                "case_id": "案件序號",
                "district": "行政區",
                "hour": "推估時段",
                "pred_prob": f"{model_label}風險",
                "risk_level": "風險等級",
                "strategy": "決策策略",
            }
        )
        st.dataframe(rec, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("高風險時段分析")
        hourly = view.assign(is_high=(view["risk_level"] == "High").astype(int)).groupby("hour", as_index=False)["is_high"].sum()
        st.plotly_chart(px.bar(hourly, x="hour", y="is_high", labels={"is_high": "高風險件數", "hour": "小時"}), width="stretch")
    with c2:
        st.subheader("臺中市各行政區風險比較")
        area = view.groupby("district", as_index=False)["pred_prob"].mean().sort_values("pred_prob", ascending=False)
        st.plotly_chart(px.bar(area, x="district", y="pred_prob", labels={"pred_prob": "平均風險機率", "district": "行政區"}), width="stretch")

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
    st.dataframe(model_scores, width="stretch", hide_index=True)

    m1, m2 = st.columns(2)
    with m1:
        fig_recall = px.bar(model_scores, x="Model", y="Recall", color="Model", title="Recall 比較")
        st.plotly_chart(fig_recall, width="stretch")
    with m2:
        fig_pr = px.bar(model_scores, x="Model", y="PR_AUC", color="Model", title="PR-AUC 比較")
        st.plotly_chart(fig_pr, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion Matrix(Logistic Regression)**")
        cm = np.array([[1012, 675], [997, 1149]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, width="stretch")
    with c2:
        st.markdown("**Confusion Matrix(Random Forest)**")
        cm = np.array([[924, 763], [895, 1251]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Confusion Matrix(AdaBoost)**")
        cm = np.array([[714, 973], [656, 1490]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, width="stretch")
    with c4:
        st.markdown("**Confusion Matrix(XGBoost)**")
        cm = np.array([[806, 881], [832, 1314]])
        cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="True", color="Count"), x=['致命', '非致命'], y=['非致命', '致命'])
        st.plotly_chart(cm_fig, width="stretch")

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
                    width="stretch",
                )
                st.plotly_chart(
                    make_shap_beeswarm(model_shap, f"{model_name} - Beeswarm Plot"),
                    width="stretch",
                )
                st.plotly_chart(
                    make_shap_waterfall(model_shap, f"{model_name} - Waterfall Plot"),
                    width="stretch",
                )

    st.markdown("**解讀建議:**")
    st.write("- Summary Bar Plot 用來比較整體特徵重要性。")
    st.write("- Beeswarm Plot 用來觀察特徵值高低如何推升或降低致命事故預測。")
    st.write("- Waterfall Plot 用來解釋單一事故案例的預測來源。")
