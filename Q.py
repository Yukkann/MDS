import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_excel("MDS_Assignment2_kidney.xlsx")

# 把 ? 轉成 NaN
df = df.replace('?', np.nan)

# 把Class缺失刪掉
df = df.dropna(subset=["Class"]).copy()
df["Class"] = df["Class"].astype(str).str.strip()

# 指定要先補值的欄位
num_cols = [
    "Red Blood Cell Count",
    "White Blood Cell Count",
    "Sodium",
    "Potassium",
    "Packed Cell Volume",
    "Hemoglobin",
    "Sugar",
    "Specific Gravity"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 題目指定補值
df["Red Blood Cells"] = df["Red Blood Cells"].fillna(df["Red Blood Cells"].mode()[0])

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# 把 y 轉成 0/1
df["Class"] = df["Class"].map({"ckd": 1, "notckd": 0})

# 再檢查一次 y
# print(df["Class"].value_counts(dropna=False))
# print(df["Class"].unique())

y = df["Class"]
X = df.drop(columns=["Class"])

# dummy encoding
X = pd.get_dummies(X, drop_first=True)

# 補掉 X 剩餘的缺失
X = X.fillna(X.median(numeric_only=True))

# 若還有非數值欄位轉float
X = X.astype(float)

# 加常數
X = sm.add_constant(X)

# 確認沒有 NaN
# print("y missing =", y.isna().sum())
# print("X missing =", X.isna().sum().sum())

# 建模
model = sm.Logit(y, X).fit()

table = pd.DataFrame({
    "estimate": model.params,
    "std.error": model.bse,
    "z value": model.tvalues,
    "p-value": model.pvalues
})

print(table)