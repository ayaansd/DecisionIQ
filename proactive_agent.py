import pandas as pd

def detect_proactive_signals(df: pd.DataFrame) -> str:
    signals = []

    # 1. Detect Potential ID Columns
    id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if id_cols:
        signals.append(f"🔑 Possible ID columns: {', '.join(id_cols)}")

    # 2. High Cardinality Columns
    high_card_cols = [col for col in df.columns if df[col].nunique() > 50 and df[col].dtype == 'object']
    if high_card_cols:
        signals.append(f"📛 High-cardinality categorical columns: {', '.join(high_card_cols)}")

    # 3. Low Variance Numeric Columns
    low_var_cols = [col for col in df.select_dtypes(include='number').columns if df[col].std() < 1e-3]
    if low_var_cols:
        signals.append(f"📉 Low-variance numeric columns: {', '.join(low_var_cols)}")

    # 4. Highly Correlated Features
    corr = df.select_dtypes(include='number').corr()
    correlated_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                correlated_pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 2)))

    if correlated_pairs:
        formatted = ', '.join([f"{a} ↔ {b} (corr={c})" for a, b, c in correlated_pairs])
        signals.append(f"🔗 Highly correlated pairs: {formatted}")

    # 5. Null-heavy columns
    null_cols = [col for col in df.columns if df[col].isna().mean() > 0.5]
    if null_cols:
        signals.append(f"⚠️ Columns with >50% missing values: {', '.join(null_cols)}")

    return "\n".join(signals) if signals else "✅ No major signals detected."
