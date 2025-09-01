import pandas as pd

def perform_eda(df):
    """
    Performs basic EDA and returns a structured dictionary with all results.
    """
    eda_results = {}
    observations = []

    try:
        observations.append("--- 🔍 Performing EDA ---")

        # --- Dataset Shape ---
        num_rows, num_cols = df.shape
        eda_results["shape"] = {"rows": num_rows, "columns": num_cols}
        observations.append(f"📊 Rows: {num_rows}, Columns: {num_cols}")

        # --- Missing Values ---
        missing = df.isnull().sum()
        missing_filtered = missing[missing > 0]
        eda_results["missing_values"] = missing_filtered.to_dict()
        total_cells = num_rows * num_cols
        total_missing_cells = missing.sum()
        missing_percentage = (total_missing_cells / total_cells * 100) if total_cells > 0 else 0

        if not missing_filtered.empty:
            observations.append("\n🚫 Missing values per column:")
            observations.append(missing_filtered.to_string())
            observations.append(f"💡 Observation: Overall, {total_missing_cells} cells ({missing_percentage:.2f}%) are missing in the dataset.")
        else:
            observations.append("✅ No missing values found.")
            observations.append("💡 Observation: The dataset is complete with no missing values.")

        # --- Categorical Value Counts ---
        eda_results["categorical_value_counts"] = {}
        categorical_columns = df.select_dtypes(include='object')

        if not categorical_columns.empty:
            observations.append("\n--- 🏷️ Categorical Column Value Counts (Top 5) ---")
            for col in categorical_columns.columns:
                unique_count = df[col].nunique()
                total_rows = df.shape[0]
                top_values = df[col].value_counts().head(5).to_dict()
                eda_results["categorical_value_counts"][col] = top_values

                observations.append(f"\n  Column: {col} – Top 5 Values:")
                for val, count in top_values.items():
                    observations.append(f"    {val}: {count}")

                if unique_count == 2:
                    observations.append(f"  💡 Observation: '{col}' appears to be a binary/boolean flag.")
                elif unique_count > (total_rows * 0.8):
                    observations.append(f"  💡 Observation: '{col}' has a very high number of unique values ({unique_count}), suggesting it might be an identifier or free-form text.")
                elif unique_count < (total_rows * 0.1) and unique_count > 2:
                    observations.append(f"  💡 Observation: '{col}' has a manageable number of unique categories ({unique_count}), suitable for group-by analysis.")
                else:
                    observations.append(f"  💡 Observation: '{col}' has {unique_count} unique values.")
        else:
            observations.append("\n⚠️ No categorical columns found for value counts.")
            observations.append("  💡 Observation: This dataset appears to be primarily numerical.")

        # ========== SMART AGENT LAYER ==========
        observations.append("\n--- 🧠 Smart Agent Observations ---")
        numeric_df = df.select_dtypes(include='number')

        # 1. High Cardinality Identifier-like Columns
        high_unique_cols = [col for col in df.columns if df[col].nunique() > 0.8 * len(df)]
        for col in high_unique_cols:
            observations.append(f"💡 '{col}' has {df[col].nunique()} unique values — likely an identifier or free-text column.")

        # 2. High Variability Columns
        for col in numeric_df.columns:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            if mean != 0 and std / mean > 1.0:
                observations.append(f"💡 '{col}' shows high variability (std > mean). A histogram or boxplot might help.")

        # 3. Skewed Distributions
        skew_vals = numeric_df.skew()
        for col, skew in skew_vals.items():
            if abs(skew) > 1:
                observations.append(f"💡 '{col}' is highly skewed (skew = {skew:.2f}). Consider a boxplot or transformation.")

        # 4. Strong Correlations
        corr_matrix = numeric_df.corr().abs()
        checked_pairs = set()
        for col1 in corr_matrix.columns:
            for col2 in corr_matrix.columns:
                if col1 != col2 and corr_matrix.loc[col1, col2] > 0.7:
                    pair = tuple(sorted((col1, col2)))
                    if pair not in checked_pairs:
                        observations.append(f"💡 '{col1}' and '{col2}' are strongly correlated (r = {corr_matrix.loc[col1, col2]:.2f}). A scatter plot could visualize this.")
                        checked_pairs.add(pair)
        
        eda_results["observations"] = "\n".join(observations)
        
        return eda_results

    except Exception as e:
        return {"error": f"❌ Error during EDA: {e}"}

if __name__ == '__main__':
    data = {'col1': [1, 2, 3, 4, 5], 'col2': ['A', 'B', 'C', 'D', 'E']}
    sample_df = pd.DataFrame(data)
    eda_results = perform_eda(sample_df)
    import json
    print("EDA Results:")
    print(json.dumps(eda_results, indent=2))