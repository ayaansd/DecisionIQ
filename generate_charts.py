# charts/generate_charts.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from colorama import Fore, Style, init
from scipy.stats import skew

init(autoreset=True)

def smart_chart_agent(df, output_dir="charts_output"):
    print(Fore.CYAN + "\n--- 📈 Starting Chart Generation ---" + Style.RESET_ALL)

    if df is None or df.empty:
        print(Fore.RED + "❌ Error: DataFrame is empty or not loaded." + Style.RESET_ALL)
        return [], []

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving charts to: {os.path.abspath(output_dir)}")

    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
    charts_count = 0
    chart_paths = []
    chart_summaries = []

    # 1️⃣ Histograms & Boxplots
    print(Fore.YELLOW + "\nGenerating Histograms and Boxplots..." + Style.RESET_ALL)
    for col in numerical_cols:
        try:
            col_skew = skew(df[col].dropna())

            # Histogram
            hist_path = os.path.join(output_dir, f'{col}_histogram.png')
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Histogram of {col} (Skew: {col_skew:.2f})')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            chart_paths.append(hist_path)
            chart_summaries.append(f"The distribution of **{col}** shows a skewness of {col_skew:.2f}, indicating {'a strong bias' if abs(col_skew) > 1 else 'a fairly balanced spread'} in values.")
            charts_count += 1

            # Boxplot for highly skewed
            if abs(col_skew) > 1:
                box_path = os.path.join(output_dir, f'{col}_boxplot.png')
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=df[col].dropna())
                plt.title(f'Boxplot of {col} (Highly Skewed)')
                plt.xlabel(col)
                plt.tight_layout()
                plt.savefig(box_path)
                plt.close()

                chart_paths.append(box_path)
                chart_summaries.append(f"The boxplot of **{col}** reveals potential outliers due to its high skewness.")
                charts_count += 1
        except Exception as e:
            print(Fore.RED + f"  ❌ Error for '{col}': {e}" + Style.RESET_ALL)

    # 2️⃣ Bar Charts for Categorical Columns
    print(Fore.YELLOW + "\nGenerating Bar Charts for categorical columns..." + Style.RESET_ALL)
    for col in categorical_cols:
        try:
            unique_vals = df[col].nunique()
            if 1 < unique_vals < 50:
                bar_path = os.path.join(output_dir, f'{col}_bar_chart.png')
                plt.figure(figsize=(10, 6))
                top_vals = df[col].value_counts().head(10)
                sns.barplot(x=top_vals.index, y=top_vals.values, palette='viridis')
                plt.title(f'Top Categories in {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(bar_path)
                plt.close()

                chart_paths.append(bar_path)
                chart_summaries.append(f"The most frequent categories in **{col}** are led by {top_vals.index[0]} with {top_vals.iloc[0]} occurrences.")
                charts_count += 1
        except Exception as e:
            print(Fore.RED + f"  ❌ Error generating bar chart for '{col}': {e}" + Style.RESET_ALL)

    # 3️⃣ Scatter Plots for Correlated Pairs
    print(Fore.YELLOW + "\nGenerating Scatter Plots for correlated pairs..." + Style.RESET_ALL)
    try:
        corr_matrix = df[numerical_cols].corr().abs()
        correlated_pairs = [(c1, c2) for c1 in corr_matrix.columns for c2 in corr_matrix.columns
                            if c1 != c2 and corr_matrix.loc[c1, c2] > 0.7]

        for col1, col2 in correlated_pairs:
            scatter_path = os.path.join(output_dir, f'{col1}_vs_{col2}_scatter.png')
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=df[col1], y=df[col2])
            plt.title(f'Scatter: {col1} vs {col2} (Corr: {corr_matrix.loc[col1, col2]:.2f})')
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.tight_layout()
            plt.savefig(scatter_path)
            plt.close()

            chart_paths.append(scatter_path)
            chart_summaries.append(f"There is a strong positive correlation ({corr_matrix.loc[col1, col2]:.2f}) between **{col1}** and **{col2}**.")
            charts_count += 1
    except Exception as e:
        print(Fore.RED + f"  ❌ Error generating scatter plots: {e}" + Style.RESET_ALL)

    # 4️⃣ Time-Series Plots
    print(Fore.YELLOW + "\nGenerating Time-Series Charts..." + Style.RESET_ALL)
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except Exception:
                continue

    if date_col:
        df_sorted = df.sort_values(by=date_col)
        for col in numerical_cols:
            try:
                ts_path = os.path.join(output_dir, f'{col}_time_series.png')
                plt.figure(figsize=(12, 6))
                sns.lineplot(x=df_sorted[date_col], y=df_sorted[col])
                plt.title(f'{col} Over Time')
                plt.xlabel(date_col)
                plt.ylabel(col)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(ts_path)
                plt.close()

                chart_paths.append(ts_path)
                chart_summaries.append(f"The trend of **{col}** over time shows {'an upward' if df_sorted[col].iloc[-1] > df_sorted[col].iloc[0] else 'a downward'} movement.")
                charts_count += 1
            except Exception as e:
                print(Fore.RED + f"  ❌ Error generating time-series for '{col}': {e}" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "  ⚠️ No 'date' or 'time' column found for time-series charts." + Style.RESET_ALL)

    print(Fore.CYAN + f"\n✅ Chart Generation Complete. Total charts created: {charts_count}\n" + Style.RESET_ALL)

    return chart_paths, chart_summaries
