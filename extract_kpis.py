import pandas as pd

def _is_financial_metric(name):
    keywords = ['sales', 'revenue', 'price', 'amount', 'cost', 'value', 'profit', 'income', 'budget', 'spend', 'transaction', 'order', 'purchase', 'total', 'net', 'gross']
    return any(k in name.lower() for k in keywords)

def _get_variability(stats):
    if 'std' in stats and 'mean' in stats and stats['mean'] != 0:
        cv = stats['std'] / stats['mean']
        if cv > 1:
            return "high variability"
        elif cv < 0.1:
            return "low variability"
    return None

def extract_kpis(df):
    if df is None or df.empty:
        return {"error": "DataFrame is empty."}

    try:
        num_df = df.select_dtypes(include='number')
        stats = num_df.describe()
        kpis_output = {}

        for col in stats.columns:
            col_stats = stats[col].to_dict()
            insights = []

            # Capture stats
            formatted_stats = {stat: f"{val:,.2f}" if isinstance(val, (int, float)) else str(val) for stat, val in col_stats.items()}

            # Capture insights
            if _is_financial_metric(col):
                insights.append(f"💡 '{col}' looks like a key financial metric.")
            
            var_note = _get_variability(col_stats)
            if var_note:
                insights.append(f"💡 '{col}' shows {var_note}.")

            kpis_output[col] = {
                "stats": formatted_stats,
                "insights": insights
            }

        return {"kpis": kpis_output, "message": "KPI Extraction Completed."}

    except Exception as e:
        return {"error": f"❌ Error extracting KPIs: {e}"}