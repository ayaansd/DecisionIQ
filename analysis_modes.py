from agents.summarize_insights import generate_summary_from_df

def generate_swot_analysis(df, model_mode="cloud"):
    prompt = "Perform a SWOT (Strengths, Weaknesses, Opportunities, Threats) analysis based on this business data."
    return generate_summary_from_df(df, custom_prompt=prompt, model_mode=model_mode)

def generate_financial_analysis(df, model_mode="cloud"):
    prompt = "Analyze financial metrics, trends, outliers, and offer business insights based on this data."
    return generate_summary_from_df(df, custom_prompt=prompt, model_mode=model_mode)

def generate_market_research(df, model_mode="cloud"):
    prompt = "Analyze this data for market trends, customer behavior, segment performance, and strategic positioning."
    return generate_summary_from_df(df, custom_prompt=prompt, model_mode=model_mode)

def generate_process_optimization(df, model_mode="cloud"):
    prompt = "Analyze this data to identify inefficiencies and suggest improvements in processes or workflows."
    return generate_summary_from_df(df, custom_prompt=prompt, model_mode=model_mode)
