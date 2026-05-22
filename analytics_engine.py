import pandas as pd
import numpy as np

def generate_crop_analytics(crop_name, matched_market_crop, yield_info, market_df):
    """
    Generates detailed statistical analytics and charting data for a given crop.
    Returns a dictionary suitable for JSON serialization.
    """
    analytics = {
        'crop_name': crop_name,
        'yield': {
            'avg': 'N/A', 'max': 'N/A', 'min': 'N/A', 'records': 0
        },
        'price': {
            'avg': 'N/A', 'max': 'N/A', 'min': 'N/A'
        },
        'latest_month': 'N/A',
        'recent_data': [],
        'chart_data': {'labels': [], 'prices': []}
    }
    
    # 1. Yield Data Processing
    if yield_info and yield_info.get('records', 0) > 0:
        analytics['yield']['avg'] = f"{yield_info.get('avg_yield', 0):.2f}"
        analytics['yield']['max'] = f"{yield_info.get('max_yield', 0):.2f}"
        analytics['yield']['min'] = f"{yield_info.get('min_yield', 0):.2f}"
        analytics['yield']['records'] = yield_info.get('records', 0)
        
    # 2. Market Data Processing
    if matched_market_crop and market_df is not None:
        market_copy = market_df.copy()
        
        # Safely convert price to numeric
        if 'Modal Price 21-01-2021 to 21-04-2026' in market_copy.columns:
            market_copy['Price'] = pd.to_numeric(
                market_copy['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''), errors='coerce'
            )
        
        # Filter for the crop
        crop_market = market_copy[market_copy['Commodity'] == matched_market_crop].copy()
        
        if not crop_market.empty:
            # Convert Price to Rs/Tonne (1 Quintal = 0.1 Tonnes, so multiply by 10)
            crop_market['Price_Tonne'] = crop_market['Price'] * 10
            
            prices = crop_market['Price_Tonne'].dropna()
            if not prices.empty:
                analytics['price']['avg'] = f"{prices.mean():,.2f}"
                analytics['price']['max'] = f"{prices.max():,.2f}"
                analytics['price']['min'] = f"{prices.min():,.2f}"
                
            # Handle Month dates for Last 30 Days and Charts
            if 'Month' in crop_market.columns:
                import re
                def parse_month(d):
                    if pd.isna(d): return pd.NaT
                    d_str = str(d).strip()
                    
                    # Try common formats first
                    for fmt in ['%B-%Y', '%b-%Y', '%m-%Y', '%B/%Y', '%b/%Y', '%m/%Y', '%d-%m-%Y', '%Y-%m-%d']:
                        try:
                            return pd.to_datetime(d_str, format=fmt)
                        except:
                            continue
                            
                    # Handle "Month-YY" manually (e.g., Feb-21)
                    parts = re.split(r'[-/]', d_str)
                    if len(parts) == 2:
                        m_str, y_str = parts[0], parts[1]
                        if len(y_str) == 2:
                            y_str = f"20{y_str}"
                        # Try to parse with expanded year
                        new_str = f"{m_str}-{y_str}"
                        for fmt in ['%B-%Y', '%b-%Y', '%m-%Y']:
                            try:
                                return pd.to_datetime(new_str, format=fmt)
                            except:
                                continue
                    
                    return pd.to_datetime(d_str, errors='coerce')
                            
                crop_market['Month_Date'] = crop_market['Month'].apply(parse_month)
                valid_dates = crop_market.dropna(subset=['Month_Date']).sort_values('Month_Date')
                
                if not valid_dates.empty:
                    # Latest Month Data
                    latest_date = valid_dates['Month_Date'].max()
                    analytics['latest_month'] = latest_date.strftime('%B %Y')
                    
                    latest_records = valid_dates[valid_dates['Month_Date'] == latest_date]
                    
                    recent_list = []
                    for _, row in latest_records.iterrows():
                        arr_str = str(row.get('Arrival Quantity 21-01-2021 to 21-04-2026', '0')).replace(',', '')
                        arr_val = pd.to_numeric(arr_str, errors='coerce')
                        
                        recent_list.append({
                            'district': row.get('District', 'Unknown'),
                            'arrival': f"{arr_val:.2f}" if pd.notna(arr_val) else "0",
                            'price': f"{row.get('Price_Tonne', 0):,.2f}"
                        })
                    analytics['recent_data'] = recent_list
                    
                    # Chart Data (monthly averages)
                    monthly_avg = valid_dates.groupby('Month_Date')['Price_Tonne'].mean().reset_index()
                    monthly_avg = monthly_avg.sort_values('Month_Date')
                    
                    analytics['chart_data']['labels'] = monthly_avg['Month_Date'].dt.strftime('%b %Y').tolist()
                    analytics['chart_data']['prices'] = monthly_avg['Price_Tonne'].round(2).tolist()

                    
    return analytics
