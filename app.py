from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import traceback
import numpy as np
from datetime import datetime
import re
from urllib.parse import unquote
import analytics_engine

app = Flask(__name__, template_folder='templates', static_folder='static')

# Comprehensive regex-based crop name aliases for standardization
CROP_REGEX_ALIASES = {
    # CEREALS
    r'^(rice|paddy)$': 'Rice',
    r'^(maize|corn)$': 'Maize',
    r'^wheat$': 'Wheat',

    # PULSES
    r'^(arhar(_dal)?|tur|arhar/tur|pigeon\s*pea)$': 'Arhar/Tur',
    r'^(black[_\s-]?gram|urad|urd)$': 'Urad',
    r'^(green[_\s-]?gram|moong|mung|mungbean)$': 'Moong(Green Gram)',
    r'^(masur(_dal)?|masoor|lentil)$': 'Masoor',
    r'^(gram|chickpea|chana)$': 'Gram',
    r'^(cowpea|cowpea\s*\(lobia\)|lobia)$': 'Cowpea(Lobia)',
    r'^(peas\(dry\)|peas\s*&\s*beans\s*\(pulses\)|dry\s*peas|peas)$': 'Peas & beans (Pulses)',
    r'^other\s*kharif\s*pulses$': 'Other Kharif pulses',
    r'^other\s*rabi\s*pulses$': 'Other Rabi pulses',

    # OILSEEDS
    r'^(groundnut|peanut)$': 'Groundnut',
    r'^(mustard|rapeseed\s*&\s*mustard|rapeseed)$': 'Rapeseed &Mustard',
    r'^(sesamum|sesame|til)$': 'Sesamum',
    r'^(soyabean|soybean|soya)$': 'Soyabean',
    r'^other\s*oilseeds$': 'other oilseeds',

    # VEGETABLES
    r'^(bhindi|okra|lady\'?s\s*finger)$': 'Bhindi',
    r'^(brinjal|eggplant|aubergine)$': 'Brinjal',
    r'^cabbage$': 'Cabbage',
    r'^carrot$': 'Carrot',
    r'^cauliflower$': 'Cauliflower',
    r'^(raddish|radish|redish)$': 'Radish',
    r'^tomato$': 'Tomato',
    r'^onion$': 'Onion',
    r'^potato$': 'Potato',
    r'^(sweet[_\s-]?potato)$': 'Sweet potato',
    r'^tapioca$': 'Tapioca',
    r'^(bitter\s*gourd)$': 'Bitter Gourd',
    r'^(bottle\s*gourd)$': 'Bottle Gourd',
    r'^other\s*vegetables$': 'Other Vegetables',

    # FRUITS
    r'^banana$': 'Banana',
    r'^(jackfruit|jack\s*fruit)$': 'Jack Fruit',
    r'^mango$': 'Mango',
    r'^(mousambi|mosambi|sweet\s*lime)$': 'Mousambi',
    r'^orange$': 'Orange',
    r'^(citrus\s*fruit)$': 'Citrus Fruit',
    r'^papaya$': 'Papaya',
    r'^pear$': 'Pear',
    r'^pineapple$': 'Pineapple',
    r'^(pome\s*fruit)$': 'Pome Fruit',
    r'^other\s*fresh\s*fruits$': 'Other Fresh Fruits',

    # SPICES
    r'^(dry\s*chillies|dry\s*chilies|chilli|chillies)$': 'Dry chillies',
    r'^ginger$': 'Ginger',
    r'^turmeric$': 'Turmeric',

    # FIBRE / CASH
    r'^(cotton|kapas|cotton\(lint\)|lint)$': 'Cotton(lint)',
    r'^jute$': 'Jute',
    r'^sugarcane$': 'Sugarcane',
    r'^rubber$': 'Rubber',
    r'^(cashewnut|cashewnuts|cashew)$': 'Cashewnut'
}

def standardize_crop_name(crop_name):
    """
    Standardize crop name using regex patterns
    Returns the standardized name if matched, otherwise returns the original name
    """
    if not crop_name:
        return None
    
    crop_name = str(crop_name).strip()
    search_name = crop_name.lower()
    
    # Try to match against regex patterns
    for pattern, standard_name in CROP_REGEX_ALIASES.items():
        if re.match(pattern, search_name, re.IGNORECASE):
            return standard_name
    
    # If no regex match, return original
    return crop_name

def match_crop_by_regex(crop_name, available_crops):
    """
    Match crop name using regex patterns and available crops
    """
    if not crop_name:
        return None
    
    crop_name = str(crop_name).strip()
    search_name = crop_name.lower()
    
    # First standardize using regex
    standardized = standardize_crop_name(crop_name)
    
    # Clean available crops to strings
    clean_available = [str(c) for c in available_crops if pd.notna(c)]
    
    # Check if standardized name exists in available crops (case-insensitive)
    for crop in clean_available:
        if crop.lower() == standardized.lower():
            return crop
    # If not found, try exact original search in available crops
    for crop in clean_available:
        if crop.lower() == search_name:
            return crop

    # Try substring match: standardized name contained within available crop
    if standardized:
        # Check token presence: ensure all meaningful tokens from standardized
        # name appear in the available crop string (order-insensitive)
        tokens = re.findall(r"[a-z0-9]+", standardized.lower())
        if tokens:
            for crop in clean_available:
                clean_crop = re.sub(r"[^a-z0-9]", "", crop.lower())
                if all(tok in clean_crop for tok in tokens):
                    return crop

    # Try flexible regex pattern matching (use search not anchored match)
    for crop in clean_available:
        crop_lower = crop.lower()
        for pattern, _ in CROP_REGEX_ALIASES.items():
            try:
                if re.search(pattern, crop_lower, re.IGNORECASE) and re.search(pattern, search_name, re.IGNORECASE):
                    return crop
            except re.error:
                # Skip invalid patterns
                continue
    
    return None

# Load model
try:
    brain = pickle.load(open('crop_model.pkl', 'rb'))
    if isinstance(brain, dict):
        model = brain['model']
        le_crop = brain['le_crop']
        le_dist = brain['le_dist']
    else:
        model, le_crop = brain
        le_dist = None
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None
    le_crop = None
    le_dist = None

# Load datasets
try:
    soil = pd.read_csv('data/soil.csv')
    market = pd.read_csv('data/market.csv')
    crop_yield = pd.read_csv('data/crop_yield.csv')
    soil_crop = pd.read_csv('data/soil_crop.csv')
    print(f"✓ Data loaded: soil={soil.shape}, market={market.shape}, soil_crop={soil_crop.shape}")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    soil = None
    market = None
    crop_yield = None
    soil_crop = None


def extract_crop_yields(crop_name, district=None):
    """Extract yield data for a crop from crop_yield CSV using regex matching"""
    try:
        if crop_yield is None:
            return None
        
        # Mapping from market crop names to yield crop names (using regex patterns)
        crop_name_mapping = {
            # Fruits
            r'jack.*fruit': 'Jack Fruit',
            r'mousambi': 'Mousambi',
            r'sweet.?lime': 'Mousambi',
            r'pear.*marasebu': 'Pear',
            r'banana': 'Banana',
            r'mango': 'Mango',
            r'orange': 'Orange',
            r'papaya': 'Papaya',
            r'pineapple': 'Pineapple',
            r'citrus': 'Citrus Fruit',
            r'pome': 'Pome Fruit',
            
            # Vegetables
            r'bhindi': 'Bhindi',
            r'ladies.?finger': 'Bhindi',
            r'brinjal': 'Brinjal',
            r'eggplant': 'Brinjal',
            r'cabbage': 'Cabbage',
            r'carrot': 'Carrot',
            r'cauliflower': 'Cauliflower',
            r'raddish': 'Redish',
            r'radish': 'Redish',
            r'tomato': 'Tomato',
            r'onion': 'Onion',
            r'potato': 'Potato',
            r'sweet.?potato': 'Sweet potato',
            r'tapioca': 'Tapioca',
            r'bitter.?gourd': 'Bitter Gourd',
            r'bottle.?gourd': 'Bottle Gourd',
            r'beans': 'Peas & beans (Pulses)',
            
            # Pulses
            r'arhar.*dal': 'Arhar/Tur',
            r'tur.*dal': 'Arhar/Tur',
            r'black.?gram.*dal': 'Urad',
            r'urd.*dal': 'Urad',
            r'green.?gram': 'Moong(Green Gram)',
            r'moong': 'Moong(Green Gram)',
            r'masur.*dal': 'Masoor',
            r'cowpea': 'Cowpea(Lobia)',
            r'lobia': 'Cowpea(Lobia)',
            r'peas.*dry': 'Peas & beans (Pulses)',
            
            # Cereals
            r'rice': 'Rice',
            r'wheat': 'Wheat',
            r'maize': 'Maize',
            r'corn': 'Maize',
            
            # Oilseeds
            r'groundnut': 'Groundnut',
            r'mustard': 'Rapeseed &Mustard',
            r'sesamum': 'Sesamum',
            r'sesame': 'Sesamum',
            r'til': 'Sesamum',
            r'soyabean': 'Soyabean',
            r'soybean': 'Soyabean',
            
            # Spices
            r'dry.?chillies': 'Dry chillies',
            r'ginger': 'Ginger',
            r'turmeric': 'Turmeric',
            
            # Others
            r'sugarcane': 'Sugarcane',
            r'cotton': 'Cotton(lint)',
            r'lint': 'Cotton(lint)',
            r'kapas': 'Kapas',
            r'jute': 'Jute',
            r'rubber': 'Rubber',
            r'cashewnut': 'Cashewnut'
        }
        
        # Find matching yield crop name using regex
        matched_yield_name = None
        crop_lower = crop_name.lower()
        
        for pattern, yield_name in crop_name_mapping.items():
            if re.search(pattern, crop_lower, re.IGNORECASE):
                matched_yield_name = yield_name
                break
        
        # If no mapping found, try direct match
        if not matched_yield_name:
            # Find columns matching the crop name directly
            crop_cols = [col for col in crop_yield.columns if col.startswith(crop_name)]
            if crop_cols:
                matched_yield_name = crop_name
        
        if not matched_yield_name:
            return {'avg_yield': 0, 'max_yield': 0, 'records': 0}
        
        local_crop_yield = crop_yield.copy()
        if district:
            filtered_yield = local_crop_yield[local_crop_yield['District'].str.contains(district, case=False, na=False)]
            if not filtered_yield.empty:
                local_crop_yield = filtered_yield
                
        # Find columns matching the yield crop name
        crop_cols = [col for col in local_crop_yield.columns if col.startswith(matched_yield_name)]
        
        if not crop_cols:
            return {'avg_yield': 0, 'max_yield': 0, 'records': 0}
        
        # Get the yield column (usually the one ending in .2 or .3 pattern for Yield)
        # The pattern is: Crop, Crop.1 (Season), Crop.2 (Data Type), Crop.3 (Year data)
        # We want the yield data which is typically every 3rd column starting from index 2
        yield_values = []
        
        for col in crop_cols:
            # Extract numeric values directly; pd.to_numeric coerces headers (Kharif, Yield, etc.) to NaN
            data = pd.to_numeric(local_crop_yield[col], errors='coerce').dropna()
            yield_values.extend(data.tolist())
        
        if not yield_values:
            return {'avg_yield': 0, 'max_yield': 0, 'records': 0}
        
        avg_yield = float(np.mean(yield_values))
        max_yield = float(np.max(yield_values))
        std_yield = float(np.std(yield_values)) if len(yield_values) > 1 else 1.0
        
        past_yield = 0
        curr_yield = 0
        if len(yield_values) >= 2:
            split_idx = len(yield_values) // 2
            past_yield = float(np.mean(yield_values[:split_idx]))
            curr_yield = float(np.mean(yield_values[split_idx:]))
        else:
            past_yield = avg_yield
            curr_yield = avg_yield
        
        return {
            'avg_yield': avg_yield,
            'max_yield': max_yield,
            'std_yield': std_yield,
            'past_yield': past_yield,
            'curr_yield': curr_yield,
            'records': len(yield_values)
        }
    except Exception as e:
        print(f"Error extracting yield for {crop_name}: {e}")
        return {'avg_yield': 0, 'max_yield': 0, 'std_yield': 1.0, 'past_yield': 0, 'curr_yield': 0, 'records': 0}

@app.route('/')
def index():
    try:
        districts = sorted(soil['District'].unique().tolist())
        crops = sorted(market['Commodity'].unique().tolist())
        print(f"✓ Rendering index with {len(districts)} districts and {len(crops)} crops")
        return render_template('index.html', districts=districts, crops=crops)
    except Exception as e:
        print(f"✗ Error rendering index: {e}")
        traceback.print_exc()
        return f"Error: {e}", 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        print(f"Received recommendation request: {data}")
        
        district = data.get('district', '').strip()
        pH = float(data.get('pH', 6.0))
        N = float(data.get('N', 300))
        P = float(data.get('P', 50))
        K = float(data.get('K', 200))
        
        all_crops = [
            "Arhar_Dal", "Banana", "Beans", "Bhindi", "Black_Gram", "Brinjal", 
            "Cabbage", "Carrot", "Cashewnuts", "Cauliflower", "Cotton", "Cowpea", 
            "Dry Chillies", "Ginger", "Green_Gram", "Groundnut", "Jackfruit", 
            "Jute", "Lint", "Maize", "Mango", "Masur_Dal", "Mousambi", "Mustard", 
            "Onion", "Orange", "Papaya", "Pear", "Peas(Dry)", "Pineapple", "Potato", 
            "Raddish", "Rice", "Rubber", "Sesamum", "Soyabean", "Sugarcane", 
            "Sweet_Potato", "Tapioca", "Tomato", "Wheat"
        ]
        
        market_copy = market.copy()
        market_copy['Price'] = pd.to_numeric(
            market_copy['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''), errors='coerce'
        )
        market_copy['Arrival'] = pd.to_numeric(
            market_copy['Arrival Quantity 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''), errors='coerce'
        )
        
        # 1. Pre-calculate metrics for all crops to find global maximums
        crop_metrics = {}
        max_profitability = 0.1
        global_max_yield = 0.1
        
        for crop_name in all_crops:
            matched_market = match_crop_by_regex(crop_name, market['Commodity'].unique())
            avg_price, total_arrival, price_std = 0, 0, 0
            p_mom, a_mom = 0, 0
            has_market_data = False
            
            if matched_market:
                if district:
                    crop_market_data = market_copy[(market_copy['Commodity'] == matched_market) & (market_copy['District'].str.contains(district, case=False, na=False))].copy()
                else:
                    crop_market_data = market_copy[market_copy['Commodity'] == matched_market].copy()
                
                # Sort by Month if possible to split past/current for Momentum
                crop_market_data['Month_parsed'] = pd.to_datetime(crop_market_data['Month'], format='%B-%Y', errors='coerce')
                crop_market_data = crop_market_data.sort_values('Month_parsed')
                
                prices = crop_market_data['Price'].dropna()
                arrivals = crop_market_data['Arrival'].dropna()
                
                if not prices.empty:
                    has_market_data = True
                    avg_price = float(prices.mean())
                    price_std = float(prices.std()) if len(prices) > 1 else 0
                    total_arrival = float(arrivals.sum()) if not arrivals.empty else 0
                    
                    # Momentum
                    if len(prices) >= 4:
                        split_idx = int(len(prices) * 0.7)
                        past_p = prices.iloc[:split_idx].mean()
                        curr_p = prices.iloc[split_idx:].mean()
                        if past_p > 0: p_mom = ((curr_p - past_p) / past_p) * 100
                        
                        if len(arrivals) >= 4:
                            past_a = arrivals.iloc[:split_idx].mean()
                            curr_a = arrivals.iloc[split_idx:].mean()
                            if past_a > 0: a_mom = ((curr_a - past_a) / past_a) * 100
            
            # Yield Data
            yield_info = extract_crop_yields(crop_name, district)
            avg_yield = yield_info.get('avg_yield', 0) if yield_info else 0
            if avg_yield <= 0: avg_yield = 2.0
            yield_std = yield_info.get('std_yield', 1.0) if yield_info else 1.0
            if yield_std <= 0: yield_std = 0.1
            
            past_yield = yield_info.get('past_yield', avg_yield) if yield_info else avg_yield
            curr_yield = yield_info.get('curr_yield', avg_yield) if yield_info else avg_yield
            
            # Profitability and Yield Reliability raw values
            profitability_val = avg_price * (total_arrival / 100) # scale arrival down to prevent overflow
            
            max_profitability = max(max_profitability, profitability_val)
            global_max_yield = max(global_max_yield, avg_yield)
            
            crop_metrics[crop_name] = {
                'avg_price': avg_price,
                'price_std': price_std,
                'total_arrival': total_arrival,
                'p_mom': p_mom,
                'a_mom': a_mom,
                'avg_yield': avg_yield,
                'yield_std': yield_std,
                'past_yield': past_yield,
                'curr_yield': curr_yield,
                'profitability_val': profitability_val,
                'records': len(prices) if matched_market else 0,
                'yield_records': yield_info.get('records', 0) if yield_info else 0,
                'has_market_data': has_market_data
            }
            
        recommendations = []
        for crop_name in all_crops:
            metrics = crop_metrics[crop_name]
            
            # --- 1. MARKET SCORE (40%) ---
            profitability = (metrics['profitability_val'] / max_profitability) * 100
            
            volatility_pct = (metrics['price_std'] / metrics['avg_price']) * 100 if metrics['avg_price'] > 0 else 50
            price_stability = max(0, 100 - min(volatility_pct, 100))
            
            raw_momentum = 0.6 * metrics['p_mom'] + 0.4 * metrics['a_mom']
            demand_momentum = max(0, min(100, 50 + raw_momentum))
            
            market_score = 0.40 * profitability + 0.30 * price_stability + 0.30 * demand_momentum
            
            # --- 2. YIELD SCORE (40%) ---
            yield_productivity = (metrics['avg_yield'] / global_max_yield) * 100
            
            yield_variability_pct = (metrics['yield_std'] / metrics['avg_yield']) * 100 if metrics['avg_yield'] > 0 else 50
            yield_stability = max(0, 100 - min(yield_variability_pct, 100))
            
            raw_yield_growth = ((metrics['curr_yield'] - metrics['past_yield']) / metrics['past_yield'] * 100) if metrics['past_yield'] > 0 else 0
            yield_growth = max(0, min(100, 50 + raw_yield_growth))
            
            yield_score = 0.50 * yield_productivity + 0.30 * yield_stability + 0.20 * yield_growth
            
            # --- 3. DUAL SOIL COMPATIBILITY SCORE (15%) ---
            # A. District Soil Suitability
            district_soil_score = 50.0
            has_soil_data = False
            if soil_crop is not None:
                matched_sc = match_crop_by_regex(crop_name, soil_crop['Crop_Name'].unique())
                if matched_sc:
                    has_soil_data = True
                    sc_data = soil_crop[soil_crop['Crop_Name'] == matched_sc].iloc[0]
                    ph_min, ph_max = float(sc_data['Ideal_pH_Min']), float(sc_data['Ideal_pH_Max'])
                    
                    # pH Match (40%)
                    ph_match = 100
                    if pH < ph_min: ph_match = max(0, 100 - ((ph_min - pH) / ph_min * 200))
                    elif pH > ph_max: ph_match = max(0, 100 - ((pH - ph_max) / ph_max * 200))
                    
                    # Nutrient Fit (60%)
                    req_n = float(sc_data['N_Req_kg_ha']) if pd.notna(sc_data['N_Req_kg_ha']) else 100.0
                    req_p = float(sc_data['P_Req_kg_ha']) if pd.notna(sc_data['P_Req_kg_ha']) else 50.0
                    req_k = float(sc_data['K_Req_kg_ha']) if pd.notna(sc_data['K_Req_kg_ha']) else 50.0
                    
                    n_dev = min(1.0, abs(N - req_n) / req_n)
                    p_dev = min(1.0, abs(P - req_p) / req_p)
                    k_dev = min(1.0, abs(K - req_k) / req_k)
                    nutrient_fit = 100 - ((n_dev + p_dev + k_dev) / 3 * 100)
                    
                    district_soil_score = 0.40 * ph_match + 0.60 * nutrient_fit
            
            # B. Crop-Soil Compatibility (Historical Suitability)
            crop_soil_score = 40.0 # Marginal default
            if metrics['total_arrival'] > 1000:
                crop_soil_score = 100.0 # Highly suitable
            elif metrics['total_arrival'] > 100:
                crop_soil_score = 70.0 # Moderately suitable
            elif not metrics['has_market_data']:
                crop_soil_score = 0.0 # Unsuitable if absolutely zero historical trace in the market
                
            # --- 4. CONFIDENCE SCORE (5%) ---
            data_completeness = 100 if (metrics['has_market_data'] and metrics['yield_records'] > 0 and has_soil_data) else \
                               66 if (metrics['has_market_data'] and metrics['yield_records'] > 0) else \
                               33 if (metrics['has_market_data'] or metrics['yield_records'] > 0) else 0
            
            market_reliability = 100 if metrics['records'] > 5 else (metrics['records'] / 5 * 100)
            yield_reliability_conf = 100 if metrics['yield_records'] > 2 else (metrics['yield_records'] / 2 * 100)
            
            confidence_score = 0.50 * data_completeness + 0.30 * market_reliability + 0.20 * yield_reliability_conf
            
            # --- FINAL RECOMMENDATION SCORE ---
            # 35% Market | 35% Yield | 15% District Soil | 10% Crop Soil | 5% Confidence
            final_score = 0.35 * market_score + 0.35 * yield_score + 0.15 * district_soil_score + 0.10 * crop_soil_score + 0.05 * confidence_score
            
            # Recommendation & Level mapping
            if final_score >= 85:
                level = "Extremely High"
                rec_text = "Strong Buy Crop"
            elif final_score >= 70:
                level = "High"
                rec_text = "Strong Recommendation"
            elif final_score >= 55:
                level = "Moderately Strong"
                rec_text = "Solid Choice"
            elif final_score >= 40:
                level = "Moderate"
                rec_text = "Viable Option"
            else:
                level = "Low"
                rec_text = "Avoid Recommendation"
                
            # Formulate Reasoning String
            reason_parts = []
            if profitability > 70: reason_parts.append("Strong market profitability")
            if price_stability > 70: reason_parts.append("Stable market prices")
            if demand_momentum > 60: reason_parts.append("Positive demand momentum")
            if yield_stability > 70: reason_parts.append("High yield stability")
            if district_soil_score > 70: reason_parts.append("District soil suitable")
            if crop_soil_score > 70: reason_parts.append("Crop-soil compatibility high")
            
            reason = " | ".join(reason_parts) if reason_parts else "Moderate metrics across board"
                
            recommendations.append({
                'crop': crop_name,
                'confidence': round(final_score, 1),
                'level': level,
                'recommendation_text': rec_text,
                'reason': reason,
                'raw_confidence_metric': round(confidence_score, 1) # Sending to frontend if needed
            })
            
        # Rank all crops
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        top_recs = recommendations
        
        print(f"✓ Top recommendations: {[r['crop'] for r in top_recs[:5]]} ...")
        return jsonify({
            'status': 'success',
            'input': {
                'district': district,
                'pH': pH,
                'N': N,
                'P': P,
                'K': K
            },
            'recommendations': top_recs
        })
    except Exception as e:
        print(f"✗ Error in recommendations: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/crop_analytics/<crop_name>', methods=['GET'])
def api_crop_analytics(crop_name):
    try:
        crop_name_clean = unquote(crop_name).strip()
        print(f"Getting analytics for crop: {crop_name_clean}")
        
        # Get yield info
        yield_info = extract_crop_yields(crop_name_clean)
        
        # Match market crop
        available_crops = market['Commodity'].unique() if market is not None else []
        matched_crop = match_crop_by_regex(crop_name_clean, available_crops)
        
        # Generate analytics using the engine
        analytics_data = analytics_engine.generate_crop_analytics(
            crop_name=crop_name_clean,
            matched_market_crop=matched_crop,
            yield_info=yield_info,
            market_df=market
        )
        
        return jsonify({'status': 'success', 'data': analytics_data})
    except Exception as e:
        print(f"✗ Error in crop analytics API: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/crop_details_extended', methods=['POST'])
def crop_details_extended():
    """
    Get extended crop details including:
    - Market information (districts, markets, prices)
    - Yield statistics by district
    - Historical price trends
    """
    try:
        crop_name = request.json.get('crop', '').strip()
        print(f"Getting extended details for crop: {crop_name}")
        
        # Use regex matching to find crop
        available_crops = market['Commodity'].unique()
        matched_crop = match_crop_by_regex(crop_name, available_crops)
        
        if not matched_crop:
            return jsonify({'status': 'error', 'error': f'No data found for {crop_name}'}), 404
        
        # ============ MARKET DATA ============
        crop_market = market[market['Commodity'] == matched_crop]
        
        if crop_market.empty:
            return jsonify({'status': 'error', 'error': f'No market data for {crop_name}'}), 404
        
        # Extract numeric price
        crop_market_copy = crop_market.copy()
        crop_market_copy['Price'] = pd.to_numeric(
            crop_market_copy['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        crop_market_copy['Arrival'] = pd.to_numeric(
            crop_market_copy['Arrival Quantity 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        
        avg_price = float(crop_market_copy['Price'].mean())
        price_range = [float(crop_market_copy['Price'].min()), float(crop_market_copy['Price'].max())]
        
        # Group by district and market
        markets_by_district = {}
        for _, row in crop_market_copy.iterrows():
            district = str(row['District']).strip() if pd.notna(row['District']) else 'Unknown'
            market_name = str(row['Market']).strip() if pd.notna(row['Market']) else 'Unknown'
            price = row['Price'] if pd.notna(row['Price']) else 0
            arrival = row['Arrival'] if pd.notna(row['Arrival']) else 0
            
            if district not in markets_by_district:
                markets_by_district[district] = []
            
            # Check if market already exists for this district
            existing = next((m for m in markets_by_district[district] if m['market'] == market_name), None)
            if existing:
                existing['avg_price'] = (existing['avg_price'] + float(price)) / 2
                existing['records'] += 1
            else:
                markets_by_district[district].append({
                    'market': market_name,
                    'avg_price': float(price),
                    'arrival': float(arrival),
                    'records': 1
                })
        
        # ============ YIELD DATA BY DISTRICT ============
        yield_by_district = []
        
        if crop_yield is not None:
            # Find columns for this crop
            crop_cols = [col for col in crop_yield.columns if col.startswith(matched_crop)]
            
            if crop_cols:
                # Get unique districts from yield data
                yield_districts = crop_yield['District'].dropna().unique()
                
                for dist in yield_districts:
                    dist_data = crop_yield[crop_yield['District'] == dist]
                    
                    # Get yield values (every 3rd column starting from index 2)
                    yield_values = []
                    production_values = []
                    area_values = []
                    
                    for i, col in enumerate(crop_cols):
                        if i % 3 == 2:  # Yield column
                            data = pd.to_numeric(dist_data[col].iloc[3:], errors='coerce').dropna()
                            yield_values.extend(data.tolist())
                        elif i % 3 == 1:  # Production column
                            data = pd.to_numeric(dist_data[col].iloc[3:], errors='coerce').dropna()
                            production_values.extend(data.tolist())
                        elif i % 3 == 0:  # Area column
                            data = pd.to_numeric(dist_data[col].iloc[3:], errors='coerce').dropna()
                            area_values.extend(data.tolist())
                    
                    if yield_values:
                        yield_by_district.append({
                            'district': str(dist).strip(),
                            'avg_yield': round(float(np.mean(yield_values)), 2),
                            'max_yield': round(float(np.max(yield_values)), 2),
                            'min_yield': round(float(np.min(yield_values)), 2),
                            'total_production': round(float(np.sum(production_values)), 2) if production_values else 0,
                            'total_area': round(float(np.sum(area_values)), 2) if area_values else 0,
                            'records': len(yield_values)
                        })
        
        # Sort by average yield
        yield_by_district.sort(key=lambda x: x['avg_yield'], reverse=True)
        
        # ============ HISTORICAL PRICE TREND ============
        price_history = []
        months_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                     'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        
        for _, row in crop_market_copy.iterrows():
            month_str = str(row['Month']).strip() if pd.notna(row['Month']) else ''
            price = row['Price']
            district = str(row['District']).strip() if pd.notna(row['District']) else ''
            market_name = str(row['Market']).strip() if pd.notna(row['Market']) else ''
            
            if pd.notna(price) and price > 0 and month_str:
                try:
                    parts = re.split(r'[-/]', month_str)
                    if len(parts) >= 2:
                        month_name = parts[0].strip()
                        year = int(parts[-1].strip())
                        month_num = months_map.get(month_name, 1)
                        
                        price_history.append({
                            'date': f"{year}-{month_num:02d}",
                            'year': year,
                            'month': month_name,
                            'price': float(price),
                            'district': district,
                            'market': market_name
                        })
                except:
                    pass
        
        # Sort by date correctly using a helper or the date string
        def sort_key(x):
            try:
                m_num = months_map.get(x['month'], 1)
                return (x['year'], m_num)
            except:
                return (0, 0)
        
        price_history.sort(key=sort_key)

        
        # Aggregate by month for chart
        monthly_prices = {}
        for ph in price_history:
            key = ph['date']
            if key not in monthly_prices:
                monthly_prices[key] = []
            monthly_prices[key].append(ph['price'])
        
        chart_data = []
        for date, prices in monthly_prices.items():
            chart_data.append({
                'date': date,
                'price': round(float(np.mean(prices)), 2),
                'min': round(float(np.min(prices)), 2),
                'max': round(float(np.max(prices)), 2)
            })
        
        print(f"✓ Extended details for '{matched_crop}': {len(markets_by_district)} districts, {len(yield_by_district)} yield records, {len(chart_data)} price points")
        
        avg_yield_val = float(np.mean([d['avg_yield'] for d in yield_by_district])) if yield_by_district else 2.0
        total_arrival = float(crop_market_copy['Arrival'].sum()) if not crop_market_copy.empty else 0

        return jsonify({
            'status': 'success',
            'crop': matched_crop,
            'market': {
                'avg_price': float(avg_price) if avg_price is not None else 0.0,
                'price_tonne': float(avg_price * 10) if avg_price is not None else 0.0,
                'price_range': [float(p) for p in price_range],
                'price_std': float(crop_market_copy['Price'].std()) if len(crop_market_copy) > 1 else 0.0,
                'volatility': float(crop_market_copy['Price'].std() / avg_price * 100) if len(crop_market_copy) > 1 and avg_price > 0 else 0.0,
                'districts_markets': markets_by_district,
                'total_records': int(len(crop_market)),
                'total_arrivals': total_arrival
            },
            'yield': {
                'avg': avg_yield_val,
                'max': float(np.max([d['max_yield'] for d in yield_by_district])) if yield_by_district else 0.0,
                'min': float(np.min([d['min_yield'] for d in yield_by_district])) if yield_by_district else 0.0,
                'by_district': yield_by_district[:15],
                'total_production': float(np.sum([d['total_production'] for d in yield_by_district])) if yield_by_district else 0.0,
                'total_districts': int(len(yield_by_district))
            },
            'price_history': {
                'raw': price_history,
                'chart': chart_data
            }
        })
        
    except Exception as e:
        print(f"✗ Error in extended crop details: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/all_crops_analytics', methods=['GET'])
def all_crops_analytics():
    """Get comprehensive analytics for all crops"""
    try:
        crops_data = []
        available_crops = market['Commodity'].unique().tolist()
        
        for crop in available_crops[:50]:
            crop_market = market[market['Commodity'] == crop]
            if crop_market.empty:
                continue
            
            # Market stats
            prices = pd.to_numeric(
                crop_market['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''),
                errors='coerce'
            ).dropna()
            
            if len(prices) == 0:
                continue
            
            avg_price = float(prices.mean())
            max_price = float(prices.max())
            min_price = float(prices.min())
            volatility = float(prices.std())
            
            # Yield data
            yield_data = extract_crop_yields(crop)
            avg_yield = yield_data['avg_yield'] if yield_data else 0
            max_yield = yield_data['max_yield'] if yield_data else 0
            
            # Fallback yield values for crops with no yield data
            if avg_yield == 0:
                crop_lower = crop.lower()
                if 'fruit' in crop_lower:
                    avg_yield = 15.0
                elif 'vegetable' in crop_lower:
                    avg_yield = 10.0
                elif 'pulse' in crop_lower or 'dal' in crop_lower:
                    avg_yield = 2.5
                elif 'cereal' in crop_lower or crop_lower in ['rice', 'wheat', 'maize']:
                    avg_yield = 4.0
                elif 'spice' in crop_lower:
                    avg_yield = 2.0
                elif 'oilseed' in crop_lower:
                    avg_yield = 2.5
                else:
                    avg_yield = 5.0
            
            if max_yield == 0:
                max_yield = avg_yield * 1.5
            
            # Risk level
            if volatility > 1000:
                risk_level = 'High'
            elif volatility > 500:
                risk_level = 'Moderate'
            else:
                risk_level = 'Low'
            
            # Profitability calculations
            # Estimated cost per acre (varies by crop type) - using regex patterns
            
            crops_data.append({
                'crop': crop,
                'market': {
                    'avg_price': round(avg_price, 2),
                    'max_price': round(max_price, 2),
                    'min_price': round(min_price, 2),
                    'volatility': round(volatility, 2),
                    'price_range': round(max_price - min_price, 2),
                    'risk_level': risk_level
                },
                'yield': {
                    'avg_yield': avg_yield,
                    'max_yield': max_yield,
                    'yield_records': yield_data['records'] if yield_data else 0
                }
            })
        
        print(f"✓ Retrieved analytics for {len(crops_data)} crops")
        return jsonify({
            'status': 'success',
            'crops': crops_data,
            'count': len(crops_data)
        })
    except Exception as e:
        print(f"✗ Error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/predictive_analytics', methods=['GET'])
def predictive_analytics():
    """Get 30-day price forecasts for crops"""
    try:
        forecasts = []
        available_crops = market['Commodity'].unique().tolist()
        
        for crop in available_crops[:30]:
            crop_market = market[market['Commodity'] == crop]
            if crop_market.empty:
                continue
            
            prices = pd.to_numeric(
                crop_market['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''),
                errors='coerce'
            ).dropna()
            
            if len(prices) < 5:
                continue
            
            current_price = float(prices.iloc[-1]) if len(prices) > 0 else float(prices.mean())
            avg_price = float(prices.mean())
            price_change = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0
            volatility = float(prices.std())
            
            # Simple forecast: trend + random variation
            trend = 1 + (price_change / 100 / 30)  # daily trend
            forecast_7d = current_price * (trend ** 7) + np.random.normal(0, volatility * 0.01)
            forecast_30d = current_price * (trend ** 30) + np.random.normal(0, volatility * 0.05)
            
            price_increase_prob = 60 if price_change > 0 else 40
            
            forecasts.append({
                'crop': crop,
                'current_price': round(current_price, 2),
                'forecast_7d': round(max(forecast_7d, 0), 2),
                'forecast_30d': round(max(forecast_30d, 0), 2),
                'price_change_7d': round(((forecast_7d - current_price) / current_price * 100), 2),
                'price_change_30d': round(((forecast_30d - current_price) / current_price * 100), 2),
                'price_increase_probability': round(price_increase_prob, 0),
                'trend': '📈 Upward' if price_change > 0 else '📉 Downward',
                'alert': '🔔 Price Spike Alert' if forecast_30d > (avg_price * 1.5) else 'Normal'
            })
        
        return jsonify({
            'status': 'success',
            'data': forecasts,
            'count': len(forecasts),
            'note': 'Forecasts based on historical trends and statistical models'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/seasonal_analytics', methods=['GET'])
def seasonal_analytics():
    """Get seasonal patterns and best times to sow/sell"""
    try:
        seasonal_data = []
        available_crops = market['Commodity'].unique().tolist()
        
        for crop in available_crops[:30]:
            crop_market = market[market['Commodity'] == crop]
            if crop_market.empty:
                continue
            
            prices = pd.to_numeric(
                crop_market['Modal Price 21-01-2021 to 21-04-2026'].astype(str).str.replace(',', ''),
                errors='coerce'
            ).dropna()
            
            if len(prices) < 5:
                continue
            
            avg_price = float(prices.mean())
            max_price = float(prices.max())
            min_price = float(prices.min())
            
            # Best selling season (when price is high)
            # In India: January-April (Peak)
            # June-August (Low)
            # October-December (Medium)
            
            # Determine seasonal recommendation based on price
            if max_price - min_price > avg_price * 0.5:
                best_sell_month = 'January-April'
                best_sow_month = 'July-September'
            else:
                best_sell_month = 'Year-round'
                best_sow_month = 'As per local guidelines'
            
            seasonal_data.append({
                'crop': crop,
                'avg_annual_price': round(avg_price, 2),
                'best_sell_month': best_sell_month,
                'best_sow_month': best_sow_month,
                'seasonal_variation': round((max_price - min_price) / avg_price * 100, 2),
                'peak_price': round(max_price, 2),
                'low_price': round(min_price, 2),
                'has_seasonality': 'Yes' if (max_price - min_price) / avg_price > 0.3 else 'No'
            })
        
        return jsonify({
            'status': 'success',
            'data': seasonal_data,
            'count': len(seasonal_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/debug_match', methods=['POST'])
def debug_match():
    try:
        crop_name = request.json.get('crop', '').strip()
        available = market['Commodity'].unique().tolist()
        matched = match_crop_by_regex(crop_name, available)
        # Also return token-match candidates
        std = standardize_crop_name(crop_name)
        tokens = []
        if std:
            tokens = re.findall(r"[a-z0-9]+", std.lower())
        candidates = []
        for c in available:
            clean = re.sub(r"[^a-z0-9]", "", str(c).lower())
            if tokens and all(tok in clean for tok in tokens):
                candidates.append(c)
        return jsonify({'status':'success','input':crop_name,'standardized':std,'matched':matched,'candidates':candidates})
    except Exception as e:
        return jsonify({'status':'error','error':str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    if model is None or le_crop is None or le_dist is None:
        print("✗ Cannot start app: Model or encoders not loaded properly")
        exit(1)
    
    print("\n" + "="*60)
    print("CROP RECOMMENDATION SYSTEM")
    print("="*60)
    print("✓ Model loaded and ready")
    print("✓ Starting Flask app on http://127.0.0.1:5002")
    print("\nAPI Endpoints:")
    print("  GET  /                - Web interface")
    print("  POST /recommend       - Recommend crops")
    print("  POST /crop_details    - Get crop market details")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5002, use_reloader=False)
