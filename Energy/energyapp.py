import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest
import datetime as dt

# Device normal ranges 
DEVICE_NORMAL_RANGES = {
    'washing_machine': {'min_power': 500, 'max_power': 2500, 'typical_energy': 1.5},
    'refrigerator': {'min_power': 150, 'max_power': 800, 'typical_energy': 0.3},
    'water_heater': {'min_power': 1500, 'max_power': 4500, 'typical_energy': 3.0},
    'ac': {'min_power': 800, 'max_power': 4000, 'typical_energy': 2.5},
    'tubelight': {'min_power': 15, 'max_power': 60, 'typical_energy': 0.04},
    'fan': {'min_power': 50, 'max_power': 120, 'typical_energy': 0.08},
    'led_light': {'min_power': 5, 'max_power': 25, 'typical_energy': 0.015},
    'tv': {'min_power': 80, 'max_power': 300, 'typical_energy': 0.15},
    'laptop': {'min_power': 30, 'max_power': 150, 'typical_energy': 0.08}
}

# Load model and scaler
try:
    model = load_model("lstm_energy_model.h5", custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load("scaler.pkl")
    iso_forest = joblib.load("isoforest.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    model, scaler, iso_forest = None, None, None

# Device analysis
def analyze_device_consumption(device_name, device_data, normal_ranges):
    """Analyze if a specific device is causing high consumption"""
    voltage = device_data.get('voltage', 0)
    current = device_data.get('current', 0)
    energy_wh = device_data.get('energy_wh', 0)
    count = device_data.get('count', 0)
    
    # Calculate current power
    current_power = voltage * current
    
    # Get normal ranges for this device
    device_normal = normal_ranges.get(device_name, {})
    min_power = device_normal.get('min_power', 0)
    max_power = device_normal.get('max_power', float('inf'))
    typical_energy = device_normal.get('typical_energy', 0)
    
    # Analysis results
    analysis = {
        'device_name': device_name.replace('_', ' ').title(),
        'current_power': current_power,
        'current_energy': energy_wh,
        'is_active': count > 0,
        'status': 'normal',
        'is_culprit': False,
        'concerns': [],
        'recommendations': []
    }
    
    # Check if device is a culprit
    if current_power > max_power * 1.2:  # 20% above normal maximum
        analysis['status'] = 'high_consumption'
        analysis['is_culprit'] = True
        analysis['concerns'].append(f"Power {current_power:.0f}W is {((current_power/max_power-1)*100):.0f}% above normal max ({max_power}W)")
        analysis['recommendations'].append(f"Check {device_name} for efficiency issues")
    
    elif current_power > max_power:
        analysis['status'] = 'above_normal'
        analysis['is_culprit'] = True
        analysis['concerns'].append(f"Power {current_power:.0f}W exceeds normal range ({min_power}-{max_power}W)")
        analysis['recommendations'].append(f"Monitor {device_name} usage")
    
    elif energy_wh > typical_energy * 2:
        analysis['status'] = 'high_energy'
        analysis['is_culprit'] = True
        analysis['concerns'].append(f"Energy {energy_wh:.3f}Wh is unusually high (typical: {typical_energy:.3f}Wh)")
        analysis['recommendations'].append(f"Reduce {device_name} usage")
    
    # Calculate cost impact (6 INR per kWh)
    daily_cost_impact = (current_power / 1000) * 24 * 6
    analysis['daily_cost_impact'] = daily_cost_impact
    
    return analysis

# Flatten input data function
def flatten_data(raw_data):
    rows = []
    for entry in raw_data:
        flat_row = {}
        flat_row['timestamp'] = entry['timestamp']
        for device in entry['devices']:
            name = device['device']
            flat_row[f'{name}_count'] = device['count']
            flat_row[f'{name}_voltage'] = device['voltage']
            flat_row[f'{name}_current'] = device['current']
            flat_row[f'{name}_energy_wh'] = device['energy_wh']
        flat_row['aggregate_energy_wh'] = entry.get('aggregate_energy_wh', 0.0)
        rows.append(flat_row)
    return pd.DataFrame(rows)

# LSTM prediction logic 
def predict_energy_lstm(new_raw_data, model, scaler):
    df_new = flatten_data(new_raw_data)
    df_new = df_new.sort_values(by='timestamp').reset_index(drop=True)
    df_new['datetime'] = pd.to_datetime(df_new['timestamp'], unit='s')
    df_new['hour'] = df_new['datetime'].dt.hour
    df_new['minute'] = df_new['datetime'].dt.minute
    df_new['second'] = df_new['datetime'].dt.second
    df_new.drop(columns=['timestamp', 'datetime'], inplace=True)
    if 'aggregate_energy_wh' not in df_new.columns:
        df_new['aggregate_energy_wh'] = 0.0
    scaled_new = scaler.transform(df_new)
    X_input = np.expand_dims(scaled_new[-5:], axis=0)
    y_scaled_pred = model.predict(X_input)[0][0]
    dummy_row = scaled_new[-1].copy()
    dummy_row[df_new.columns.get_loc('aggregate_energy_wh')] = y_scaled_pred
    predicted = scaler.inverse_transform([dummy_row])[0][df_new.columns.get_loc('aggregate_energy_wh')]
    return predicted

# MAIN ANALYSIS FUNCTION
def comprehensive_device_analysis(input_devices, model, scaler):
    """Main function that identifies culprit devices"""
    
    # Create raw data structure
    now = datetime.now().timestamp()
    new_raw_data = [{
        "timestamp": now,
        "devices": input_devices,
        "aggregate_energy_wh": 0.0
    }]
    
    # Get aggregate prediction
    aggregate_prediction = predict_energy_lstm(new_raw_data, model, scaler)
    
    # Analyze each device
    device_analyses = {}
    culprit_devices = []
    total_cost_impact = 0
    
    for device_info in input_devices:
        device_name = device_info['device']
        analysis = analyze_device_consumption(device_name, device_info, DEVICE_NORMAL_RANGES)
        device_analyses[device_name] = analysis
        
        if analysis['is_culprit']:
            culprit_devices.append(device_name)
            total_cost_impact += analysis['daily_cost_impact']
    
    return {
        'aggregate_prediction': aggregate_prediction,
        'device_analyses': device_analyses,
        'culprit_devices': culprit_devices,
        'total_cost_impact': total_cost_impact,
        'main_culprit': culprit_devices[0] if culprit_devices else None
    }

# Streamlit App UI
st.set_page_config(page_title="UtilityEnergyX - Device Analysis", layout="wide")
st.title("🔌 UtilityEnergy: Device-Wise Energy Analysis")
st.markdown("**Find out which specific device is causing your high electricity bill!**")

# List of devices
devices = ["washing_machine", "refrigerator", "water_heater", "ac", "tubelight", "fan", "led_light", "tv", "laptop"]
input_devices = []

# Input layout
col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
col1.markdown("**Device**")
col2.markdown("**Count**")
col3.markdown("**Voltage**")
col4.markdown("**Current**")
col5.markdown("**Energy (Wh)**")

# Inputs for each device
for dev in devices:
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
    col1.markdown(dev.replace("_", " ").title())
    count = col2.number_input(f"{dev}_count", min_value=0, value=1, step=1, format="%d", label_visibility="collapsed")
    voltage = col3.number_input(f"{dev}_voltage", value=220.0, step=0.1, format="%.1f", label_visibility="collapsed")
    current = col4.number_input(f"{dev}_current", value=1.0, step=0.01, format="%.2f", label_visibility="collapsed")
    energy = col5.number_input(f"{dev}_energy_wh", value=0.05, step=0.001, format="%.4f", label_visibility="collapsed")
    input_devices.append({
        "device": dev,
        "count": count,
        "voltage": voltage,
        "current": current,
        "energy_wh": energy
    })

# Analysis button
st.markdown("---")

if st.button("Analyze Energy Consumption", type="primary"):
    if not model or not scaler:
        st.error("Models not loaded properly!")
    else:
        try:
            # Run comprehensive analysis
            analysis_results = comprehensive_device_analysis(input_devices, model, scaler)
            
            # Display main results
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric("Aggregate Energy Prediction", f"{analysis_results['aggregate_prediction']:.4f} Wh")
            
            with col2:
                st.metric("Problematic Devices", len(analysis_results['culprit_devices']))
            
            with col3:
                st.metric("Daily Excess Cost", f"₹{analysis_results['total_cost_impact']:.2f}")
            
            
            st.markdown("---")
            st.subheader("Which device is causing high consumption?")
            
            if not analysis_results['culprit_devices']:
                st.success("✅ **ALL DEVICES ARE NORMAL** - No specific device is causing excessive consumption.")
            
            elif len(analysis_results['culprit_devices']) == 1:
                main_culprit = analysis_results['culprit_devices'][0]
                culprit_analysis = analysis_results['device_analyses'][main_culprit]
                st.error(f"🚨 **CULPRIT IDENTIFIED: {culprit_analysis['device_name']}**")
                st.write(f"**Reason:** {'; '.join(culprit_analysis['concerns'])}")
                st.write(f"**Action:** {'; '.join(culprit_analysis['recommendations'])}")
                st.write(f"**Cost Impact:** ₹{culprit_analysis['daily_cost_impact']:.2f} per day")
            
            else:
                # Multiple culprits
                main_culprit = max(analysis_results['culprit_devices'], 
                                 key=lambda d: analysis_results['device_analyses'][d]['daily_cost_impact'])
                main_analysis = analysis_results['device_analyses'][main_culprit]
                
                st.error(f"🚨 **PRIMARY CULPRIT: {main_analysis['device_name']}**")
                st.warning(f"**Additional Issues:** {len(analysis_results['culprit_devices'])-1} other devices also problematic")
            
            # Device-wise breakdown
            st.markdown("---")
            st.subheader("Device-Wise Analysis")
            
            # Create a table
            table_data = []
            for device_name, analysis in analysis_results['device_analyses'].items():
                status_emoji = "🚨" if analysis['is_culprit'] else "✅"
                table_data.append({
                    "Device": analysis['device_name'],
                    "Power (W)": f"{analysis['current_power']:.1f}",
                    "Energy (Wh)": f"{analysis['current_energy']:.4f}",
                    "Status": f"{status_emoji} {analysis['status']}",
                    "Daily Cost (₹)": f"{analysis['daily_cost_impact']:.2f}"
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True)
            
            # Detailed concerns and recommendations
            if analysis_results['culprit_devices']:
                st.markdown("---")
                st.subheader("Issues & Recommendations")
                
                for device_name in analysis_results['culprit_devices']:
                    analysis = analysis_results['device_analyses'][device_name]
                    
                    with st.expander(f"🔴 {analysis['device_name']} - Issues & Solutions"):
                        st.write("**Issues Identified:**")
                        for concern in analysis['concerns']:
                            st.write(f"• {concern}")
                        
                        st.write("**Recommendations:**")
                        for rec in analysis['recommendations']:
                            st.write(f"• {rec}")
                        
                        st.write(f"**Financial Impact:** ₹{analysis['daily_cost_impact']:.2f} per day, ₹{analysis['daily_cost_impact']*30:.2f} per month")
            
            # Summary for management
            st.markdown("---")
            st.subheader("Executive Summary")
            
            if not analysis_results['culprit_devices']:
                st.info("All devices are operating within normal parameters. Energy consumption is optimized.")
            else:
                monthly_excess = analysis_results['total_cost_impact'] * 30
                annual_excess = monthly_excess * 12
                
                summary_text = f"""
                **Energy Analysis Results:**
                - **Problematic Devices:** {len(analysis_results['culprit_devices'])} out of 9 devices
                - **Primary Culprit:** {analysis_results['device_analyses'][analysis_results['main_culprit']]['device_name'] if analysis_results['main_culprit'] else 'None'}
                - **Financial Impact:** ₹{analysis_results['total_cost_impact']:.2f} per day, ₹{monthly_excess:.2f} per month, ₹{annual_excess:.2f} per year
                - **Action Required:** Immediate attention to {len(analysis_results['culprit_devices'])} device(s)
                """
                st.warning(summary_text)
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            
# Footer
st.markdown("---")
st.markdown("*UtilityEnergy helps to identify which specific appliances are causing high electricity bills*")