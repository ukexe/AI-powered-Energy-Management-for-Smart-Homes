import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Functions for data generation and modeling

def generate_sample_data(sensor_type, num_buildings=3):
    days = pd.date_range('1/1/2023', periods=365)
    data = {}
    for n in range(1, num_buildings + 1):
        if sensor_type == 'Current Sensors':
            series = pd.Series(np.random.randint(0, 100, 365), index=days)
        elif sensor_type == 'Light Sensors':
            series = pd.Series(np.random.randint(0, 1000, 365), index=days)
        elif sensor_type == 'Temperature Sensors':
            series = pd.Series(np.random.uniform(10, 30, 365), index=days)
        elif sensor_type == 'Motion Sensors':
            series = pd.Series(np.random.randint(0, 2, 365), index=days)
        data[f'Building {n}'] = series
    return data

def create_models(data):
    models = {}
    for name, series in data.items():
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        models[name] = LinearRegression().fit(X, y)
    return models

def make_predictions(models, data):
    pred_data = {}
    for name, model in models.items():
        preds = model.predict(np.arange(len(data[name])).reshape(-1, 1))
        pred_data[name] = pd.DataFrame({'Actual': data[name],
                                        'Predicted': preds,
                                        'Projected': preds,
                                        'Efficiency': np.random.uniform(0.5, 1.5, 365),
                                        'Temperature': np.random.uniform(20, 30, 365)})
    return pred_data

def adjust_predictions(building, weight, pred_data):
    pred_data[building][['Predicted', 'Projected']] *= weight

def implement_changes(building, percent, pred_data):
    pred_data[building]['Projected'] *= (1 - percent / 100)

# Suggestions dictionary
suggestions = {
    'Building 1': ['Reduce AC', 'Upgrade filters'],
    'Building 2': ['Install sensors', 'Improve insulation'],
    'Building 3': ['Adjust temperature setpoints', 'Improve roof insulation', 'Reduce equipment energy use',
                   'Tune-up heating system', 'Upgrade to high-efficiency heating and cooling',
                   'Consider renewable energy options']
}

# Streamlit App

st.title('Energy Dashboard')

sensor_type = st.selectbox('Select sensor type:', ['Current Sensors', 'Light Sensors', 'Temperature Sensors', 'Motion Sensors'])
data = generate_sample_data(sensor_type)
models = create_models(data)
pred_data = make_predictions(models, data)

building = st.selectbox('Select building:', list(data.keys()))

efficiency = st.slider('Efficiency change: ', 0.5, 1.5, 1.0, 0.1)
savings = st.slider('Projected savings (%): ', 0, 30, 10, 5)

adjust_predictions(building, efficiency, pred_data)
implement_changes(building, savings, pred_data)

# Display Suggestions
if building in suggestions:
    st.header('Suggestions:')
    for s in suggestions[building]:
        st.write('- ' + s)

# Display Message
st.write(f'Adjusted {building} predictions based on {efficiency:.2f} efficiency change.')
st.write(f'Projections updated with {savings}% savings.')

# Display Predicted Savings
st.write(f'Predicted Savings: {pred_data[building]["Actual"].sum():.2f} kWh')

# Radar Chart
radar_data = pred_data[building][['Actual', 'Predicted', 'Projected', 'Efficiency', 'Temperature']].mean().values
radar_categories = ['Actual', 'Predicted', 'Projected', 'Efficiency', 'Temperature']
fig_radar = go.Figure(go.Scatterpolar(
    r=radar_data,
    theta=radar_categories,
    fill='toself'
))
fig_radar.update_layout(title=f'{building} - Radar Chart', polar=dict(radialaxis=dict(visible=True)))
st.plotly_chart(fig_radar, use_container_width=True)

# Line Chart
st.line_chart(pred_data[building][['Actual', 'Predicted', 'Projected']])

# Area Chart
st.area_chart(pred_data[building][['Actual', 'Projected']])

# Histogram
hist_data, bin_edges = np.histogram(pred_data[building]['Actual'], bins=20)
hist_df = pd.DataFrame({'Counts': hist_data, 'Bin Edges': bin_edges[:-1]})
st.bar_chart(hist_df.set_index('Bin Edges'))

# Bar Chart
st.bar_chart(pred_data[building][['Actual', 'Predicted']])
