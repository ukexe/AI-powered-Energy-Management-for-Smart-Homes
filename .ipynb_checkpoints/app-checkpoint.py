import pandas as pd
import numpy as np  
from sklearn.linear_model import LinearRegression
from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Function to generate sample data
def generate_sample_data(num_buildings=3):
    days = pd.date_range('1/1/2023', periods=365)
    data = {}
    for n in range(1, num_buildings+1):
        series = pd.Series(np.random.randint(50, 100, 365), index=days) 
        data[f'Building {n}'] = series
    return data

# Function to create linear regression models
def create_models(data):
    models = {}
    for name, series in data.items():
        X = np.arange(len(series)).reshape(-1, 1)  
        y = series.values   
        model = LinearRegression().fit(X, y)
        models[name] = model
    return models

# Function to make predictions
def make_predictions(models, data):
    pred_data = {}
    for name, model in models.items():
        X_pred = np.arange(len(data[name])).reshape(-1, 1)  
        preds = model.predict(X_pred)
        df = pd.DataFrame({'Actual': data[name], 
                           'Predicted': preds,
                           'Projected': preds,
                           'Efficiency': np.random.uniform(0.5, 1.5, 365),  # Example efficiency data
                           'Temperature': np.random.uniform(20, 30, 365)  # Example temperature data
                          })  
        pred_data[name] = df
    return pred_data
    
# Suggestions
suggestions = {
    'Building 1': [
        'Reduce weekday AC use', 
        'Raise weekend temperature'
    ],
    'Building 2': [
        'Install efficient lighting',
        'Upgrade HVAC filters'
    ], 
    'Building 3': [
        'Adjust temperature setpoints',
        'Improve insulation'
    ]  
}

# Function to adjust predictions
def adjust_predictions(models, pred_data, building, weight):
    model = models[building]
    X = np.arange(len(pred_data[building])).reshape(-1, 1)  
    preds = model.predict(X)
    pred_data[building]['Predicted'] = preds * weight
    pred_data[building]['Projected'] = preds * weight

# Function to implement changes
def implement_changes(pred_data, building, percent):
    projected = pred_data[building]['Predicted'] * (1 - percent/100)    
    pred_data[building]['Projected'] = projected

# Function to visualize dashboard
def visualize_dashboard(building, pred_data, visualization_type, weight=1.0, percent=10):
    adjust_predictions(models, pred_data, building, weight)
    implement_changes(pred_data, building, percent)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    
    # Visualization 1: Line plot for Actual, Predicted, and Projected
    cols = ['Actual', 'Predicted', 'Projected']    
    palette = sns.color_palette("husl", len(cols))
    pred_data[building][cols].plot(ax=axes[0, 0], title=f'{building} - Actual, Predicted, Projected', color=palette)
    
    # Visualization 2: Area Chart for Actual and Projected
    pred_data[building][['Actual', 'Projected']].plot.area(ax=axes[0, 1], title=f'{building} - Area Chart (Actual, Projected)', color=palette[1:])
    
    # Visualization 3: Histogram for Actual energy consumption
    pred_data[building]['Actual'].plot.hist(ax=axes[1, 0], bins=20, color='skyblue', edgecolor='black', title=f'{building} - Actual Histogram')
    
    # Additional visualizations based on user selection
    if visualization_type == 'Savings':
        savings = pred_data[building]['Actual'] - pred_data[building]['Predicted']
        savings.plot(ax=axes[1, 1], title=f'{building} - Savings', color=palette[2])
    elif visualization_type == 'Difference':
        difference = pred_data[building]['Actual'] - pred_data[building]['Projected']
        difference.plot(ax=axes[1, 1], title=f'{building} - Difference (Actual - Projected)', color=palette[2])

    # Visualization 4: Radar Chart for Actual, Predicted, Projected, Efficiency, and Temperature
    radar_data = pred_data[building][cols + ['Efficiency', 'Temperature']].mean().values
    radar_categories = cols + ['Efficiency', 'Temperature']
    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_data,
        theta=radar_categories,
        fill='toself'
    ))
    fig_radar.update_layout(title=f'{building} - Radar Chart', polar=dict(radialaxis=dict(visible=True)))
    fig_radar.show()

    plt.tight_layout()
    plt.show()

    savings = pred_data[building]['Actual'].sum()
    print(f'\nPredicted Savings: {savings:.2f} kWh')
    
    if building in suggestions:
        print('\nSuggestions:')
        for s in suggestions[building]:
            print('- ' + s)
    else:   
        print('\nNo suggestions currently available')

# Main Execution
data = generate_sample_data(num_buildings=3)
models = create_models(data)
pred_data = make_predictions(models, data)

interact(visualize_dashboard, 
         building=list(data.keys()),
         pred_data=widgets.fixed(pred_data),
         visualization_type=widgets.Dropdown(
             options=['Savings', 'Difference'],
             value='Savings',
             description='Visualization Type:'
         ),
         weight=widgets.FloatSlider(min=0.5, max=1.5, value=1.0, step=0.1),
         percent=widgets.FloatSlider(min=0, max=30,  step=5, value=10))
