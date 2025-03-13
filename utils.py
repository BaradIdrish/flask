import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def clean_data(df):
    """Clean and prepare the data for analysis."""
    required_columns = ['Year', 'Crop', 'Yield', 'Rainfall']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna()
    df['Year'] = pd.to_numeric(df['Year'])
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
    return df

def generate_plots(df):
    """Generate various plots for agricultural data analysis."""
    plots = {}
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Year', y='Yield')
    plt.title('Crop Yield Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Yield (tons/hectare)')
    plots['yield_trend'] = fig_to_base64()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Rainfall', y='Yield')
    plt.title('Rainfall vs Yield Correlation')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Yield (tons/hectare)')
    plots['rainfall_correlation'] = fig_to_base64()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Crop', y='Yield')
    plt.title('Average Yield by Crop Type')
    plt.xticks(rotation=45)
    plots['crop_comparison'] = fig_to_base64()
    return plots

def predict_yield(df, future_years=5):
    """Predict future crop yields based on historical data."""
    predictions = {}
    for crop in df['Crop'].unique():
        crop_data = df[df['Crop'] == crop].copy()
        X = crop_data[['Year']].values
        y = crop_data['Yield'].values
        model = LinearRegression()
        model.fit(X, y)
        last_year = df['Year'].max()
        future_X = np.array(range(last_year + 1, last_year + future_years + 1)).reshape(-1, 1)
        future_yields = model.predict(future_X)
        predictions[crop] = {
            'years': future_X.flatten().tolist(),
            'yields': future_yields.tolist(),
            'trend': model.coef_[0],
            'confidence': model.score(X, y)
        }
        plt.figure(figsize=(10, 6))
        plt.plot(X, y, 'o-', label='Historical Data')
        plt.plot(future_X, future_yields, 'r--', label='Predictions')
        plt.title(f'Yield Predictions for {crop}')
        plt.xlabel('Year')
        plt.ylabel('Yield (tons/hectare)')
        plt.legend()
        predictions[crop]['plot'] = fig_to_base64()
        plt.close()
    return predictions

def fig_to_base64():
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')