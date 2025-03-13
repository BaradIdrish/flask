import os
import logging
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from utils import clean_data, generate_plots, predict_yield

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Sample data structure
SAMPLE_DATA = {
    'Year': [2020, 2020, 2020, 2020, 2020,
             2021, 2021, 2021, 2021, 2021,
             2022, 2022, 2022, 2022, 2022],
    'Crop': ['Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton',
             'Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton',
             'Wheat', 'Rice', 'Corn', 'Soybeans', 'Cotton'],
    'Yield': [2.6, 3.1, 4.6, 3.3, 2.3,
              3.0, 3.5, 5.0, 3.6, 2.6,
              2.9, 3.3, 4.9, 3.5, 2.5],
    'Rainfall': [750, 1150, 850, 880, 680,
                 1000, 1400, 1100, 950, 750,
                 850, 1250, 1000, 920, 720]
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            # Use sample data if no file is uploaded
            df = pd.DataFrame(SAMPLE_DATA)
        else:
            file = request.files['file']
            if file.filename == '':
                df = pd.DataFrame(SAMPLE_DATA)
            else:
                df = pd.read_csv(file)

        # Clean and process the data
        df = clean_data(df)

        # Generate plots
        plots = generate_plots(df)

        # Generate predictions
        predictions = predict_yield(df)

        # Basic statistics
        stats = {
            'total_records': len(df),
            'avg_yield': df['Yield'].mean().round(2),
            'max_yield': df['Yield'].max(),
            'min_yield': df['Yield'].min(),
            'correlation': df['Rainfall'].corr(df['Yield']).round(2)
        }

        return render_template('analysis.html', 
                            stats=stats,
                            plots=plots,
                            predictions=predictions,
                            tables=[df.head().to_html(classes='table table-striped')])

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        flash(f"Error analyzing data: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/analyze_crop', methods=['POST'])
def analyze_crop():
    try:
        crop_name = request.form.get('crop')
        if not crop_name:
            flash("Please select a crop to analyze", "error")
            return redirect(url_for('index'))

        # Load sample data
        df = pd.DataFrame(SAMPLE_DATA)
        df = clean_data(df)

        # Filter data for selected crop
        crop_df = df[df['Crop'] == crop_name].copy()
        if len(crop_df) == 0:
            flash(f"No data available for {crop_name}", "error")
            return redirect(url_for('index'))

        # Generate plots for specific crop
        plots = generate_plots(crop_df)

        # Get predictions for specific crop
        predictions = predict_yield(crop_df)[crop_name]

        # Calculate statistics for specific crop
        stats = {
            'total_records': len(crop_df),
            'avg_yield': crop_df['Yield'].mean().round(2),
            'max_yield': crop_df['Yield'].max(),
            'min_yield': crop_df['Yield'].min(),
            'correlation': crop_df['Rainfall'].corr(crop_df['Yield']).round(2)
        }

        return render_template('crop_analysis.html',
                             crop_name=crop_name,
                             stats=stats,
                             plots=plots,
                             predictions=predictions,
                             tables=[crop_df.sort_values('Year').to_html(classes='table table-striped')])

    except Exception as e:
        logger.error(f"Error during crop analysis: {str(e)}")
        flash(f"Error analyzing crop data: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    try:
        # Create sample data file
        df = pd.DataFrame(SAMPLE_DATA)
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='sample_agriculture_data.csv'
        )
    except Exception as e:
        logger.error(f"Error downloading CSV: {str(e)}")
        flash("Error downloading the file", "error")
        return redirect(url_for('index'))
    

if __name__ == "__main__":
    app.run(debug=True)
