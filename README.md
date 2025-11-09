<<<<<<< HEAD
# Australian Weather Prediction Project

A comprehensive machine learning web application that predicts rainfall in Australia. This project demonstrates the implementation of both Logistic Regression and Random Forest models for weather prediction, wrapped in a user-friendly web interface.

## Quick Start Guide

1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Set Up Python Environment (Python 3.8+ recommended)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
venv\Scripts\activate
# For Linux/Mac:
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the Dataset**
- Download `weatherAUS.csv` from [Kaggle's Weather in Australia dataset](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
- Place the CSV file in the project root directory

5. **Generate Models**
```bash
python generate_models.py
```
This will create two model files:
- `model_lr.pkl` (Logistic Regression model)
- `model_rf.pkl` (Random Forest model)

6. **Run the Application**
```bash
python app.py
```

7. **Access the Web Interface**
- Open your browser
- Go to `http://localhost:5000`

## Project Components

### 1. Data and Models
- `weatherAUS.csv`: Historical weather data from Australian weather stations
- `model_lr.pkl`: Trained Logistic Regression model
- `model_rf.pkl`: Trained Random Forest model

### 2. Core Files
- `app.py`: Flask web application
- `generate_models.py`: Script for training and saving models
- `requirements.txt`: Project dependencies
- `Weatherpredictionreview2.ipynb`: Jupyter notebook with data analysis

### 3. Web Interface
- `/templates`
  - `index.html`: Main prediction form
  - `result.html`: Displays prediction results
  - `visualize.html`: Data visualization page
- `/static`
  - `style.css`: Web interface styling

## Required Input Features

To make a prediction, you'll need the following weather measurements:

### Numeric Features
| Feature | Description | Unit |
|---------|------------|------|
| MinTemp | Minimum temperature | Celsius |
| MaxTemp | Maximum temperature | Celsius |
| Rainfall | Amount of rainfall | mm |
| Evaporation | Evaporation | mm |
| Sunshine | Hours of bright sunshine | hours |
| WindGustSpeed | Strongest wind gust speed | km/h |
| Humidity9am | Humidity at 9am | % |
| Humidity3pm | Humidity at 3pm | % |
| Pressure9am | Atmospheric pressure at 9am | hPa |
| Pressure3pm | Atmospheric pressure at 3pm | hPa |
| Temp9am | Temperature at 9am | Celsius |
| Temp3pm | Temperature at 3pm | Celsius |

### Categorical Features
| Feature | Description | Values |
|---------|------------|--------|
| RainToday | Did it rain today? | Yes/No |
| Location | Weather station location | Various Australian cities |

## Dependencies

Key packages required (see `requirements.txt` for versions):
```
flask
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

## Troubleshooting Guide

### Common Issues and Solutions

1. **Missing Model Files**
   - Error: `FileNotFoundError: Missing required file: model_lr.pkl` or `model_rf.pkl`
   - Solution: Run `python generate_models.py` to create the model files

2. **Missing Dataset**
   - Error: `FileNotFoundError: Missing required file: weatherAUS.csv`
   - Solution: Download the dataset from Kaggle and place it in the project root

3. **Package Installation Issues**
   - Error: Import errors or missing packages
   - Solution: Ensure you're in the virtual environment and run:
     ```bash
     pip install -r requirements.txt
     ```

4. **Model Generation Errors**
   - Error: Issues during model training
   - Solution: Check if `weatherAUS.csv` is corrupted or modified. Download a fresh copy if needed.

## How to Use the Web Interface

1. **Making Predictions**
   - Navigate to the home page
   - Fill in all required weather measurements
   - Select the location from the dropdown
   - Choose whether it rained today (Yes/No)
   - Click "Predict" to get results

2. **Interpreting Results**
   - The application shows predictions from both models
   - Prediction probabilities are displayed
   - Compare results between models for better insight

## Project Extension

Want to improve the project? Here are some suggestions:
1. Add more machine learning models (XGBoost, LightGBM)
2. Implement feature importance visualization
3. Add real-time weather data integration
4. Create an API endpoint for predictions
5. Add model performance metrics visualization

## Support

If you encounter any issues:
1. Check the troubleshooting guide above
2. Verify all requirements are installed
3. Ensure Python version compatibility (3.8+)
4. Check file permissions

## License

This project is open-source and available under the MIT License.
=======
# Dynamic-Rain-Prediction-via-Long-Range-Statistical-Sequence-Learning-Models-
>>>>>>> 8305b885421d63a62b6365225acf123203e0793a
