# Sales Prediction Using Python

This project aims to predict product sales based on advertising expenditure across different channels (current dataset:- TV, Radio, Newspaper) and on various other datasets also. By leveraging machine learning techniques, the model helps businesses make informed decisions regarding advertising costs and optimize their marketing strategies.

## Project Overview

In businesses that offer products or services, predicting future sales is crucial for strategic planning. This project:

- Analyzes the relationship between advertising spending and sales
- Creates and compares different machine learning models
- Provides data-driven insights for optimizing advertising budget allocation
- Visualizes key findings for better understanding

## Dataset

The dataset contains information about advertising spending across three channels (TV, Radio, Newspaper) and the resulting sales:

- TV: Advertising dollars spent on TV ads
- Radio: Advertising dollars spent on Radio ads
- Newspaper: Advertising dollars spent on Newspaper ads
- Sales: Units of products sold

## Features

- **Exploratory Data Analysis**: Comprehensive analysis of advertising and sales data
- **Data Cleaning**: Handling missing values and outliers
- **Feature Engineering**: Creating interaction terms and additional features to improve predictions
- **Model Selection**: Comparing multiple models (Linear Regression, Random Forest, XGBoost)
- **Hyperparameter Tuning**: Optimizing model parameters for better performance
- **Business Insights**: Translating model results into actionable business recommendations

## Installation

```bash
# Clone this repository
git clone https://github.com/KingAarya78/Sales-Prediction-Tool.git

# Navigate to the project directory
cd sales-prediction

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Run the main script
python sales_prediction.py
```

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Results

The model successfully predicts sales based on advertising expenditure, with the following key findings:

- TV advertising has the strongest influence on sales
- There are synergistic effects between advertising channels
- The optimal advertising budget allocation is provided based on model insights

![feature_importance](https://github.com/user-attachments/assets/352cdc38-67d6-4f7e-91a6-c1432d767624)


## Visualizations

The project includes several visualizations to help understand the data and model performance:

- Correlation heatmap of advertising channels and sales
- Feature importance for sales prediction
- Actual vs Predicted sales
- Residual analysis plots

## Future Improvements

- Incorporate time-series analysis for seasonal trends
- Include additional factors like pricing, promotions, and competitor activities
- Develop an interactive dashboard for real-time analysis

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Advertising Dataset](https://www.kaggle.com/purbar/advertising-data)
- This project is performed as a task given in my Intership at Codsoft company.
