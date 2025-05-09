import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better window management
plt.close('all')  # Close any open figures
import os  # Add this for file operations

# Create output directory if it doesn't exist
def ensure_output_directory(directory='output'):
    """
    Creates output directory if it doesn't exist
    
    Parameters:
    - directory: Directory path for saving outputs
    
    Returns:
    - Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")
    else:
        print(f"Using existing output directory: {directory}")
    return directory

# Function to safely save figures without overwriting
def save_figure(fig_path):
    """
    Saves a figure only if it doesn't already exist
    
    Parameters:
    - fig_path: Full path for saving the figure
    """
    if not os.path.exists(fig_path):
        plt.savefig(fig_path)
        print(f"Saved: {fig_path}")
    else:
        print(f"File already exists (not overwritten): {fig_path}")

# Function for exploratory data analysis
def explore_data(df, output_dir):
    """
    Perform exploratory data analysis on the dataset
    """
    print("Dataset Information:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nCorrelation Analysis:")
    correlation = df.corr()
    print(correlation['sales'].sort_values(ascending=False))
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    save_figure(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    # Create a single figure for pairplot
    sns.pairplot(df, x_vars=['tv', 'radio', 'newspaper'], y_vars='sales', height=4)
    plt.suptitle('Relationship between Advertising Channels and Sales', y=1.02)
    save_figure(os.path.join(output_dir, 'channel_relationships.png'))
    plt.show()
    plt.close('all')  # Close all figures
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sales'], kde=True)
    plt.title('Distribution of Sales')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    save_figure(os.path.join(output_dir, 'sales_distribution.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    return df

# Function to clean the data
def clean_data(df):
    """
    Handle missing values and outliers in the dataset
    """
    print("\nCleaning Data:")
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype.kind in 'ifc':  # if column is integer, float or complex
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    
    # Remove outliers using IQR method
    print("\nOutlier Treatment:")
    rows_before = len(df)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"Column '{col}': {outliers} outliers detected")
        
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    rows_after = len(df)
    print(f"Rows removed due to outliers: {rows_before - rows_after}")
    
    return df

# Function for feature engineering
def feature_engineering(df):
    """
    Create new features to improve model performance
    """
    print("\nFeature Engineering:")
    
    # Create interaction terms between advertising channels
    df['tv_radio'] = df['tv'] * df['radio']
    df['tv_newspaper'] = df['tv'] * df['newspaper']
    df['radio_newspaper'] = df['radio'] * df['newspaper']
    
    # Create squared terms to capture non-linear relationships
    df['tv_squared'] = df['tv'] ** 2
    df['radio_squared'] = df['radio'] ** 2
    df['newspaper_squared'] = df['newspaper'] ** 2
    
    # Create ratio features
    df['tv_ratio'] = df['tv'] / (df['radio'] + 1)  # Adding 1 to avoid division by zero
    df['newspaper_ratio'] = df['newspaper'] / (df['tv'] + 1)
    
    print(f"Original features: {['tv', 'radio', 'newspaper']}")
    print(f"New features added: {[col for col in df.columns if col not in ['tv', 'radio', 'newspaper', 'sales']]}")
    
    return df

# Function for feature selection
def feature_selection(X, y, output_dir):
    """
    Identify the most important features for the model
    """
    print("\nFeature Importance Analysis:")
    
    # Use Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Sales Prediction')
    plt.tight_layout()
    save_figure(os.path.join(output_dir, 'feature_importance.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    return feature_importance

# Function for model selection with hyperparameter tuning
def model_selection(X_train, y_train, X_test, y_test):
    """
    Select the best model and tune hyperparameters
    """
    print("\nModel Selection and Evaluation:")
    
    # Define base models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42, eval_metric='rmse')
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Train model
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'cv_rmse': cv_rmse,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'model': model
        }
        
        print(f"{name} - CV RMSE: {cv_rmse:.4f}, Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    # Find best model
    best_model_name = min(results, key=lambda x: results[x]['test_rmse'])
    best_model = results[best_model_name]['model']
    print(f"\nBest Model: {best_model_name}")
    
    # Hyperparameter tuning for best model
    if best_model_name == 'Linear Regression':
        # Linear Regression doesn't need much tuning
        final_model = best_model
    
    elif best_model_name == 'Random Forest':
        print("\nTuning Random Forest Hyperparameters...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                                  param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        final_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    
    elif best_model_name == 'XGBoost':
        print("\nTuning XGBoost Hyperparameters...")
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300]
        }
        grid_search = GridSearchCV(XGBRegressor(random_state=42, eval_metric='rmse'), 
                                  param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        final_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    
    # Final evaluation
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"\nFinal Model Performance - RMSE: {final_rmse:.4f}, R²: {final_r2:.4f}")
    
    return final_model, y_pred

# Function to plot predictions
def plot_predictions(y_test, y_pred, X_test, output_dir):
    """
    Visualize model predictions against actual values
    """
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.tight_layout()
    save_figure(os.path.join(output_dir, 'actual_vs_predicted.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    save_figure(os.path.join(output_dir, 'residual_plot.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    save_figure(os.path.join(output_dir, 'residual_distribution.png'))
    plt.show()
    plt.close()  # Close the figure after showing
    
    # If we have feature data, show predictions by TV advertising spend
    if X_test is not None and 'tv' in X_test.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test['tv'], y_test, alpha=0.7, label='Actual')
        plt.scatter(X_test['tv'], y_pred, alpha=0.7, label='Predicted')
        plt.xlabel('TV Advertising Spend')
        plt.ylabel('Sales')
        plt.title('Sales vs TV Advertising')
        plt.legend()
        plt.tight_layout()
        save_figure(os.path.join(output_dir, 'tv_predictions.png'))
        plt.show()
        plt.close()  # Close the figure after showing

# Function to provide business insights and recommendations
def business_insights(model, X, feature_importance, output_dir):
    """
    Extract business insights from the model
    """
    print("\nBusiness Insights and Recommendations:")
    
    # Most influential channels
    top_features = feature_importance.head(3)['Feature'].values
    print(f"Top influencing factors for sales: {', '.join(top_features)}")
    
    # Coefficient analysis (for linear regression)
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': model.coef_
        }).sort_values(by='Coefficient', ascending=False)
        
        print("\nImpact of each advertising channel (Linear Regression Coefficients):")
        print(coef_df)
        
        # Plot coefficients
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
        plt.title('Impact of Advertising Channels on Sales')
        plt.tight_layout()
        save_figure(os.path.join(output_dir, 'coefficient_impact.png'))
        plt.show()
        plt.close()  # Close the figure after showing
    
    # Recommendations
    print("\nRecommendations for Advertising Strategy:")
    print("1. Focus on the most influential advertising channels")
    print("2. Optimize budget allocation based on channel effectiveness")
    print("3. Consider synergistic effects between channels (interaction terms)")
    print("4. Regularly monitor and update the model as new data becomes available")
    
    # Example budget allocation
    print("\nExample Budget Allocation Strategy:")
    channels = ['tv', 'radio', 'newspaper']
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = {ch: 0 for ch in channels}
        for i, col in enumerate(X.columns):
            for ch in channels:
                if ch in col:
                    importances[ch] += model.feature_importances_[i]
    elif hasattr(model, 'coef_'):
        # For linear regression
        importances = {ch: 0 for ch in channels}
        for i, col in enumerate(X.columns):
            for ch in channels:
                if ch == col:
                    importances[ch] = abs(model.coef_[i])
    
    # Normalize to get percentages
    total = sum(importances.values())
    if total > 0:  # Avoid division by zero
        budget_allocation = {ch: (imp/total)*100 for ch, imp in importances.items()}
        
        print("Recommended budget allocation based on model:")
        for ch, alloc in budget_allocation.items():
            print(f"{ch.capitalize()}: {alloc:.1f}%")

def main():
    """
    Main function to execute the sales prediction pipeline
    """
    print("="*80)
    print("                     SALES PREDICTION USING PYTHON")
    print("="*80)
    
    # Create output directory
    output_dir = ensure_output_directory()
    
    # Load the dataset
    try:
        df = pd.read_csv('advertising.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: advertising.csv file not found.")
        print("Creating sample data for demonstration...")
        
        # Create sample data
        np.random.seed(42)
        n = 200
        tv = np.random.uniform(0, 300, n)
        radio = np.random.uniform(0, 50, n)
        newspaper = np.random.uniform(0, 100, n)
        
        # Create sales with realistic relationships
        sales = 4.5 + 0.05*tv + 0.25*radio + 0.1*newspaper + 0.0003*tv*radio + np.random.normal(0, 3, n)
        
        df = pd.DataFrame({
            'tv': tv,
            'radio': radio,
            'newspaper': newspaper,
            'sales': sales
        })
        
        # Add some missing values for demonstration
        for col in df.columns:
            mask = np.random.random(len(df)) < 0.05
            df.loc[mask, col] = np.nan
    
    # Step 1: Exploratory Data Analysis
    df = explore_data(df, output_dir)
    
    # Step 2: Data Cleaning
    df = clean_data(df)
    
    # Step 3: Feature Engineering
    df = feature_engineering(df)
    
    # Step 4: Define features and target
    X = df.drop('sales', axis=1)
    y = df['sales']
    
    # Step 5: Feature Selection
    feature_importance = feature_selection(X, y, output_dir)
    
    # Step 6: Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 7: Data Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to dataframes to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Step 8: Model Selection and Evaluation
    best_model, y_pred = model_selection(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Step 9: Visualize predictions
    plot_predictions(y_test, y_pred, X_test, output_dir)
    
    # Step 10: Extract business insights
    business_insights(best_model, X, feature_importance, output_dir)
    
    print("\nSales Prediction Project Completed Successfully!")
    print(f"All output files saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()