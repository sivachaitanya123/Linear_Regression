#!/usr/bin/env python3
"""
Test Script for Amazon Sales Analytics Project
Validates all core functionality and dependencies
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def test_dependencies():
    """Test all required dependencies"""
    print("🔍 Testing Dependencies...")
    
    try:
        # Test core data science libraries
        print("✅ NumPy:", np.__version__)
        print("✅ Pandas:", pd.__version__)
        print("✅ Matplotlib:", plt.matplotlib.__version__)
        print("✅ Seaborn:", sns.__version__)
        
        # Test machine learning libraries
        from sklearn import __version__ as sk_version
        print("✅ Scikit-learn:", sk_version)
        
        # Test statistical libraries
        print("✅ Statsmodels:", sm.__version__)
        print("✅ SciPy:", stats.__version__)
        
        # Test visualization libraries
        import plotly
        print("✅ Plotly:", plotly.__version__)
        
        print("✅ All dependencies are working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Dependency test failed: {e}")
        return False

def test_data_generation():
    """Test Amazon sales data generation"""
    print("\n📊 Testing Data Generation...")
    
    try:
        # Generate sample Amazon sales data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic sales features
        marketing_spend = np.random.uniform(10000, 100000, n_samples)
        website_traffic = np.random.uniform(50000, 200000, n_samples)
        avg_product_price = np.random.uniform(20, 200, n_samples)
        seasonal_factor = np.sin(2 * np.pi * np.arange(n_samples) / 365) * 0.3
        promotion_active = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Generate revenue with realistic relationships
        revenue = (
            50000 +                    # Base revenue
            0.8 * marketing_spend +    # Marketing impact
            0.3 * website_traffic +    # Traffic impact
            100 * avg_product_price +  # Price impact
            20000 * seasonal_factor +  # Seasonal impact
            15000 * promotion_active + # Promotion impact
            np.random.normal(0, 5000, n_samples)  # Random noise
        )
        
        # Create DataFrame
        sales_df = pd.DataFrame({
            'marketing_spend': marketing_spend,
            'website_traffic': website_traffic,
            'avg_product_price': avg_product_price,
            'seasonal_factor': seasonal_factor,
            'promotion_active': promotion_active,
            'revenue': revenue
        })
        
        print(f"✅ Generated {len(sales_df)} sales records")
        print(f"✅ Revenue range: ${sales_df['revenue'].min():,.0f} - ${sales_df['revenue'].max():,.0f}")
        print(f"✅ Average revenue: ${sales_df['revenue'].mean():,.0f}")
        
        return sales_df
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return None

def test_simple_linear_regression(sales_df):
    """Test simple linear regression"""
    print("\n📈 Testing Simple Linear Regression...")
    
    try:
        # Prepare data
        X = sales_df[['marketing_spend']]
        y = sales_df['revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Build model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Model trained successfully")
        print(f"✅ Intercept: ${model.intercept_:,.2f}")
        print(f"✅ Coefficient: {model.coef_[0]:.4f}")
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ RMSE: ${rmse:,.2f}")
        print(f"✅ MAE: ${mae:,.2f}")
        
        return model, (mse, rmse, mae, r2)
        
    except Exception as e:
        print(f"❌ Simple linear regression failed: {e}")
        return None, None

def test_multiple_linear_regression(sales_df):
    """Test multiple linear regression"""
    print("\n🎯 Testing Multiple Linear Regression...")
    
    try:
        # Prepare data
        feature_columns = ['marketing_spend', 'website_traffic', 'avg_product_price', 
                          'seasonal_factor', 'promotion_active']
        X = sales_df[feature_columns]
        y = sales_df['revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ Multiple regression model trained")
        print(f"✅ Intercept: ${model.intercept_:,.2f}")
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ RMSE: ${rmse:,.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        print("✅ Feature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"   • {row['Feature']}: {row['Coefficient']:.2f}")
        
        return model, (mse, rmse, mae, r2)
        
    except Exception as e:
        print(f"❌ Multiple linear regression failed: {e}")
        return None, None

def test_regularization(sales_df):
    """Test regularization techniques"""
    print("\n🛡️ Testing Regularization...")
    
    try:
        # Prepare data
        feature_columns = ['marketing_spend', 'website_traffic', 'avg_product_price', 
                          'seasonal_factor', 'promotion_active']
        X = sales_df[feature_columns]
        y = sales_df['revenue']
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Test different regularization techniques
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Elastic Net (α=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            results[name] = r2
            print(f"✅ {name}: R² = {r2:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Regularization test failed: {e}")
        return None

def test_cross_validation(sales_df):
    """Test cross-validation"""
    print("\n🔄 Testing Cross-Validation...")
    
    try:
        # Prepare data
        feature_columns = ['marketing_spend', 'website_traffic', 'avg_product_price', 
                          'seasonal_factor', 'promotion_active']
        X = sales_df[feature_columns]
        y = sales_df['revenue']
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform cross-validation
        model = LinearRegression()
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        
        print(f"✅ Cross-validation scores: {cv_scores}")
        print(f"✅ Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return cv_scores
        
    except Exception as e:
        print(f"❌ Cross-validation test failed: {e}")
        return None

def test_visualization():
    """Test visualization capabilities"""
    print("\n📊 Testing Visualization...")
    
    try:
        # Create sample data for visualization
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 1, 100)
        
        # Test matplotlib
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6)
        plt.plot(x, 2*x + 1, 'r-', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Test Visualization')
        plt.close()  # Close to avoid displaying
        print("✅ Matplotlib visualization working")
        
        # Test seaborn
        df = pd.DataFrame({'x': x, 'y': y})
        sns.regplot(data=df, x='x', y='y')
        plt.close()  # Close to avoid displaying
        print("✅ Seaborn visualization working")
        
        # Test plotly
        fig = px.scatter(x=x, y=y, title="Test Plotly Visualization")
        fig.update_layout(showlegend=False)
        print("✅ Plotly visualization working")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_model_evaluation_metrics():
    """Test model evaluation metrics"""
    print("\n📈 Testing Model Evaluation Metrics...")
    
    try:
        # Create sample predictions
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 480])
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"✅ MSE: {mse:.2f}")
        print(f"✅ RMSE: {rmse:.2f}")
        print(f"✅ MAE: {mae:.2f}")
        print(f"✅ MAPE: {mape:.2f}%")
        print(f"✅ R²: {r2:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model evaluation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Amazon Sales Analytics Project - Test Suite")
    print("=" * 60)
    
    # Test dependencies
    if not test_dependencies():
        print("❌ Dependency test failed. Please check your installation.")
        return False
    
    # Test data generation
    sales_df = test_data_generation()
    if sales_df is None:
        print("❌ Data generation failed.")
        return False
    
    # Test simple linear regression
    simple_model, simple_metrics = test_simple_linear_regression(sales_df)
    if simple_model is None:
        print("❌ Simple linear regression failed.")
        return False
    
    # Test multiple linear regression
    multiple_model, multiple_metrics = test_multiple_linear_regression(sales_df)
    if multiple_model is None:
        print("❌ Multiple linear regression failed.")
        return False
    
    # Test regularization
    reg_results = test_regularization(sales_df)
    if reg_results is None:
        print("❌ Regularization test failed.")
        return False
    
    # Test cross-validation
    cv_scores = test_cross_validation(sales_df)
    if cv_scores is None:
        print("❌ Cross-validation test failed.")
        return False
    
    # Test visualization
    if not test_visualization():
        print("❌ Visualization test failed.")
        return False
    
    # Test model evaluation metrics
    if not test_model_evaluation_metrics():
        print("❌ Model evaluation test failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("✅ Your Amazon Sales Analytics project is ready to use!")
    print("\n📚 Next Steps:")
    print("1. Start with notebooks/01_business_context.ipynb")
    print("2. Follow the learning path through all notebooks")
    print("3. Deploy your models using the AWS deployment guide")
    print("\n🚀 Happy learning!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 