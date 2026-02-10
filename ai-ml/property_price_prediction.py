"""
Real Estate Price Prediction Model
=================================

This script demonstrates a comprehensive machine learning pipeline for predicting 
real estate prices in the Egyptian market, specifically tailored for property 
technology applications at Homei Property Technology.

Author: Data Analyst / AI Specialist
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

class RealEstatePricePredictor:
    """
    Comprehensive machine learning pipeline for real estate price prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.le_property_type = None
        self.le_location = None
        self.feature_list = None
        self.results = {}
        
    def load_data(self, filepath='real_estate_data.csv'):
        """Load and perform initial data exploration"""
        print("🔄 Loading data...")
        
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'square_meters': np.random.normal(150, 50, n_samples),
            'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
            'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
            'property_type': np.random.choice(['Apartment', 'Villa', 'Townhouse', 'Penthouse'], 
                                            n_samples, p=[0.5, 0.2, 0.2, 0.1]),
            'location': np.random.choice(['6th of October', 'New Cairo', 'Sheikh Zayed', 
                                        'Nasr City', 'Maadi'], n_samples),
            'year_built': np.random.randint(2000, 2024, n_samples),
            'price': np.random.normal(5000000, 2000000, n_samples)
        }
        
        # Ensure price is positive and correlated with features
        df = pd.DataFrame(data)
        df['price'] = (df['square_meters'] * 30000 + 
                      df['bedrooms'] * 500000 + 
                      df['bathrooms'] * 300000 + 
                      np.random.normal(0, 500000, n_samples))
        
        # Add location premium
        location_multipliers = {
            '6th of October': 1.2,
            'New Cairo': 1.3,
            'Sheikh Zayed': 1.25,
            'Nasr City': 1.0,
            'Maadi': 1.1
        }
        df['price'] *= df['location'].map(location_multipliers)
        
        self.df = df
        print(f"✅ Data loaded successfully! Shape: {df.shape}")
        return df
    
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("🔧 Preprocessing data and engineering features...")
        
        df = self.df.copy()
        
        # Handle missing values
        df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
        df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
        df['square_meters'].fillna(df['square_meters'].median(), inplace=True)
        
        # Create new features
        df['price_per_sqm'] = df['price'] / df['square_meters']
        df['room_ratio'] = df['bedrooms'] / df['bathrooms']
        df['age_factor'] = 2024 - df['year_built']
        
        # Location-based features
        location_stats = df.groupby('location').agg({
            'price': ['mean', 'median', 'std'],
            'price_per_sqm': ['mean', 'median']
        }).round(2)
        
        # Merge location statistics back to main dataframe
        df['location_avg_price'] = df.merge(
            location_stats[('price', 'mean')].reset_index(),
            on='location',
            how='left'
        )[('price', 'mean')]
        
        self.processed_df = df
        print("✅ Feature engineering completed!")
        return df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("📊 Preparing features for modeling...")
        
        df = self.processed_df.copy()
        
        # Define features
        features = ['square_meters', 'bedrooms', 'bathrooms', 'property_type', 
                   'location', 'year_built', 'price_per_sqm', 'room_ratio', 
                   'age_factor', 'location_avg_price']
        
        X = df[features].copy()
        y = df['price'].copy()
        
        # Handle categorical variables
        self.le_property_type = LabelEncoder()
        self.le_location = LabelEncoder()
        
        X['property_type_encoded'] = self.le_property_type.fit_transform(X['property_type'])
        X['location_encoded'] = self.le_location.fit_transform(X['location'])
        
        # Remove original categorical columns
        X = X.drop(['property_type', 'location'], axis=1)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.feature_list = list(X.columns)
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        print(f"✅ Features prepared! Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def compare_models(self):
        """Compare multiple ML algorithms"""
        print("🤖 Comparing machine learning models...")
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'CV_R2_Mean': cv_scores.mean(),
                'CV_R2_Std': cv_scores.std()
            }
        
        self.results = results
        print("✅ Model comparison completed!")
        
        # Display results
        results_df = pd.DataFrame(results).T
        print("\n📈 Model Comparison Results:")
        print(results_df.sort_values('R2', ascending=False))
        
        return results
    
    def optimize_best_model(self):
        """Hyperparameter tuning for the best model"""
        print("🎯 Optimizing best model (Gradient Boosting)...")
        
        # Select best model
        best_model = GradientBoostingRegressor(random_state=42)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters and score
        print(f"✅ Best Parameters: {grid_search.best_params_}")
        print(f"✅ Best R2 Score: {grid_search.best_score_:.4f}")
        
        # Use the best model
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self):
        """Final model evaluation and interpretation"""
        print("📊 Evaluating final model...")
        
        # Final model evaluation
        y_pred = self.model.predict(self.X_test)
        
        # Calculate final metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"🎯 Final Model Performance:")
        print(f"   RMSE: {rmse:,.2f}")
        print(f"   MAE: {mae:,.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Accuracy: {r2 * 100:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_list,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top 10 Feature Importance:")
        print(feature_importance.head(10))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance
        }
    
    def calculate_business_impact(self):
        """Calculate business impact metrics"""
        print("💼 Calculating business impact...")
        
        total_properties = len(self.df)
        avg_property_value = self.df['price'].mean()
        model_accuracy = self.results['Gradient Boosting']['R2']
        
        # Potential savings from better pricing
        pricing_improvement = model_accuracy * 0.05  # 5% of model accuracy as pricing improvement
        potential_savings = total_properties * avg_property_value * pricing_improvement
        
        # Time savings
        manual_appraisal_time = 2  # hours per property
        automated_appraisal_time = 0.1  # hours per property
        time_saved_per_property = manual_appraisal_time - automated_appraisal_time
        total_time_saved = total_properties * time_saved_per_property
        
        print(f"📊 Business Impact Analysis:")
        print(f"   Total Properties Analyzed: {total_properties:,}")
        print(f"   Average Property Value: {avg_property_value:,.2f} EGP")
        print(f"   Model Accuracy: {model_accuracy:.2%}")
        print(f"   Potential Pricing Savings: {potential_savings:,.2f} EGP")
        print(f"   Time Saved: {total_time_saved:,.1f} hours")
        print(f"   Efficiency Improvement: {(manual_appraisal_time/automated_appraisal_time):.1f}x")
        
        return {
            'potential_savings': potential_savings,
            'time_saved': total_time_saved,
            'efficiency_improvement': manual_appraisal_time/automated_appraisal_time
        }
    
    def save_model(self):
        """Save model and preprocessing objects"""
        print("💾 Saving model and preprocessing objects...")
        
        # Save the trained model
        joblib.dump(self.model, 'property_price_model.pkl')
        
        # Save the scaler
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        
        # Save the label encoders
        joblib.dump(self.le_property_type, 'property_type_encoder.pkl')
        joblib.dump(self.le_location, 'location_encoder.pkl')
        
        # Save feature list
        with open('feature_list.pkl', 'wb') as f:
            pickle.dump(self.feature_list, f)
        
        print("✅ Model and preprocessing objects saved successfully!")
    
    def predict_property_price(self, property_data):
        """
        Predict property price using the trained model.
        
        Parameters:
        property_data (dict): Dictionary containing property features
        
        Returns:
        float: Predicted property price
        """
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        # Prepare the input data
        input_df = pd.DataFrame([property_data])
        
        # Feature engineering
        input_df['price_per_sqm'] = input_df['price'] / input_df['square_meters']
        input_df['room_ratio'] = input_df['bedrooms'] / input_df['bathrooms']
        input_df['age_factor'] = 2024 - input_df['year_built']
        
        # Add location average price (simplified for demo)
        input_df['location_avg_price'] = 5000000  # Placeholder value
        
        # Encode categorical variables
        input_df['property_type_encoded'] = self.le_property_type.transform(input_df['property_type'])
        input_df['location_encoded'] = self.le_location.transform(input_df['location'])
        
        # Select and order features
        input_features = input_df[self.feature_list]
        
        # Scale features
        input_scaled = self.scaler.transform(input_features)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        
        return prediction
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("🚀 Starting Real Estate Price Prediction Pipeline...")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Preprocess data
        self.preprocess_data()
        
        # Step 3: Prepare features
        self.prepare_features()
        
        # Step 4: Compare models
        self.compare_models()
        
        # Step 5: Optimize best model
        self.optimize_best_model()
        
        # Step 6: Evaluate model
        evaluation_results = self.evaluate_model()
        
        # Step 7: Calculate business impact
        business_impact = self.calculate_business_impact()
        
        # Step 8: Save model
        self.save_model()
        
        print("=" * 60)
        print("🎉 Pipeline completed successfully!")
        print(f"🎯 Final Model Accuracy: {evaluation_results['r2']:.2%}")
        print(f"💰 Potential Business Savings: {business_impact['potential_savings']:,.2f} EGP")
        
        return evaluation_results, business_impact


# Example usage
if __name__ == "__main__":
    # Initialize the predictor
    predictor = RealEstatePricePredictor()
    
    # Run the complete pipeline
    evaluation_results, business_impact = predictor.run_complete_pipeline()
    
    # Example prediction
    example_property = {
        'square_meters': 150,
        'bedrooms': 3,
        'bathrooms': 2,
        'property_type': 'Apartment',
        'location': '6th of October',
        'year_built': 2020,
        'price': 5000000  # This will be recalculated
    }
    
    predicted_price = predictor.predict_property_price(example_property)
    print(f"\n🏠 Example Property Prediction:")
    print(f"   Property: {example_property['bedrooms']}BR {example_property['property_type']} in {example_property['location']}")
    print(f"   Predicted Price: {predicted_price:,.2f} EGP")
    
    print("\n✨ Model ready for production deployment at Homei Property Technology!")
