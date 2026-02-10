# Technical Interview Questions & Answers

## 🎯 **Data Analyst / AI Specialist Interview Preparation**

This document contains comprehensive technical questions and answers specifically tailored for the Homei Property Technology Data Analyst / AI Specialist position.

---

## 🐍 **Python Questions**

### **Q1: How would you handle missing data in a real estate dataset?**

**Answer:**
```python
import pandas as pd
import numpy as np

# Identify missing data
missing_data = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Strategies for handling missing data:

# 1. For numerical columns - use median (robust to outliers)
df['price'].fillna(df['price'].median(), inplace=True)
df['square_meters'].fillna(df['square_meters'].median(), inplace=True)

# 2. For categorical columns - use mode or 'Unknown'
df['location'].fillna(df['location'].mode()[0], inplace=True)
df['property_type'].fillna('Unknown', inplace=True)

# 3. For time series - use forward/backward fill
df['listing_date'].fillna(method='ffill', inplace=True)

# 4. Advanced - use predictive imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(random_state=42)
df[['price', 'square_meters', 'bedrooms']] = imputer.fit_transform(
    df[['price', 'square_meters', 'bedrooms']]
)

# 5. Create missing indicator features
df['price_missing'] = df['price'].isnull().astype(int)
```

### **Q2: Write a Python function to detect outliers in property prices using IQR method**

**Answer:**
```python
def detect_outliers_iqr(df, column):
    """
    Detect outliers using Interquartile Range (IQR) method
    
    Parameters:
    df: DataFrame
    column: Column name to check for outliers
    
    Returns:
    DataFrame with outlier indicators
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    # Add outlier indicator
    df[f'{column}_outlier'] = outliers
    
    # Summary statistics
    outlier_count = outliers.sum()
    outlier_percentage = (outlier_count / len(df)) * 100
    
    print(f"Outlier Analysis for {column}:")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
    
    return df

# Usage
df_with_outliers = detect_outliers_iqr(df, 'price')
```

### **Q3: How would you optimize a pandas DataFrame for memory usage?**

**Answer:**
```python
def optimize_dataframe_memory(df):
    """
    Optimize DataFrame memory usage by downcasting numeric types
    and converting categorical data to category dtype
    """
    start_memory = df.memory_usage().sum() / 1024**2
    print(f"Initial memory usage: {start_memory:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns to categorical
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If cardinality is low
            df[col] = df[col].astype('category')
    
    end_memory = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_memory - end_memory) / start_memory
    
    print(f"Optimized memory usage: {end_memory:.2f} MB")
    print(f"Memory reduction: {reduction:.1f}%")
    
    return df
```

---

## 🗄️ **SQL Questions**

### **Q1: Write a query to find the top 5 most expensive properties by location**

**Answer:**
```sql
WITH RankedProperties AS (
    SELECT 
        p.property_id,
        p.title,
        p.price,
        p.square_meters,
        l.city,
        l.district,
        p.price / p.square_meters AS price_per_sqm,
        ROW_NUMBER() OVER (PARTITION BY l.city ORDER BY p.price DESC) AS city_rank,
        ROW_NUMBER() OVER (PARTITION BY l.district ORDER BY p.price DESC) AS district_rank
    FROM Properties p
    JOIN Locations l ON p.location_id = l.location_id
    WHERE p.status = 'Available'
)
SELECT 
    city,
    district,
    property_id,
    title,
    price,
    price_per_sqm
FROM RankedProperties
WHERE city_rank <= 5
ORDER BY city, city_rank;
```

### **Q2: How would you identify properties that are overpriced compared to their area average?**

**Answer:**
```sql
WITH AreaAverages AS (
    SELECT 
        l.city,
        l.district,
        p.property_type,
        AVG(p.price / p.square_meters) AS avg_price_per_sqm,
        STDDEV(p.price / p.square_meters) AS std_price_per_sqm,
        COUNT(*) AS property_count
    FROM Properties p
    JOIN Locations l ON p.location_id = l.location_id
    WHERE p.status = 'Available'
    GROUP BY l.city, l.district, p.property_type
    HAVING COUNT(*) >= 5  -- Ensure statistical significance
),
OverpricedProperties AS (
    SELECT 
        p.property_id,
        p.title,
        p.price,
        p.square_meters,
        p.price / p.square_meters AS current_price_per_sqm,
        aa.avg_price_per_sqm,
        aa.std_price_per_sqm,
        (p.price / p.square_meters) / aa.avg_price_per_sqm AS price_ratio,
        CASE 
            WHEN (p.price / p.square_meters) > (aa.avg_price_per_sqm + 2 * aa.std_price_per_sqm) 
            THEN 'Significantly Overpriced'
            WHEN (p.price / p.square_meters) > (aa.avg_price_per_sqm + aa.std_price_per_sqm) 
            THEN 'Overpriced'
            ELSE 'Fairly Priced'
        END AS pricing_category
    FROM Properties p
    JOIN Locations l ON p.location_id = l.location_id
    JOIN AreaAverages aa ON l.city = aa.city 
                         AND l.district = aa.district 
                         AND p.property_type = aa.property_type
)
SELECT 
    property_id,
    title,
    price,
    square_meters,
    current_price_per_sqm,
    avg_price_per_sqm,
    price_ratio,
    pricing_category
FROM OverpricedProperties
WHERE pricing_category IN ('Overpriced', 'Significantly Overpriced')
ORDER BY price_ratio DESC;
```

### **Q3: Write a query to calculate month-over-month price changes for each location**

**Answer:**
```sql
WITH MonthlyPrices AS (
    SELECT 
        l.city,
        DATE_TRUNC('month', t.sale_date) AS sale_month,
        AVG(t.sale_price) AS avg_monthly_price,
        COUNT(*) AS transaction_count
    FROM Transactions t
    JOIN Properties p ON t.property_id = p.property_id
    JOIN Locations l ON p.location_id = l.location_id
    WHERE t.sale_date >= CURRENT_DATE - INTERVAL '12 months'
    GROUP BY l.city, DATE_TRUNC('month', t.sale_date)
),
MonthlyChanges AS (
    SELECT 
        city,
        sale_month,
        avg_monthly_price,
        transaction_count,
        LAG(avg_monthly_price) OVER (PARTITION BY city ORDER BY sale_month) AS prev_month_price,
        LAG(transaction_count) OVER (PARTITION BY city ORDER BY sale_month) AS prev_month_transactions
    FROM MonthlyPrices
)
SELECT 
    city,
    sale_month,
    avg_monthly_price,
    prev_month_price,
    transaction_count,
    prev_month_transactions,
    ROUND(
        (avg_monthly_price - prev_month_price) / prev_month_price * 100, 2
    ) AS price_change_pct,
    ROUND(
        (transaction_count - prev_month_transactions) / prev_month_transactions * 100, 2
    ) AS volume_change_pct,
    CASE 
        WHEN avg_monthly_price > prev_month_price THEN '📈 Increasing'
        WHEN avg_monthly_price < prev_month_price THEN '📉 Decreasing'
        ELSE '➡️ Stable'
    END AS price_trend
FROM MonthlyChanges
WHERE prev_month_price IS NOT NULL
ORDER BY city, sale_month DESC;
```

---

## 📊 **Power BI / Tableau Questions**

### **Q1: How would you design a dashboard for real estate executives?**

**Answer:**
**Dashboard Design Strategy:**

1. **Executive Summary Section**
   - Key KPIs: Total Market Value, Active Listings, Average Days on Market
   - Year-over-Year comparisons with trend indicators
   - Market health score (0-100)

2. **Market Overview**
   - Geographic heat map of property values
   - Price trends by location (line charts)
   - Market volume analysis

3. **Performance Metrics**
   - Sales team performance rankings
   - Property type performance comparison
   - ROI and profitability analysis

4. **Interactive Features**
   - Location drill-down capabilities
   - Time range selectors (7D, 30D, 90D, 1Y)
   - Property type filters
   - Export to PDF/Excel functionality

**DAX Measures Example:**
```dax
// Market Health Score
Market Health Score = 
VAR PriceTrend = [YoY Price Change]
VAR VolumeTrend = [YoY Volume Change]
VAR InventoryTurnover = [Inventory Turnover Rate]
RETURN
(PriceTrend * 0.4) + (VolumeTrend * 0.3) + (InventoryTurnover * 0.3)

// Year-over-Year Price Change
YoY Price Change = 
VAR CurrentPeriod = SELECTEDDATE('Calendar'[Date])
VAR PreviousPeriod = SAMEPERIODLASTYEAR('Calendar'[Date])
RETURN
DIVIDE(
    [Average Price] - CALCULATE([Average Price], PreviousPeriod),
    CALCULATE([Average Price], PreviousPeriod)
)
```

### **Q2: How would you optimize Power BI performance for large datasets?**

**Answer:**
**Performance Optimization Strategies:**

1. **Data Model Optimization**
   - Use star schema design
   - Create proper relationships
   - Use integer keys for relationships
   - Remove unnecessary columns

2. **DAX Optimization**
   - Use variables for complex calculations
   - Avoid CALCULATE on large tables
   - Use FILTER instead of WHERE in DAX
   - Implement time intelligence correctly

3. **Visual Optimization**
   - Limit number of visuals per page
   - Use appropriate visual types
   - Enable visual-level filters
   - Use drill-through instead of many pages

4. **Data Refresh Optimization**
   - Use incremental refresh
   - Schedule refresh during off-peak hours
   - Use DirectQuery for large datasets
   - Implement data caching

**Example Optimized DAX:**
```dax
// Optimized Sales Calculation
Total Sales = 
VAR SelectedDates = DATESBETWEEN('Calendar'[Date], MIN('Sales'[Date]), MAX('Sales'[Date]))
VAR FilteredSales = FILTER('Sales', 'Sales'[Date] IN SelectedDates)
RETURN
SUMX(FilteredSales, 'Sales'[Amount])

// Using Variables for Performance
Complex KPI = 
VAR TotalRevenue = [Total Revenue]
VAR TotalCost = [Total Cost]
VAR RevenueGrowth = [Revenue Growth Rate]
RETURN
DIVIDE(TotalRevenue - TotalCost, TotalRevenue) * (1 + RevenueGrowth)
```

---

## 🤖 **Machine Learning Questions**

### **Q1: How would you build a property price prediction model?**

**Answer:**
**Complete ML Pipeline:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class PropertyPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        
    def preprocess_data(self, df):
        """Complete preprocessing pipeline"""
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.create_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical(df)
        
        return df
    
    def handle_missing_values(self, df):
        """Strategic missing value handling"""
        # Numerical columns - median imputation
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - mode imputation
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def create_features(self, df):
        """Advanced feature engineering"""
        # Price per square meter
        df['price_per_sqm'] = df['price'] / df['square_meters']
        
        # Room ratio
        df['room_ratio'] = df['bedrooms'] / df['bathrooms']
        
        # Age factor
        df['age_factor'] = 2024 - df['year_built']
        
        # Location-based features
        location_stats = df.groupby('location')['price'].agg(['mean', 'std'])
        df['location_avg_price'] = df['location'].map(location_stats['mean'])
        df['location_price_std'] = df['location'].map(location_stats['std'])
        
        return df
    
    def train_model(self, X, y):
        """Train and optimize model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        
        # Grid search
        gb = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2')
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        self.model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model R²: {r2:.4f}")
        print(f"Model RMSE: {rmse:.2f}")
        
        return self.model
    
    def predict_price(self, property_data):
        """Make predictions on new data"""
        # Preprocess input
        processed_data = self.preprocess_data(pd.DataFrame([property_data]))
        
        # Scale features
        scaled_data = self.scaler.transform(processed_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)[0]
        
        return prediction
```

### **Q2: How would you evaluate and improve model performance?**

**Answer:**
**Model Evaluation & Improvement Strategy:**

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
    
    def comprehensive_evaluation(self):
        """Complete model evaluation"""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'R²': r2_score(self.y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'MAE': mean_absolute_error(self.y_test, y_pred),
            'MAPE': np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_test, self.y_test, cv=5)
        metrics['CV_R²_Mean'] = cv_scores.mean()
        metrics['CV_R²_Std'] = cv_scores.std()
        
        return metrics
    
    def residual_analysis(self):
        """Analyze prediction residuals"""
        y_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_pred
        
        # Plot residuals
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residual vs Predicted
        axes[0,0].scatter(y_pred, residuals)
        axes[0,0].axhline(y=0, color='r', linestyle='--')
        axes[0,0].set_xlabel('Predicted Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title('Residuals vs Predicted')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('Q-Q Plot')
        
        # Histogram of residuals
        axes[1,0].hist(residuals, bins=30)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Residual Distribution')
        
        # Actual vs Predicted
        axes[1,1].scatter(self.y_test, y_pred)
        axes[1,1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--')
        axes[1,1].set_xlabel('Actual Values')
        axes[1,1].set_ylabel('Predicted Values')
        axes[1,1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        plt.show()
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.X_test.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
            plt.show()
            
            return importance_df
```

---

## 🏢 **Business Scenario Questions**

### **Q1: How would you identify the best investment opportunities in the real estate market?**

**Answer:**
**Investment Opportunity Analysis Framework:**

```python
def identify_investment_opportunities(df):
    """Comprehensive investment opportunity analysis"""
    
    # 1. Price Competitiveness Analysis
    location_stats = df.groupby('location').agg({
        'price': ['mean', 'median', 'std'],
        'price_per_sqm': ['mean', 'median']
    })
    
    # 2. Market Activity Analysis
    market_activity = df.groupby('location').agg({
        'transaction_count': 'sum',
        'days_on_market': 'mean',
        'inventory_turnover': 'mean'
    })
    
    # 3. Growth Potential Analysis
    growth_metrics = calculate_growth_potential(df)
    
    # 4. Risk Assessment
    risk_scores = calculate_risk_scores(df)
    
    # 5. Investment Score Calculation
    investment_score = (
        price_competitiveness * 0.3 +
        market_activity_score * 0.25 +
        growth_potential * 0.25 +
        risk_adjustment * 0.2
    )
    
    return investment_score.sort_values(ascending=False)

def calculate_growth_potential(df):
    """Calculate growth potential indicators"""
    # Historical price appreciation
    price_appreciation = calculate_price_appreciation(df)
    
    # Development indicators
    development_score = calculate_development_indicators(df)
    
    # Infrastructure projects
    infrastructure_impact = calculate_infrastructure_impact(df)
    
    return {
        'price_appreciation': price_appreciation,
        'development_score': development_score,
        'infrastructure_impact': infrastructure_impact
    }
```

### **Q2: How would you help the sales team improve their performance using data?**

**Answer:**
**Sales Performance Optimization Strategy:**

```python
class SalesPerformanceOptimizer:
    def __init__(self, sales_data, property_data, customer_data):
        self.sales_data = sales_data
        self.property_data = property_data
        self.customer_data = customer_data
    
    def analyze_sales_performance(self):
        """Comprehensive sales performance analysis"""
        
        # 1. Individual Agent Performance
        agent_metrics = self.calculate_agent_metrics()
        
        # 2. Property Type Performance
        property_performance = self.analyze_property_performance()
        
        # 3. Customer Segmentation
        customer_segments = self.segment_customers()
        
        # 4. Lead Conversion Analysis
        conversion_analysis = self.analyze_conversion_rates()
        
        return {
            'agent_metrics': agent_metrics,
            'property_performance': property_performance,
            'customer_segments': customer_segments,
            'conversion_analysis': conversion_analysis
        }
    
    def calculate_agent_metrics(self):
        """Calculate individual agent performance metrics"""
        agent_stats = self.sales_data.groupby('agent_id').agg({
            'sale_amount': ['sum', 'mean', 'count'],
            'commission': 'sum',
            'days_to_close': 'mean',
            'customer_satisfaction': 'mean'
        }).round(2)
        
        # Calculate performance score
        agent_stats['performance_score'] = (
            (agent_stats[('sale_amount', 'sum')] / agent_stats[('sale_amount', 'sum')].max()) * 0.4 +
            (agent_stats[('sale_amount', 'count')] / agent_stats[('sale_amount', 'count')].max()) * 0.3 +
            (1 - agent_stats[('days_to_close', 'mean')] / agent_stats[('days_to_close', 'mean')].max()) * 0.3
        )
        
        return agent_stats.sort_values('performance_score', ascending=False)
    
    def generate_recommendations(self):
        """Generate actionable recommendations for sales team"""
        
        recommendations = []
        
        # 1. Performance Improvement
        low_performers = self.identify_low_performers()
        for agent in low_performers:
            recommendations.append({
                'agent_id': agent,
                'type': 'performance_improvement',
                'action': 'Additional training on property valuation',
                'priority': 'high'
            })
        
        # 2. Lead Assignment Optimization
        optimal_assignments = self.optimize_lead_assignment()
        recommendations.extend(optimal_assignments)
        
        # 3. Pricing Strategy
        pricing_recommendations = self.generate_pricing_recommendations()
        recommendations.extend(pricing_recommendations)
        
        return recommendations
```

---

## 📋 **Behavioral Questions**

### **Q1: Tell me about a time you used data to make a business decision**

**Answer:**
"In my previous role, I noticed that our property listings were staying on the market longer than industry average. I analyzed the data and found that properties priced 15% above market average took 45% longer to sell. I created a pricing optimization model that recommended optimal listing prices based on market conditions. This reduced average days on market from 120 to 75 days and increased sales volume by 20%."

### **Q2: How do you handle conflicting requirements from different stakeholders?**

**Answer:**
"I prioritize requirements based on business impact and technical feasibility. I facilitate meetings with all stakeholders to understand their needs, then create a prioritized roadmap. I communicate transparently about timelines and trade-offs, and ensure everyone understands the rationale behind decisions. For example, when the sales team wanted real-time pricing data but the infrastructure couldn't support it, I implemented a phased approach starting with daily updates while working toward real-time capabilities."

### **Q3: How do you stay updated with the latest data analysis and AI technologies?**

**Answer:**
"I regularly follow industry blogs and research papers, participate in online communities like Kaggle and Stack Overflow, and take online courses on emerging technologies. I also experiment with new tools and techniques in personal projects. For instance, I recently completed a course on advanced deep learning techniques and applied them to improve our property price prediction accuracy by 8%."

---

## 🎯 **Questions to Ask the Interviewer**

### **Technical Questions:**
1. What data sources and tools does Homei Property Technology currently use?
2. What are the biggest data challenges the company is facing?
3. How does the company approach data governance and quality?
4. What ML models or AI systems are currently in production?

### **Business Questions:**
1. What are the key business metrics for the Data Analyst / AI Specialist role?
2. How does the data team collaborate with other departments?
3. What are the growth plans for the data analytics function?
4. What success would look like in this role in the first 6-12 months?

### **Team Questions:**
1. What is the team structure and how would this role fit in?
2. What are the backgrounds of current team members?
3. What is the company culture around data-driven decision making?
4. What opportunities for professional development are available?

---

## 📚 **Additional Resources**

### **Technical Skills to Practice:**
- Advanced SQL queries and optimization
- Python data manipulation (pandas, numpy)
- Machine learning model development
- Data visualization best practices
- Statistical analysis and hypothesis testing

### **Domain Knowledge to Study:**
- Real estate market dynamics
- Property valuation methods
- Egyptian real estate market specifics
- Investment analysis techniques
- Market research methodologies

### **Tools to Master:**
- Power BI / Tableau advanced features
- Python ML libraries (scikit-learn, pandas)
- SQL database optimization
- Data pipeline tools (Airflow, dbt)
- Cloud platforms (AWS, Azure)

---

*This comprehensive preparation guide covers all technical aspects required for the Data Analyst / AI Specialist position at Homei Property Technology. Practice these questions and answers to demonstrate your expertise and readiness for the role.*
