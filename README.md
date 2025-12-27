# Telco Customer Churn Analysis – Streamlit App

## Overview
This project is an interactive Streamlit web application for performing Exploratory Data Analysis (EDA) and predictive modeling on the Telco Customer Churn dataset. The app enables users to:
- Explore the dataset and its features
- Visualize data distributions and relationships
- Gain key business insights
- Predict customer churn using a logistic regression model

## Features

### 1. Dataset Overview
- View dataset shape, column types, missing values, and duplicates
- Preview the first few rows of the data
- Visualize churn distribution and understand class imbalance

### 2. Data Distribution
- Explore the distribution of numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)
- Visualize effects of transformations: Log, Square Root, Box-Cox, Yeo-Johnson, and Standardization
- Understand skewness and implications for modeling

### 3. Numerical Analysis
- Compare numerical features against churn status using boxplots
- Key findings on tenure, monthly charges, and total charges

### 4. Categorical Analysis
- Analyze the impact of categorical features (Contract, InternetService, PaymentMethod, TechSupport, OnlineSecurity) on churn
- Visualize churn rates across categories
- Business insights for reducing churn

### 5. Key Insights
- Summarized actionable insights from EDA
- Focus on contract types, tenure, charges, and value-added services

### 6. Predictive Modeling
- Logistic Regression model for churn prediction
- User-friendly form to input customer details and predict churn probability
- Model performance metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC

### 7. About the Developer
- Professional profile, social links, and contact information

## How to Run the App

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Customer-Churn-Analysis-.git
   cd Customer-Churn-Analysis-
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## File Structure
```
Customer-Churn-Analysis-/
│
├── app.py                        # Main Streamlit application
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Requirements
- Python 3.7+
- Streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used is [WA_Fn-UseC_-Telco-Customer-Churn.csv], which contains customer information, service details, and churn status.

## Model Details
- **Algorithm:** Logistic Regression
- **Preprocessing:**
  - Missing value handling
  - Categorical encoding (one-hot)
  - Feature scaling (StandardScaler)
- **Metrics:** Accuracy, Precision, Recall, F1 Score, ROC-AUC

## Usage
- Use the sidebar to navigate between EDA, insights, and prediction sections.
- In the "Predictive Modeling" section, enter customer details to get a churn prediction and see model metrics.

## Customization
- You can extend the app by adding more models, advanced visualizations, or business logic.
- For production use, consider model validation, hyperparameter tuning, and explainability features.

## Author
**Yash Vasudeva**  
[LinkedIn](https://www.linkedin.com/in/yash-vasudeva/) | [GitHub](https://github.com/yashvasudeva1) | [Portfolio](https://yashvasudeva.vercel.app/)

## License
This project is for educational and demonstration purposes.
