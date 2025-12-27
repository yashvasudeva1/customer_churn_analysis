import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Telco Customer Churn EDA", layout="wide")

st.title("Telco Customer Churn – Exploratory Data Analysis")
st.divider()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
option = st.sidebar.radio(
    "Go to section:",
    [
        "Dataset Overview",
        "Data Distribution",
        "Numerical Analysis",
        "Categorical Analysis",
        "Key Insights",
        "Predictive Modeling"
    ]
)

# -----------------------------
# Dataset Overview
# -----------------------------
if option == "Dataset Overview":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    with col2:
        st.subheader("Columns")
        st.write(f"Numerical Columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        st.write(f"Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:    
        st.subheader("Missing Values")
        missing = df.isnull().sum().to_frame(name='Values')
        st.dataframe(missing)
    with col4:
        st.subheader("Duplicates")
        dup_ids = df[df.duplicated("customerID", keep=False)]
        duplicate_counts = {}
        for col in df.columns:
            if col != "customerID":
                duplicate_counts[col] = dup_ids[col].duplicated().sum()
        duplicate_df = pd.DataFrame.from_dict(
            duplicate_counts,
            orient="index",
            columns=["Values"]
        )
        st.dataframe(duplicate_df)

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.divider()

    st.subheader("Churn Distribution")
    churn_counts = df['Churn'].value_counts(normalize=True)
    st.write(churn_counts)
    st.write("The dataset is imbalanced. Most customers do not churn, so accuracy alone is not a reliable metric.")
    with st.expander("Visualize Churn Distribution"):
        st.bar_chart(churn_counts)

# -----------------------------
# Data Distribution
# -----------------------------
elif option == "Data Distribution":
    st.subheader("Data Distributions")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_col = st.selectbox(
        "Select a numerical column",
        numerical_cols
    )
    data = df[selected_col].dropna()
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    # Original
    sns.histplot(data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Original")

    # Log
    sns.histplot(np.log1p(data), kde=True, ax=axes[0, 1])
    axes[0, 1].set_title("Log (log1p)")

    # Square root
    sns.histplot(np.sqrt(data), kde=True, ax=axes[0, 2])
    axes[0, 2].set_title("Square Root")

    # Box-Cox (positive values only)
    data_boxcox, _ = boxcox(data + 1e-6)
    sns.histplot(data_boxcox, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Box-Cox")

    # Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    data_yj = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
    sns.histplot(data_yj, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Yeo-Johnson")

    # Standardization
    sns.histplot((data - data.mean()) / data.std(), kde=True, ax=axes[1, 2])
    axes[1, 2].set_title("Standardized (Z-score)")

    plt.suptitle(f"Distribution Transformations for {selected_col}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    st.pyplot(fig)
    st.write("Even after transformations, features still exhibit skewness. We have to consider this when modeling.")

# -----------------------------
# Numerical Analysis
# -----------------------------
elif option == "Numerical Analysis":
    st.subheader("Numerical Features vs Churn")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[0])
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1])
    sns.boxplot(x='Churn', y='TotalCharges', data=df, ax=axes[2])

    st.pyplot(fig)
    st.subheader("Findings")
    st.write(
        """
        - Customers with low tenure churn more frequently  
        - Retention efforts should focus on new customers (first 6–12 months)  
        - High-priced plans should include added value, discounts, or loyalty benefits to reduce churn
        - Identifying high-spend churners early can save significant revenue  
        """
    )
# -----------------------------
# Categorical Analysis
# -----------------------------
elif option == "Categorical Analysis":
    st.subheader("Categorical Features vs Churn")

    
    col5, col6 = st.columns(2)
    with col5:
        category = st.selectbox(
            "Select a categorical feature:",
            [
                'Contract',
                'InternetService',
                'PaymentMethod',
                'TechSupport',
                'OnlineSecurity'
            ]
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=category, hue='Churn', data=df, ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)
    with col6:
        st.subheader("Observations")
        st.write(
        """
        - Contract type, tech support, and security services have a strong impact on customer churn  
        - Month-to-month contracts see the highest churn rates  
        - Offering value-added services can significantly reduce churn
        - Fiber Optic Internet Service has the most amount of churn (Possible reason : Fiber Optic is Sensitive and more prone to Disturbance)
        - If Technical Support is not present then amount of churn is high
        - Online Security absence also leads to high churn
        """
        )
# -----------------------------
# Key Insights
# -----------------------------
elif option == "Key Insights":
    st.subheader("Key EDA Insights")

    st.markdown(
        """
        - Customers on **month-to-month contracts** churn the most  
        - **Low tenure** is the strongest indicator of churn  
        - Higher **MonthlyCharges** increase churn probability  
        - Value-added services like **TechSupport** and **OnlineSecurity** reduce churn  
        - Churn prediction should focus on **recall for churned customers**
        """
    )

def about_the_coder():
    # We use a non-indented string to prevent Markdown from treating it as code
    html_code = """
    <style>
    .coder-card {
        background-color: transparent;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 10px;
        padding: 20px;
        display: flex;
        align-items: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .coder-img {
        width: 100px; /* Slightly larger for better visibility */
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid #FF4B4B; /* Streamlit Red */
        margin-right: 25px;
        flex-shrink: 0; /* Prevents image from shrinking */
    }
    .coder-info h3 {
        margin: 0;
        font-family: 'Source Sans Pro', sans-serif;
        color: inherit;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .coder-info p {
        margin: 10px 0;
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    .social-links {
        margin-top: 12px;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    .social-links a {
        text-decoration: none;
        color: #FF4B4B;
        font-weight: bold;
        font-size: 0.95rem;
        transition: color 0.3s;
    }
    .social-links a:hover {
        color: #ff2b2b;
        text-decoration: underline;
    }
    /* Mobile responsiveness */
    @media (max-width: 600px) {
        .coder-card {
            flex-direction: column;
            text-align: center;
            padding: 15px;
        }
        .coder-img {
            margin-right: 0;
            margin-bottom: 15px;
            width: 80px;
            height: 80px;
        }
        .social-links {
            justify-content: center;
        }
    }
    </style>  
    <div class="coder-card">
        <img src="https://ui-avatars.com/api/?name=Yash+Vasudeva&size=120&background=FF4B4B&color=fff&bold=true&rounded=true" class="coder-img" alt="Yash Vasudeva"/>
        <div class="coder-info">
            <h3>Developed by Yash Vasudeva</h3>
            <p>
                Results-driven Data & AI Professional skilled in <b>Data Analytics</b>, 
                <b>Machine Learning</b>, and <b>Deep Learning</b>. 
                Passionate about transforming raw data into business value and building intelligent solutions.
            </p>
            <div class="social-links">
                <a href="https://www.linkedin.com/in/yash-vasudeva/" target="_blank">LinkedIn</a>
                <a href="https://github.com/yashvasudeva1" target="_blank">GitHub</a>
                <a href="mailto:vasudevyash@gmail.com">Contact</a>
                <a href="https://yashvasudeva.vercel.app/" target="_blank">Portfolio</a>
            </div>
        </div>
    </div>
    """
        
    st.markdown(html_code, unsafe_allow_html=True)



if option == "Predictive Modeling":
    st.subheader("Predictive Modeling")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df.drop(columns=['customerID'], inplace=True)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    input_dict = {col: 0 for col in X.columns}
    input_dict['tenure'] = tenure
    input_dict['MonthlyCharges'] = monthly_charges
    input_dict['TotalCharges'] = total_charges

    if contract != "Month-to-month":
        input_dict[f"Contract_{contract}"] = 1

    if internet != "DSL":
        input_dict[f"InternetService_{internet}"] = 1

    input_dict[f"PaymentMethod_{payment}"] = 1
    input_dict[f"TechSupport_{tech_support}"] = 1
    input_dict[f"OnlineSecurity_{online_security}"] = 1
    input_dict[f"PaperlessBilling_{paperless}"] = 1

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    st.markdown("### Prediction")

    if st.button("Predict Churn"):
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"Customer is likely to churn (Probability: {probability:.2f})")
        else:
            st.success(f"Customer is likely to stay (Probability: {1 - probability:.2f})")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        st.markdown("### Model Performance Metrics")

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        m2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
        m3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
        m4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")
        m5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.2f}")

st.divider()
if __name__ == "__main__":
    about_the_coder()