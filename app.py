# Import necessary libraries
import streamlit as st  # Web application framework
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
from sklearn.ensemble import RandomForestClassifier  # Random Forest ML model
from sklearn.linear_model import LogisticRegression  # Logistic Regression ML model
from sklearn.model_selection import train_test_split  # Data splitting utility
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.inspection import permutation_importance  # Feature importance calculation
import plotly.express as px  # Interactive plotting library
import plotly.graph_objects as go  # Low-level Plotly interface
from plotly.subplots import make_subplots  # Subplot creation for Plotly
import seaborn as sns  # Statistical data visualization
from PIL import Image  # Image processing
import base64  # Base64 encoding/decoding
import os  # Operating system interface
import datetime  # Date and time manipulation
from io import StringIO, BytesIO  # I/O utilities
from xgboost import XGBClassifier  # XGBoost ML model
from reportlab.lib.pagesizes import letter  # PDF page size definition
from reportlab.pdfgen import canvas  # PDF generation
from reportlab.platypus import Table, TableStyle  # PDF table creation
from reportlab.lib import colors  # Color definitions for PDF
from reportlab.lib.utils import ImageReader  # Image handling for PDF
import io  # Core I/O functionality
import matplotlib.pyplot as plt  # Plotting library
from sklearn.metrics import (  # Model evaluation metrics
    roc_curve, auc, precision_recall_curve, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, accuracy_score, average_precision_score
)

# --- App Configuration ---
st.set_page_config(page_title="Corporate Tax Fraud Detection Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

# -------------------------------
# DEMO USER CREDENTIALS
# -------------------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "analyst": "analyst123"
}

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None


# -------------------------------
# FUNCTION TO SHOW CENTERED LOGO
# -------------------------------


def show_logo():
    # Read image and encode as base64
    with open("logo.png", "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode()

    # Title (white, bold, big) + centered logo
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 25px;">
            <h1 style="font-size: 40px; font-weight: 900; color: white; margin-bottom: 20px;">
                Corporate Tax Fraud Analytics Dashboard
            </h1>
            <img src="data:image/png;base64,{encoded}" width="150">
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------
# LOGIN PAGE
# -------------------------------
if not st.session_state.authenticated:
    show_logo()
    st.title("🔐 User Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            if username in USER_CREDENTIALS and password == USER_CREDENTIALS[username]:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success(f"✅ Logged in as {username}")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    st.stop()  # Prevent rest of app from loading until login

# -------------------------------
# DASHBOARD HEADER
# -------------------------------
show_logo()

# User info + Logout button (aligned right)
role = "Admin" if st.session_state.username == "admin" else "Analyst"
col1, col2 = st.columns([6, 2])
with col2:
    st.markdown(f"✅ Logged in as **{st.session_state.username}** ({role})")
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()

# -------------------------------
# DASHBOARD CONTENT
# -------------------------------
st.title("📊 Dashboard")

# Example content (replace with your own)
st.write("Welcome to the secured dashboard!")

# --- Header ---
st.title("🏢 Corporate Income Tax Fraud Detection Dashboard")
st.markdown("---")

# --- Header Image ---
try:
    if os.path.exists("A_digital_photograph_captures_a_person_using_a_lap.png"):
        image = Image.open("A_digital_photograph_captures_a_person_using_a_lap.png")
        st.image(image, caption="Corporate Income Tax Analysis System", use_container_width=True)
    else:
        st.info("Header image not found; continuing without it.")
except Exception as e:
    st.warning(f"Image load error: {e}")

st.markdown("---")

# --- Sample CSV Download ---
@st.cache_data
def get_sample_csv():
    sample_data = pd.DataFrame({
        'Company ID': [1, 2, 3, 4, 5],
        'Corporate Income Tax Rate': ['$25.0%', '$30.0%', '$22.5%', '$28.0%', '$26.0%'],
        'Net Sales': ['$1,000,000', '$1,500,000', '$800,000', '$2,000,000', '$1,200,000'],
        'Operating Expenses': ['$300,000', '$400,000', '$250,000', '$550,000', '$350,000'],
        'Other Income': ['$50,000', '$60,000', '$30,000', '$70,000', '$45,000'],
        'Gross Profit Margin': ['$20.0%', '$22.0%', '$18.0%', '$25.0%', '$21.0%'],
        'Operating Cash Flow': ['$100,000', '$200,000', '$80,000', '$250,000', '$150,000'],
        'Current Assets': ['$500,000', '$600,000', '$400,000', '$750,000', '$550,000'],
        'Current Liabilities': ['$200,000', '$250,000', '$150,000', '$300,000', '$220,000'],
        'Effective Tax Rate': ['$18.0%', '$21.0%', '$16.5%', '$23.0%', '$19.5%'],
        'Return on Assets (ROA)': ['$5.0%', '$6.5%', '$4.0%', '$7.0%', '$5.5%'],
        'Business Location': ['New York', 'California', 'Texas', 'Florida', 'Illinois'],
        'Industry Type': ['Finance', 'Retail', 'Manufacturing', 'Technology', 'Healthcare'],
        'Inventory Turnover': [8, 10, 12, 14, 9],
        'Number of Employees': [150, 200, 250, 300, 180],
        'Changes in Key Personnel': ['Yes', 'No', 'Yes', 'No', 'Yes']
    })
    return sample_data.to_csv(index=False).encode('utf-8')


col_download, _ = st.columns([1, 3])
with col_download:
    st.download_button(
        label="📄 Download Sample CSV",
        data=get_sample_csv(),
        file_name="sample_corporate.csv",
        mime="text/csv",
        help="Click to download a sample CSV file with the expected data format."
    )

# --- Upload and Process Data ---
uploaded_file = st.file_uploader("📁 Upload your corporate_processed.csv file", type=["csv"], key="uploader")

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Data Preview:", df.head())
    except Exception as e:
        st.error(f"❌ Failed to read CSV file: {e}")
        st.stop()


    # --- Data Cleaning Function ---
    def clean_currency(val):
        if pd.isna(val):
            return np.nan
        return float(str(val).replace('$', '').replace(',', '').replace('%', '').strip())

    monetary_cols = [
        'Corporate Income Tax Rate', 'Net Sales', 'Operating Expenses', 'Other Income',
        'Gross Profit Margin', 'Operating Cash Flow', 'Current Assets',
        'Current Liabilities', 'Effective Tax Rate', 'Return on Assets (ROA)',
        'Inventory Turnover', 'Number of Employees'
    ]

    df_cleaned = df.copy()
    for col in monetary_cols:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(clean_currency)

    # Convert 'Changes in Key Personnel' from categorical to numerical
    if 'Changes in Key Personnel' in df_cleaned.columns:
        df_cleaned['Changes in Key Personnel'] = df_cleaned['Changes in Key Personnel'].apply(
            lambda x: 1 if str(x).lower() == 'yes' else 0)

    # Generate Fraud labels based on tax rate discrepancy (for supervised learning demo)
    if 'Corporate Income Tax Rate' in df_cleaned.columns and 'Effective Tax Rate' in df_cleaned.columns:
        df_cleaned['Fraud'] = (df_cleaned['Effective Tax Rate'] < df_cleaned['Corporate Income Tax Rate'] * 0.8).astype(
            int)
    else:
        st.error(
            "Required columns for label generation not found. Make sure 'Corporate Income Tax Rate' and 'Effective Tax Rate' exist.")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("📌 Filters")
    if 'Business Location' in df_cleaned.columns:
        location_options = df_cleaned['Business Location'].unique()
        selected_location = st.sidebar.multiselect(
            "Select Locations",
            location_options,
            default=list(location_options),
            help="Filter data by business location."
        )
    else:
        selected_location = []

    if 'Industry Type' in df_cleaned.columns:
        industry_options = df_cleaned['Industry Type'].unique()
        selected_industry = st.sidebar.multiselect(
            "Select Industry Types",
            industry_options,
            default=list(industry_options),
            help="Filter data by industry type."
        )
    else:
        selected_industry = []

    df_filtered = df_cleaned.copy()
    if selected_location:
        df_filtered = df_filtered[df_filtered['Business Location'].isin(selected_location)]
    if selected_industry:
        df_filtered = df_filtered[df_filtered['Industry Type'].isin(selected_industry)]

    st.sidebar.success(f"✅ Filtered dataset: {df_filtered.shape[0]} records")

    # --- Missing Value Handling ---
    with st.expander("⚙️ Handle Missing Values"):
        imputation_strategy = st.selectbox(
            "Choose a strategy for handling missing values:",
            ['None', 'Mean Imputation', 'Median Imputation', 'Drop Rows'],
            help="Select how to deal with empty cells in financial data."
        )

        if imputation_strategy == 'Mean Imputation':
            for col in monetary_cols:
                if col in df_filtered.columns and df_filtered[col].isnull().any():
                    df_filtered[col] = df_filtered[col].fillna(df_filtered[col].mean())
            st.success("✅ Missing values imputed with column mean.")
        elif imputation_strategy == 'Median Imputation':
            for col in monetary_cols:
                if col in df_filtered.columns and df_filtered[col].isnull().any():
                    df_filtered[col] = df_filtered[col].fillna(df_filtered[col].median())
            st.success("✅ Missing values imputed with column median.")
        elif imputation_strategy == 'Drop Rows':
            initial_rows = df_filtered.shape[0]
            df_filtered.dropna(subset=[col for col in monetary_cols if col in df_filtered.columns], inplace=True)
            dropped_rows = initial_rows - df_filtered.shape[0]
            st.success(f"✅ Dropped {dropped_rows} rows with missing values.")
        else:
            st.info("No imputation applied.")

    # --- Model Configuration in Sidebar ---
    st.sidebar.header("🤖 Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Choose a Supervised Model",
        ['Random Forest', 'Logistic Regression', 'XGBoost'],
        help="Random Forest for non-linear patterns, Logistic for linear interpretability, XGBoost for boosting."
    )
    test_size = st.sidebar.slider(
        "Test Set Size",
        0.1, 0.4, 0.2, step=0.05,
        help="Proportion of data for testing."
    )

    all_features = [col for col in monetary_cols + ['Changes in Key Personnel'] if col in df_filtered.columns]
    features_selected = st.sidebar.multiselect(
        "Select features for modeling",
        all_features,
        default=all_features,
        help="Choose the financial metrics to be used by the classification model."
    )

    if features_selected:
        # Prepare data
        features = df_filtered[features_selected].copy()
        target = df_filtered['Fraud']
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Split data
        # if target only has one class, train_test_split with stratify will fail; handle gracefully
        stratify_arg = target if len(target.unique()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=test_size,
                                                            random_state=42, stratify=stratify_arg)

        # Train model
        if model_choice == 'Random Forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_choice == 'XGBoost':
            model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Predictions on full data
        y_full_pred = model.predict(features_scaled)
        df_filtered['Predicted Fraud'] = y_full_pred
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_full_proba = model.predict_proba(features_scaled)[:, 1]
            df_filtered['Fraud Probability'] = y_full_proba
            # probability for test set for ROC/PR and PDF
            y_proba = model.predict_proba(X_test)[:, 1]

        total_records = df_filtered.shape[0]
        total_fraud = df_filtered['Fraud'].sum()
        total_pred_fraud = int(df_filtered['Predicted Fraud'].sum())
        percent_pred_fraud = round((total_pred_fraud / total_records) * 100, 2) if total_records > 0 else 0

        # --- Dashboard Overview ---
        st.header("📊 Executive Dashboard Overview")

        # KPI Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Companies", total_records)
        with col2:
            delta_val = f"{(total_fraud / total_records * 100):.1f}%" if total_records > 0 else "0%"
            st.metric("Actual Fraud Cases", int(total_fraud), delta=delta_val)
        with col3:
            st.metric("Predicted Fraud Cases", total_pred_fraud, delta=f"{percent_pred_fraud}%")
        with col4:
            st.metric("Model Accuracy", f"{accuracy:.2%}", delta=f"{accuracy * 100:.1f}%")

        # KPI Metrics Row 2
        col5, col6, col7, col8 = st.columns(4)
        if 'Industry Type' in df_filtered.columns:
            top_fraud_industry = df_filtered[df_filtered['Predicted Fraud'] == 1]['Industry Type'].mode().iloc[
                0] if total_pred_fraud > 0 else "N/A"
            with col5:
                st.metric("Top Fraud Industry", top_fraud_industry)
        if 'Business Location' in df_filtered.columns:
            top_fraud_location = df_filtered[df_filtered['Predicted Fraud'] == 1]['Business Location'].mode().iloc[
                0] if total_pred_fraud > 0 else "N/A"
            with col6:
                st.metric("Top Fraud Location", top_fraud_location)
        with col7:
            avg_tax_rate = df_filtered[
                'Corporate Income Tax Rate'].mean() if 'Corporate Income Tax Rate' in df_filtered.columns else 0
            st.metric("Avg. Tax Rate", f"{avg_tax_rate:.1f}%")
        with col8:
            avg_roa = df_filtered[
                'Return on Assets (ROA)'].mean() if 'Return on Assets (ROA)' in df_filtered.columns else 0
            st.metric("Avg. ROA", f"{avg_roa:.1f}%")

        # Charts Row
        col_charts1, col_charts2 = st.columns(2)

        with col_charts1:
            # Fraud Distribution Pie Chart
            fig_pie = px.pie(df_filtered, names='Predicted Fraud', title="Predicted Fraud Distribution",
                             color_discrete_map={0: 'green', 1: 'red'})
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_charts2:
            # Probability Histogram
            if 'Fraud Probability' in df_filtered.columns:
                fig_hist = px.histogram(df_filtered, x='Fraud Probability', color='Predicted Fraud',
                                        title="Fraud Probability Distribution", nbins=20,
                                        color_discrete_map={0: 'green', 1: 'red'})
                st.plotly_chart(fig_hist, use_container_width=True)

        # Detailed Charts Row
        st.subheader("🔍 Detailed Visualizations")

        col_det1, col_det2 = st.columns(2)

        with col_det1:
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix (Test Set)",
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               color_continuous_scale='Blues')
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_det2:
            # Feature Importance (if Random Forest or XGBoost)
            if model_choice in ['Random Forest', 'XGBoost'] and features_selected:
                importances = model.feature_importances_
                feature_imp_df = pd.DataFrame({
                    'Feature': features_selected,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True)
                fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature', orientation='h',
                                 title="Feature Importance", text='Importance')
                st.plotly_chart(fig_imp, use_container_width=True)

        # Location/Industry Bar Charts
        if 'Business Location' in df_filtered.columns and total_pred_fraud > 0:
            fraud_by_loc = df_filtered[df_filtered['Predicted Fraud'] == 1][
                'Business Location'].value_counts().reset_index()
            fraud_by_loc.columns = ['Location', 'Count']
            fig_loc = px.bar(fraud_by_loc, x='Count', y='Location', orientation='h',
                             title="Predicted Fraud by Location", color='Count',
                             color_continuous_scale='Reds')
            st.plotly_chart(fig_loc, use_container_width=True)

        if 'Industry Type' in df_filtered.columns and total_pred_fraud > 0:
            fraud_by_ind = df_filtered[df_filtered['Predicted Fraud'] == 1][
                'Industry Type'].value_counts().reset_index()
            fraud_by_ind.columns = ['Industry', 'Count']
            fig_ind = px.bar(fraud_by_ind, x='Count', y='Industry', orientation='h',
                             title="Predicted Fraud by Industry", color='Count',
                             color_continuous_scale='Oranges')
            st.plotly_chart(fig_ind, use_container_width=True)

        # --- ROC Curve & PR Curve ---
        if y_proba is not None:
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)

            # PR
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)

            col_roc, col_pr = st.columns(2)

            with col_roc:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                             line=dict(dash='dash'), name='Random Guess'))
                fig_roc.update_layout(title=f"ROC Curve (AUC = {auc_score:.2f})",
                                      xaxis_title="False Positive Rate",
                                      yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_pr:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='PR Curve'))
                fig_pr.update_layout(title=f"Precision-Recall Curve (AP = {avg_precision:.2f})",
                                     xaxis_title="Recall",
                                     yaxis_title="Precision")
                st.plotly_chart(fig_pr, use_container_width=True)

        # --- Metric Comparison: Fraud vs Non-Fraud ---
        if features_selected:
            fraud_normal_comp = df_filtered.groupby('Predicted Fraud')[features_selected].mean().T
            # Ensure both columns present
            cols = fraud_normal_comp.columns.tolist()
            if 0 not in cols:
                fraud_normal_comp[0] = np.nan
            if 1 not in cols:
                fraud_normal_comp[1] = np.nan
            fraud_normal_comp = fraud_normal_comp[[0, 1]]
            fraud_normal_comp.columns = ['Normal (0)', 'Fraud (1)']

            fig_comp = px.bar(
                fraud_normal_comp,
                barmode='group',
                title="Average Metrics: Normal vs Fraud",
                labels={"value": "Average Value", "index": "Feature"}
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # --- Full Classification Report ---
        report_dict = classification_report(y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"], output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        st.subheader("📋 Classification Report")
        st.dataframe(report_df.style.format({
            'precision': "{:.2f}",
            'recall': "{:.2f}",
            'f1-score': "{:.2f}",
            'support': "{:.0f}"
        }))

        # Data Table
        st.subheader("📋 Detailed Results")
        display_cols = []
        if 'Company ID' in df_filtered.columns:
            display_cols.append('Company ID')
        display_cols += features_selected
        if 'Business Location' in df_filtered.columns:
            display_cols += ['Business Location']
        if 'Industry Type' in df_filtered.columns:
            display_cols += ['Industry Type']
        display_cols += ['Fraud', 'Predicted Fraud']
        if 'Fraud Probability' in df_filtered.columns:
            display_cols += ['Fraud Probability']
        st.dataframe(df_filtered[display_cols])

        # --- Admin Tools ---
        # --- Results Download Section ---
        st.subheader("📊 Export Results")

        # Direct CSV Download
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="fraud_dashboard_results.csv",
            mime="text/csv"
        )


        # Report Generation in PDF (canvas-based)
        if st.button("📄 Generate Report (PDF)"):
            try:
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter

                # Header & metadata
                c.setFont("Helvetica-Bold", 16)
                c.drawString(50, height - 50, "Fraud Detection Dashboard Report")

                c.setFont("Helvetica", 10)
                c.drawString(50, height - 80, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                c.drawString(50, height - 100, f"Model: {model_choice}")
                c.drawString(50, height - 120, f"Accuracy: {accuracy:.2%}")

                # Compute some metrics for the report
                precision_val = precision_score(y_test, y_pred, zero_division=0)
                recall_val = recall_score(y_test, y_pred, zero_division=0)
                f1_val = f1_score(y_test, y_pred, zero_division=0)
                auc_val = roc_auc_score(y_test, y_proba) if y_proba is not None else None

                c.drawString(50, height - 140, f"Precision (Overall): {precision_val:.2%}")
                c.drawString(50, height - 160, f"Recall (Overall): {recall_val:.2%}")
                c.drawString(50, height - 180, f"F1-score (Overall): {f1_val:.2%}")
                if auc_val is not None:
                    c.drawString(50, height - 200, f"AUC: {auc_val:.2f}")
                c.drawString(50, height - 220, f"Total Records: {total_records}")
                c.drawString(50, height - 240, f"Predicted Frauds: {total_pred_fraud}")

                # Class-wise metrics (safe extraction)
                report_for_classes = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                nf = report_for_classes.get("0", {"precision": 0, "recall": 0, "f1-score": 0})
                f_ = report_for_classes.get("1", {"precision": 0, "recall": 0, "f1-score": 0})

                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, height - 270, "Class-wise Performance:")
                c.setFont("Helvetica", 10)
                c.drawString(60, height - 290,
                             f"Not Fraud → Precision: {nf['precision']:.2%}, Recall: {nf['recall']:.2%}, F1: {nf['f1-score']:.2%}")
                c.drawString(60, height - 310,
                             f"Fraud     → Precision: {f_['precision']:.2%}, Recall: {f_['recall']:.2%}, F1: {f_['f1-score']:.2%}")

                # Fraud Distribution Pie (as image)
                try:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    vc = df_filtered['Predicted Fraud'].value_counts().sort_index()
                    labels = ['Not Fraud', 'Fraud']
                    counts = [int(vc.get(0, 0)), int(vc.get(1, 0))]
                    colors = ['#4CAF50', '#F44336']  # Green for not fraud, red for fraud
                    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                    ax.set_title("Predicted Fraud Distribution")
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

                    img_buf = io.BytesIO()
                    plt.savefig(img_buf, format="PNG", dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    img_buf.seek(0)
                    c.drawImage(ImageReader(img_buf), 50, height - 550, width=250, height=250)
                except Exception as e:
                    c.drawString(50, height - 300, f"Error creating pie chart: {str(e)}")

                # Fraud Probability Histogram (if exists)
                if 'Fraud Probability' in df_filtered.columns:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        df_filtered['Fraud Probability'].hist(bins=20, ax=ax, color='skyblue', edgecolor='black')
                        ax.set_title("Fraud Probability Distribution")
                        ax.set_xlabel("Probability")
                        ax.set_ylabel("Count")
                        ax.grid(axis='y', alpha=0.75)

                        img_buf2 = io.BytesIO()
                        plt.savefig(img_buf2, format="PNG", dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        img_buf2.seek(0)
                        c.drawImage(ImageReader(img_buf2), 320, height - 550, width=250, height=250)
                    except Exception as e:
                        c.drawString(320, height - 300, f"Error creating histogram: {str(e)}")

                c.showPage()

                # Page 2: Confusion Matrix
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Not Fraud", "Fraud"],
                                yticklabels=["Not Fraud", "Fraud"], ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")

                    img_buf3 = io.BytesIO()
                    plt.savefig(img_buf3, format="PNG", dpi=100, bbox_inches='tight')
                    plt.close(fig)
                    img_buf3.seek(0)
                    c.drawImage(ImageReader(img_buf3), 50, height - 400, width=300, height=300)
                except Exception as e:
                    c.drawString(50, height - 100, f"Error creating confusion matrix: {str(e)}")

                # ROC Curve (if available)
                if y_proba is not None:
                    try:
                        fpr_pdf, tpr_pdf, _ = roc_curve(y_test, y_proba)
                        roc_auc_pdf = auc(fpr_pdf, tpr_pdf)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.plot(fpr_pdf, tpr_pdf, color='darkorange', lw=2,
                                label=f'ROC curve (AUC = {roc_auc_pdf:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic')
                        ax.legend(loc="lower right")
                        ax.grid(alpha=0.3)

                        img_buf4 = io.BytesIO()
                        plt.savefig(img_buf4, format="PNG", dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        img_buf4.seek(0)
                        c.drawImage(ImageReader(img_buf4), 320, height - 400, width=300, height=300)
                    except Exception as e:
                        c.drawString(320, height - 100, f"Error creating ROC curve: {str(e)}")

                c.showPage()

                # Page 3: Feature importance & distribution charts
                if model_choice in ['Random Forest', 'XGBoost'] and features_selected:
                    try:
                        importances = model.feature_importances_
                        feature_imp_df = pd.DataFrame({
                            'Feature': features_selected,
                            'Importance': importances
                        }).sort_values(by='Importance', ascending=False)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=feature_imp_df, x='Importance', y='Feature', ax=ax, palette='viridis')
                        ax.set_title("Feature Importance")
                        ax.set_xlabel("Importance")
                        ax.set_ylabel("Feature")

                        img_buf6 = io.BytesIO()
                        plt.savefig(img_buf6, format="PNG", dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        img_buf6.seek(0)
                        c.drawImage(ImageReader(img_buf6), 50, height - 400, width=500, height=300)
                    except Exception as e:
                        c.drawString(50, height - 100, f"Error creating feature importance chart: {str(e)}")

                # Fraud by Location chart
                if 'Business Location' in df_filtered.columns and total_pred_fraud > 0:
                    try:
                        fraud_by_loc = df_filtered[df_filtered['Predicted Fraud'] == 1][
                            'Business Location'].value_counts().reset_index()
                        fraud_by_loc.columns = ['Location', 'Count']
                        # Limit to top 10 locations for better visualization
                        fraud_by_loc = fraud_by_loc.head(10)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(data=fraud_by_loc, x='Count', y='Location', ax=ax, palette='Reds_r')
                        ax.set_title("Top 10 Locations with Predicted Fraud")
                        ax.set_xlabel("Count")
                        ax.set_ylabel("Location")

                        img_buf7 = io.BytesIO()
                        plt.savefig(img_buf7, format="PNG", dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        img_buf7.seek(0)
                        c.drawImage(ImageReader(img_buf7), 50, height - 700, width=500, height=300)
                    except Exception as e:
                        c.drawString(50, height - 400, f"Error creating location chart: {str(e)}")

                c.showPage()

                # Finalize PDF
                c.save()

                # Provide download
                st.download_button(
                    label="📥 Download Report PDF",
                    data=buffer.getvalue(),
                    file_name="fraud_report.pdf",
                    mime="application/pdf"
                )

                st.success("PDF report generated successfully!")

            except Exception as e:
                st.error(f"Error generating PDF report: {str(e)}")
                st.info("Please check if all required dependencies are installed and try again.")