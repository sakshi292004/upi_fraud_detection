import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="UPI Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ----------------- Dynamic CSS for styling -----------------
st.markdown(
    """
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #87CEFA, #0a2342);
        color: white;
    }

    /* Headers in yellow */
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700;
    }

    /* Buttons with hover effect */
    div.stButton > button:first-child {
        background-color: #FFD700;
        color: #0a2342;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        background-color: #FFC700;
    }

    /* Tabs with hover effect */
    .stTabs [role="tab"] {
        background-color: rgba(70, 130, 180, 0.8);
        color: #FFD700;
        font-weight: bold;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"]:hover {
        background-color: rgba(70, 130, 180, 1);
        transform: scale(1.05);
    }

    /* Dataframe header styling */
    .dataframe th {
        background-color: #FFD700 !important;
        color: #0a2342 !important;
    }

    /* Card-like form style */
    .stForm {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }

    /* Fade-in animation for images */
    @keyframes fadeIn {
      from {opacity: 0;}
      to {opacity: 1;}
    }
    img {
      animation: fadeIn 2s ease-in-out;
      transition: transform 0.5s ease-in-out;
    }
    img:hover {
      transform: scale(1.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load Model -----------------
try:
    with open("unsupervised_upi_fraud_iforest.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ----------------- Features -----------------
numeric_features = [
    "TransactionAmount", "CustomerAge", "LoginAttempts",
    "AccountBalance", "TransactionDuration",
    "tx_hour", "tx_dayofweek", "time_since_prev"
]

categorical_features = [
    "TransactionType", "Location", "Channel", "CustomerOccupation"
]

# ----------------- Header -----------------
col1, col2 = st.columns([1, 2])
with col1:
    st.image("upi_ikmage3.jpeg", width=250)
with col2:
    st.title("💳 UPI Fraud Detection System")
    st.markdown("### Detect fraudulent transactions in real-time 🚀")

# ----------------- Center Image5 -----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")  # Empty
with col2:
    #st.image("upi_image5.jpeg", width=250)  # Centered image
    st.image(r"C:\Users\saksh\upi_fraud_detection\upi_image5.jpg", width=750)

    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)  # Add spacing
with col3:
    st.write("")  # Empty

# ----------------- Project Description -----------------
st.markdown(
    """
    <div style='
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        color: white;
        font-size: 16px;
        line-height: 1.6;
        transition: all 0.3s ease;
        animation: fadeIn 2s ease-in-out;
    '>
        <h2 style='color:#FFD700;'>UPI Fraud Detection System</h2>
        <p>
            The UPI Fraud Detection System is designed to detect and prevent fraudulent transactions in real-time.
            The system utilizes the Isolation Forest algorithm to identify anomalous transaction patterns.
            The frontend is built using Streamlit, providing an interactive and user-friendly interface for monitoring transactions,
            while the backend is powered by FastAPI, ensuring efficient handling of API requests.
            Transaction data is securely managed, enabling fast and reliable detection, thereby minimizing financial losses and enhancing user trust.
        </p>
        <p>
            <b>Created by:</b> Sakshi Ubale<br>
            <b>LinkedIn:</b> <a href='https://www.linkedin.com/in/sakshi-ubale-0754bb258' target='_blank' style='color:#FFD700;'>https://www.linkedin.com/in/sakshi-ubale-0754bb258</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- Center Big Image1 -----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write("")  # Empty
with col2:
    st.image("upi_image_1.jpeg", width=700)
with col3:
    st.write("")  # Empty

st.markdown("---")

# ----------------- Tabs -----------------
tab1, tab2 = st.tabs(["📝 Manual Entry", "📂 Upload CSV"])

# ----------------- Tab 1: Manual Entry -----------------
with tab1:
    st.subheader("Enter Transaction Details")
    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            TransactionAmount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=1000.0)
            CustomerAge = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
            LoginAttempts = st.number_input("Login Attempts", min_value=0, value=1)
            AccountBalance = st.number_input("Account Balance (₹)", min_value=0.0, value=5000.0)

        with col2:
            TransactionDuration = st.number_input("Transaction Duration (seconds)", min_value=0.0, value=10.0)
            tx_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
            tx_dayofweek = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
            time_since_prev = st.number_input("Time since previous transaction (seconds)", min_value=0.0, value=3600.0)

        TransactionType = st.selectbox("Transaction Type", ["Online", "POS", "ATM", "UPI"])
        Location = st.text_input("Location", "Mumbai")
        Channel = st.selectbox("Channel", ["Mobile", "Web", "Branch"])
        CustomerOccupation = st.text_input("Customer Occupation", "Engineer")

        submitted = st.form_submit_button("🔍 Predict Fraud")

    if submitted:
        input_data = pd.DataFrame([{
            "TransactionAmount": TransactionAmount,
            "CustomerAge": CustomerAge,
            "LoginAttempts": LoginAttempts,
            "AccountBalance": AccountBalance,
            "TransactionDuration": TransactionDuration,
            "tx_hour": tx_hour,
            "tx_dayofweek": tx_dayofweek,
            "time_since_prev": time_since_prev,
            "TransactionType": TransactionType,
            "Location": Location,
            "Channel": Channel,
            "CustomerOccupation": CustomerOccupation
        }])
        try:
            pred = model.predict(input_data)[0]
            if pred == -1:
                st.error("🚨 Fraudulent Transaction Detected!")
                st.balloons()
            else:
                st.success("✅ Transaction is Normal")
                st.balloons()
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------- Tab 2: CSV Upload -----------------
with tab2:
    st.subheader("Upload Transactions CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Automatically create engineered features if missing
        if "TransactionDate" in data.columns and "PreviousTransactionDate" in data.columns:
            data['TransactionDate'] = pd.to_datetime(data['TransactionDate'])
            data['PrevDate'] = pd.to_datetime(data['PreviousTransactionDate'])
            if 'tx_hour' not in data.columns:
                data['tx_hour'] = data['TransactionDate'].dt.hour
            if 'tx_dayofweek' not in data.columns:
                data['tx_dayofweek'] = data['TransactionDate'].dt.dayofweek
            if 'time_since_prev' not in data.columns:
                data['time_since_prev'] = (data['TransactionDate'] - data['PrevDate']).dt.total_seconds().fillna(0)
        else:
            st.warning("⚠ CSV missing 'TransactionDate' or 'PreviousTransactionDate'. Some features may be missing.")

        # Ensure all required columns exist
        if not all(col in data.columns for col in numeric_features + categorical_features):
            st.error(f"CSV must contain these columns: {numeric_features + categorical_features}")
        else:
            try:
                preds = model.predict(data)
                data["FraudPrediction"] = ["Fraud" if p == -1 else "Normal" for p in preds]

                # Display table with color and hover
                st.dataframe(
                    data.style.applymap(
                        lambda x: "background-color: red; color:white;" if x == "Fraud" else "background-color: lightgreen; color:black;",
                        subset=["FraudPrediction"]
                    ).set_properties(**{'transition': 'all 0.3s ease'})
                )

                # Fraud statistics
                fraud_count = (data["FraudPrediction"] == "Fraud").sum()
                normal_count = (data["FraudPrediction"] == "Normal").sum()

                st.markdown("### 📊 Fraud Statistics")
                
                # Pie chart with explode and shadow
                fig, ax = plt.subplots()
                sizes = [fraud_count, normal_count]
                colors = ["red", "green"]
                explode = (0.1, 0)
                ax.pie(sizes, labels=["Fraud", "Normal"], autopct="%1.1f%%", startangle=90, colors=colors, explode=explode, shadow=True)
                ax.axis("equal")
                st.pyplot(fig)

                # Animated cards
                st.markdown("### 📈 Transaction Summary")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        f"""
                        <div style='
                            background-color:red;
                            border-radius:15px;
                            padding:20px;
                            text-align:center;
                            color:white;
                            font-size:24px;
                            animation: fadeIn 2s ease-in-out;' >
                            🚨 Fraud Transactions<br><b>{fraud_count}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown(
                        f"""
                        <div style='
                            background-color:green;
                            border-radius:15px;
                            padding:20px;
                            text-align:center;
                            color:white;
                            font-size:24px;
                            animation: fadeIn 2s ease-in-out;' >
                            ✅ Normal Transactions<br><b>{normal_count}</b>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Failed to predict CSV: {e}")
