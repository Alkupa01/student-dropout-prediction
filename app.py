import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# LOAD MODEL & METADATA
# ================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    label_map = joblib.load("label_map.pkl")

    # ⬇️ AMBIL KOLOM LANGSUNG DARI MODEL
    columns = list(model.feature_names_in_)

    return model, columns, label_map   

model, columns, label_map = load_artifacts() 


# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Student Dropout Prediction",
    layout="wide"
)

# ================================
# HELPER FUNCTIONS
# ================================
def prepare_input(user_input: dict, columns: list):
    df = pd.DataFrame([user_input])
    df = df.reindex(columns=columns, fill_value=0)
    return df


def risk_message(label, prob):
    if label == "Dropout":
        if prob >= 0.7:
            return "🔴 High Risk: Immediate academic intervention recommended."
        else:
            return "🟠 Medium Risk: Monitor student performance closely."
    elif label == "Enrolled":
        return "🟡 Medium Risk: Student is progressing but requires monitoring."
    else:
        return "🟢 Low Risk: Student is likely to graduate successfully."


# ================================
# UI HEADER
# ================================
st.title("🎓 Student Academic Outcome Prediction")
st.write("""
This system predicts **student academic status** using Machine Learning.  
The prediction is intended as an **early warning system**, **not a final decision**.
""")

# ================================
# SIDEBAR
# ================================
st.sidebar.header("Prediction Mode")
mode = st.sidebar.radio("Choose input method:", ["Single Student", "Upload CSV"])

# ================================
# SINGLE STUDENT MODE
# ================================
if mode == "Single Student":

    st.subheader("📌 Single Student Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age at enrollment", min_value=15, max_value=60, value=18)
        gender = st.selectbox("Gender", ["Male", "Female"])
        scholarship = st.selectbox("Scholarship holder", ["Yes", "No"])

    with col2:
        tuition = st.selectbox("Tuition fees up to date", ["Yes", "No"])
        displaced = st.selectbox("Displaced", ["Yes", "No"])
        special_needs = st.selectbox("Educational special needs", ["Yes", "No"])

    if st.button("🔍 Predict"):

        # ================================
        # USER INPUT MAPPING
        # ================================
        user_input = {
            "Age at enrollment": age,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Scholarship holder": 1 if scholarship == "Yes" else 0,
            "Tuition fees up to date": 1 if tuition == "Yes" else 0,
            "Displaced": 1 if displaced == "Yes" else 0,
            "Educational special needs": 1 if special_needs == "Yes" else 0
        }

        df_input = prepare_input(user_input, columns)

        # ================================
        # PREDICTION
        # ================================
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]

        label = prediction
        confidence = np.max(probabilities)

        # ================================
        # OUTPUT
        # ================================
        st.success(f"🎯 Prediction Result: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.info(risk_message(label, confidence))


# ================================
# CSV UPLOAD MODE
# ================================
else:
    st.subheader("📂 Batch Prediction (Upload CSV)")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🚀 Run Prediction"):

            # Samakan kolom
            df_full = pd.DataFrame(0, index=df.index, columns=columns)

            for col in df.columns:
                if col in df_full.columns:
                    df_full[col] = df[col]

            preds = model.predict(df_full)
            probs = model.predict_proba(df_full).max(axis=1)

            df["Prediction"] = preds
            df["Confidence"] = probs

            st.success("Prediction completed!")
            st.dataframe(df)

            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                file_name="student_prediction_results.csv",
                mime="text/csv"
            )


# ================================
# FOOTER
# ================================
st.markdown("---")
st.caption(
    "⚠️ This system is an early warning tool and should not be used as the sole basis for academic decisions."
)
