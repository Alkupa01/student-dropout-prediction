import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL & METADATA
# ================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    label_map = joblib.load("label_map.pkl")

    # Ambil kolom langsung dari model
    columns = list(model.feature_names_in_)

    return model, columns, label_map

model, columns, label_map = load_artifacts()

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Dropout Prediction",
    layout="wide"
)

# UI HEADER
st.title("🎓 Student Academic Outcome Prediction")
st.write("""
This system predicts **student academic status** using Machine Learning.  
The prediction is intended as an **early warning system**, **not a final decision**.
""")

st.markdown("---")

# CSV UPLOAD 
# ================================
st.subheader("📂 Upload CSV")

uploaded_file = st.file_uploader(
    "Upload preprocessed CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🚀 Run Prediction"):

        # Samakan kolom dengan model
        df_full = pd.DataFrame(0, index=df.index, columns=columns)

        for col in df.columns:
            if col in df_full.columns:
                df_full[col] = df[col]

        preds = model.predict(df_full)
        probs = model.predict_proba(df_full).max(axis=1)

        st.success("Prediction completed!")

        # OUTPUT TABLE & DOWNLOAD
        output_df = pd.DataFrame({
            "Student": df["Name"] if "Name" in df.columns else [f"Student {i+1}" for i in range(len(preds))],
            "Prediction": preds,
            "Confidence": [f"{p*100:.2f}%" for p in probs]
        })

        st.subheader("📊 Prediction Results")
        st.dataframe(output_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Results",
            output_df.to_csv(index=False),
            file_name="student_prediction_results.csv",
            mime="text/csv"
        )

# FOOTER
st.markdown("---")
st.caption(
    "⚠️ This system is an early warning tool and should not be used as the sole basis for academic decisions."
)
