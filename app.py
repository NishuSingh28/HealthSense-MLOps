import streamlit as st
import pandas as pd

from src.pipelines.prediction_pipeline import (
    PredictionPipeline,
)

st.set_page_config(
    page_title="HealthSense MLOps",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #021024 0%,
        #052659 30%,
        #0B2E59 60%,
        #5483B3 100%
    );
    color: white;
}

/* Main Container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Headings */
h1 {
    color: #FFFFFF;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
}

h2, h3 {
    color: #DCEBFF;
    font-weight: 700;
}

/* Glass Cards */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.35);
}

/* Inputs */
.stNumberInput input {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.15);
}

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Labels */
label {
    color: #EAF4FF !important;
    font-weight: 600 !important;
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #7FD8FF,
        #A1C4FD
    );
    color: #021024;
    border: none;
    border-radius: 14px;
    height: 3.2em;
    font-size: 20px;
    font-weight: 700;
    transition: 0.3s;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(
        90deg,
        #A1C4FD,
        #C2E9FB
    );
}

/* Success Box */
.stSuccess {
    background-color: rgba(0,255,170,0.12);
    border-radius: 14px;
    padding: 1rem;
    border: 1px solid rgba(0,255,170,0.25);
}

/* Divider */
hr {
    border: 1px solid rgba(255,255,255,0.08);
}

</style>
""",
    unsafe_allow_html=True,
)

st.title("🌊 HealthSense MLOps")
st.markdown(
    "<h3 style='text-align:center;'>AI-Powered Obesity Risk Prediction Dashboard</h3>",
    unsafe_allow_html=True,
)

st.markdown("---")

with st.form("prediction_form"):

    st.subheader("👤 Personal Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

    with col2:
        Age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
        )

    with col3:
        Height = st.number_input(
            "Height (m)",
            min_value=1.0,
            max_value=2.5,
            value=1.75,
        )

    with col4:
        Weight = st.number_input(
            "Weight (kg)",
            min_value=20.0,
            max_value=300.0,
            value=80.0,
        )

    st.markdown("---")

    st.subheader("🍔 Lifestyle Information")

    col5, col6, col7, col8 = st.columns(4)

    with col5:

        family_history_with_overweight = (
            st.selectbox(
                "Family History",
                ["yes", "no"],
            )
        )

        FAVC = st.selectbox(
            "High Calorie Food",
            ["yes", "no"],
        )

    with col6:

        FCVC = st.number_input(
            "Vegetable Intake",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
        )

        NCP = st.number_input(
            "Main Meals",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
        )

    with col7:

        CAEC = st.selectbox(
            "Snacking",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always",
            ],
        )

        CALC = st.selectbox(
            "Alcohol",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always",
            ],
        )

    with col8:

        CH2O = st.number_input(
            "Water Intake",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
        )

        FAF = st.number_input(
            "Physical Activity",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
        )

    st.markdown("---")

    st.subheader("🚬 Additional Habits")

    col9, col10, col11 = st.columns(3)

    with col9:

        SMOKE = st.selectbox(
            "Smoking",
            ["yes", "no"],
        )

    with col10:

        SCC = st.selectbox(
            "Calorie Monitoring",
            ["yes", "no"],
        )

    with col11:

        TUE = st.number_input(
            "Technology Usage",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
        )

    st.markdown("---")

    MTRANS = st.selectbox(
        "Transportation Type",
        [
            "Public_Transportation",
            "Walking",
            "Automobile",
            "Motorbike",
            "Bike",
        ],
    )

    submitted = st.form_submit_button(
        "🔍 Predict Obesity Risk"
    )

if submitted:

    input_data = pd.DataFrame(
        {
            "Gender": [Gender],
            "Age": [Age],
            "Height": [Height],
            "Weight": [Weight],
            "family_history_with_overweight": [
                family_history_with_overweight
            ],
            "FAVC": [FAVC],
            "FCVC": [FCVC],
            "NCP": [NCP],
            "CAEC": [CAEC],
            "SMOKE": [SMOKE],
            "CH2O": [CH2O],
            "SCC": [SCC],
            "FAF": [FAF],
            "TUE": [TUE],
            "CALC": [CALC],
            "MTRANS": [MTRANS],
        }
    )

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(
        input_data
    )

    st.markdown("---")

    st.success(
        f"### Predicted Obesity Level: {prediction[0]}"
    )