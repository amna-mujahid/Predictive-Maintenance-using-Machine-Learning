import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)


# --------------------------------------------------
# LOAD TRAINED MODEL AND ENCODERS
# --------------------------------------------------

model = joblib.load("random_forest_model (1).pkl")
label_encoder = joblib.load("label_encoder.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
feature_columns = joblib.load("feature_columns (1).pkl")


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚙️ Machine Failure Type Prediction")

st.write(
    "Predict the type of machine failure using "
    "operating conditions such as temperature, "
    "rotational speed, torque, tool wear, and product type."
)

st.divider()


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("Enter Machine Parameters")

col1, col2 = st.columns(2)


with col1:

    product_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"]
    )

    air_temperature = st.number_input(
        "Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature [K]",
        min_value=305.0,
        max_value=315.0,
        value=310.0,
        step=0.1
    )


with col2:

    rotational_speed = st.number_input(
        "Rotational Speed [rpm]",
        min_value=1000,
        max_value=3000,
        value=1500,
        step=10
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=80.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )


st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Predict Failure Type", type="primary"):

    # Encode Product Type using the SAME encoder
    # that was used during model training

    encoded_type = ordinal_encoder.transform(
        [[product_type]]
    )[0][0]


    # Create input dataframe

    input_data = pd.DataFrame({

        "Type": [encoded_type],

        "Air temperature [K]": [
            air_temperature
        ],

        "Process temperature [K]": [
            process_temperature
        ],

        "Rotational speed [rpm]": [
            rotational_speed
        ],

        "Torque [Nm]": [
            torque
        ],

        "Tool wear [min]": [
            tool_wear
        ]
    })


    # Make sure columns are in exactly the
    # same order as during model training

    input_data = input_data[feature_columns]


    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    prediction_encoded = model.predict(input_data)


    # Convert numerical prediction back
    # to original failure name

    prediction = label_encoder.inverse_transform(
        prediction_encoded
    )[0]


    # --------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------

    st.subheader("Prediction Result")


    if prediction == "No Failure":

        st.success(
            f"🟢 Predicted Failure Type: {prediction}"
        )

    else:

        st.error(
            f"🔴 Predicted Failure Type: {prediction}"
        )


    # --------------------------------------------------
    # PREDICTION PROBABILITIES
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]


        probability_df = pd.DataFrame({

            "Failure Type": label_encoder.classes_,

            "Probability": probabilities

        }).sort_values(
            by="Probability",
            ascending=False
        )


        st.subheader("Prediction Probabilities")


        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)


        probability_df = probability_df.rename(
            columns={
                "Probability": "Probability (%)"
            }
        )


        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )