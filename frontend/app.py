import streamlit as st
import requests
import pandas as pd
import time
import io

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color:  #d5d7db; /* Light grey */
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("📊 Step Count Prediction System")


# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file (Date,StepCount)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📌 Uploaded Data Preview:")
    st.write(df.head())

    if st.button("Train Model"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/upload_csv", files=files)

        if response.status_code == 200:
            st.success("✅ Model training started!")
            
            # Check training status
            training_complete = False
            while not training_complete:
                time.sleep(2)  
                status_response = requests.get("http://127.0.0.1:5000/training_status")

                if status_response.status_code == 200:
                    status = status_response.json()["status"]
                    st.info(status)

                    if status == "Training Completed!":
                        training_complete = True
                        st.success("🎉 Model training completed successfully!")

        else:
            st.error("❌ Error: " + response.text)


st.subheader("🤔Confused?")
def get_sample_csv():
    sample_data = pd.DataFrame({
        "Date": ["2025-03-01", "2025-03-02", "2025-03-03"],
        "StepCount": [5000, 7000, 8000]
    })
    csv_buffer = io.StringIO()
    sample_data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

# Download button for the sample CSV
st.download_button(
    label="Download Sample CSV",
    data=get_sample_csv(),
    file_name="sample.csv",
    mime="text/csv"
)


# Display Weekly & Monthly Trends
st.header("📈 Step Count Trends")
if st.button("Show Weekly Trend"):
    st.image("http://127.0.0.1:5000/weekly_trend", caption="Weekly Step Count Trend")

if st.button("Show Monthly Trend"):
    st.image("http://127.0.0.1:5000/monthly_trend", caption="Monthly Step Count Trend")

# Prediction Section
st.header("🔮 Predict Future Step Count")
date_input = st.date_input("Select a date for prediction")


if st.button("Predict Steps"):
    response = requests.post("http://127.0.0.1:5000/predict", json={"date": str(date_input)})

    if response.status_code == 200:
        result = response.json()
        st.write("📊 Predicted Steps:", result["predicted_steps"])
        st.write("🏃‍♂️ Activity Status:", result["activity_status"])
        st.write("💡 Health Tip:", result["health_tips"])
    else:
        st.write("❌ Error:", response.json()["error"])


if st.button("🗑️ Clear Data Before Exit"):
    response = requests.post("http://127.0.0.1:5000/clear_data")
    if response.status_code == 200:
        st.success("✅ All uploaded data has been cleared!")

        # Reset displayed data
        uploaded_file = None  # Remove uploaded file
        df = None  # Remove dataframe preview
        date_input = None  # Clear date input
        st.rerun()  # Refresh the UI
    else:
        st.error("❌ Failed to clear data!")

