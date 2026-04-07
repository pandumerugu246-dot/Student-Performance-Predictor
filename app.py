import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# TITLE
# -------------------------------
st.title("🎓 Student Performance Predictor (Advanced)")

# -------------------------------
# DATASET
# -------------------------------
data = {
    "Name": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "DOB": ["2000-05-10", "2001-07-15", "1999-08-20", "2002-01-25",
            "2000-12-12", "2001-03-30", "1998-11-05", "2002-06-18"],
    "StudyHours": [2,4,3,5,6,4,7,8],
    "SleepHours": [7,6,8,5,7,6,8,5],
    "PreviousScore": [55,65,60,70,75,68,80,85],
    "FinalScore": [58,68,63,73,78,70,83,88]
}

df = pd.DataFrame(data)

st.subheader("Dataset Preview")
st.dataframe(df)

# -------------------------------
# PREPROCESSING
# -------------------------------
st.subheader("Data Preprocessing")

# Convert DOB to datetime
df["DOB"] = pd.to_datetime(df["DOB"])

# Create Age
current_year = 2026
df["Age"] = current_year - df["DOB"].dt.year

# Extract Birth Month
df["BirthMonth"] = df["DOB"].dt.month

# Check missing values
st.write("Missing Values:")
st.write(df.isnull().sum())

# Check duplicates
st.write("Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Drop unnecessary columns
df = df.drop(["Name", "DOB"], axis=1)

st.write("Processed Data:")
st.dataframe(df)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
st.subheader("Feature Engineering")

df["TotalEffort"] = df["StudyHours"] + df["SleepHours"]

st.write("After Feature Engineering:")
st.dataframe(df)

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("Data Visualization")

# Study vs Score
fig1 = plt.figure()
plt.scatter(df["StudyHours"], df["FinalScore"])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study vs Score")
st.pyplot(fig1)

# Age vs Score
fig2 = plt.figure()
plt.scatter(df["Age"], df["FinalScore"])
plt.xlabel("Age")
plt.ylabel("Final Score")
plt.title("Age vs Score")
st.pyplot(fig2)

# Histogram
fig3 = plt.figure()
plt.hist(df["FinalScore"])
plt.title("Score Distribution")
st.pyplot(fig3)

# Correlation
st.subheader("Correlation Matrix")
st.dataframe(df.corr())

# -------------------------------
# OUTLIER DETECTION
# -------------------------------
st.subheader("Outlier Detection")

fig4 = plt.figure()
plt.boxplot(df["FinalScore"])
plt.title("Outliers in Final Score")
st.pyplot(fig4)

# -------------------------------
# MODEL TRAINING
# -------------------------------
st.subheader("Model Training")

X = df.drop("FinalScore", axis=1)
y = df["FinalScore"]

# Train-Test Split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.success("Model Trained Successfully")

# -------------------------------
# EVALUATION
# -------------------------------
st.subheader("Model Evaluation")

y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# st.write("MAE:", mae)
# st.write("MSE:", mse)
# st.write("RMSE:", rmse)

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("Predict Score")

study_hours = st.slider("Study Hours", 0.0, 12.0, 5.0)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
previous_score = st.slider("Previous Score", 0, 100, 60)
age = st.slider("Age", 15, 30, 20)
birth_month = st.slider("Birth Month", 1, 12, 6)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    total_effort = study_hours + sleep_hours

    input_data = np.array([[study_hours, sleep_hours, previous_score, age, birth_month, total_effort]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Score: {prediction[0]:.2f}")