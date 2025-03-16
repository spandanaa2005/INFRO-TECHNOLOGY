import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def load_data(filepath):
    """Loads and preprocesses the disease symptom dataset."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"File not found: {filepath}. Please ensure the file exists.")
        return None
    except PermissionError:
        st.error(f"Permission denied: {filepath}. Please check file permissions.")
        return None

def preprocess_data(df):
    """Preprocesses the data, handles missing values, and encodes labels."""

    if df is None:
        return None, None, None, None

    # Handle missing values (replace with 0 for simplicity, adjust as needed)
    df = df.fillna(0)

    # Separate features and target
    X = df.drop('diseases', axis=1)
    y = df['diseases']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    """Trains a Random Forest Classifier model."""
    if X_train is None or y_train is None:
        return None
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_disease(model, input_symptoms, feature_names, label_encoder):
    """Predicts disease based on user input symptoms."""
    if model is None or input_symptoms is None or feature_names is None or label_encoder is None:
        return None, None

    input_data = pd.DataFrame(0, index=[0], columns=feature_names)
    for symptom in input_symptoms:
        if symptom in feature_names:
            input_data[symptom] = 1

    prediction = model.predict(input_data)
    confidence = model.predict_proba(input_data)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease, confidence[0][prediction[0]]

def main():
    """Main function to run the Streamlit app."""
    st.title("AI-Powered Disease Prediction")

    filepath = "C:\\Users\\DELL\\OneDrive\\Desktop\\infro tech\\Disease_symptom_and_treatment.csv" # Replace with your dataset path.
    df = load_data(filepath)

    if df is not None:
        X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
        model = train_model(X_train, y_train)

        if model is not None:
            feature_names = X_train.columns.tolist()
            symptoms = st.multiselect("Enter your symptoms:", feature_names)

            if st.button("Predict"):
                if symptoms:
                    predicted_disease, confidence = predict_disease(model, symptoms, feature_names, label_encoder)
                    st.write(f"Predicted Disease: {predicted_disease}")
                    st.write(f"Confidence Score: {confidence:.4f}")
                else:
                    st.warning("Please enter at least one symptom.")
        else:
            st.error("Model training failed.")
    else:
        st.error("Data loading failed.")

if __name__ == "__main__":
    main()