import base64
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

def img_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img_data = img_file.read()
        return base64.b64encode(img_data).decode()

image_path = "Image/Neural_Networks.webp"

try:
    img_base64 = img_to_base64(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('data:image/jpeg;base64,{img_base64}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            padding: 0;
        }}
        .title-container {{
            text-align: center;
            color: white;
            font-size: 4em;
            margin-top: 350px;
            margin-bottom: 20px;
            
        }}
        
        </style>
        """, unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("Image not found at the specified path.")

class MentalHealthPreprocessor:

    def __init__(self):

        self.encoders = {}
        self.columns_to_drop = []
        self.numerical_features = None
        self.categorical_features = None

    def fit(self, data):

        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = data.select_dtypes(include=['object']).columns.tolist()

        corr_matrix = data[self.numerical_features].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.columns_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.6)]
        data = data.drop(columns=self.columns_to_drop, errors='ignore')
        
        self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        for col in self.categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le  

        return data

    def ensure_all_features(self, data):

        for col in self.numerical_features:
            if col not in data.columns:
                data[col] = 0
                
        for col in self.categorical_features:
            if col not in data.columns and col in self.encoders:
                data[col] = 0
                
        return data[self.numerical_features + [c for c in self.categorical_features if c in self.encoders]]

    def transform(self, data):
        data = data.drop(columns=self.columns_to_drop, errors='ignore')

        for col in self.categorical_features:
            if col in data.columns and col in self.encoders:
                try:
                    data[col] = self.encoders[col].transform(data[col].astype(str))

                except ValueError as e:
                    print(f"Error transforming {col}: {str(e)}")
                    data[col] = 0

        return self.ensure_all_features(data)
    
    

def safe_load_file(file_path, default=None):
    try:
        with open(file_path, "rb") as file:

            return pickle.load(file)
        
    except (FileNotFoundError, pickle.PickleError) as e:
        st.error(f"Error loading file from {file_path}: {str(e)}")

        return default

@st.cache_resource
def load_preprocessor():

    return safe_load_file("Model/preprocessor.pkl")
    
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Model/neural_network.keras")

        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

        return None

@st.cache_resource
def load_cleaning():

    return safe_load_file("Model/cleaning.pkl", pd.DataFrame())

def predict_depression(input_data, model, preprocessor):

    try:
        processed_data = preprocessor.transform(pd.DataFrame([input_data]))
        
        expected_features = 13  
        current_features = processed_data.shape[1]
        
        if current_features < expected_features:
            missing_features = expected_features - current_features
            st.warning(f"Missing {missing_features} features. Adding zero padding.")
            processed_data = np.pad(processed_data, ((0, 0), (0, missing_features)), mode='constant')

        elif current_features > expected_features:
            st.warning(f"Too many features ({current_features}). Truncating to {expected_features}.")
            processed_data = processed_data[:, :expected_features]
        
        processed_data = np.array(processed_data, dtype=np.float32)
        output = model.predict(processed_data)
        prediction = "Depression Detected" if output[0] > 0.5 else "No Depression"

        return prediction, output[0][0]
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        
        return "Prediction Error", 0.0

st.title("Mental Health Depression Prediction Application")
st.sidebar.header("Navigation Bar ->")
page = st.sidebar.radio("Pages", ["Home", "Upload Data", "Manual Entry", "Visualizations", "Bias_Evaluation"])

if page == "Home":
    st.write("#### Predicting Depression using Artificial Intelligence")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("##### This app allows users to predict the likelihood of depression based on survey responses.")
    st.write("##### Depression is a mental health disorder that affects millions worldwide, impacting mood, energy levels, and daily life.")
    st.write("##### It can range from mild sadness to severe clinical depression, requiring professional intervention.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### Final Thought -")
    st.write("##### Depression is a serious but treatable condition. Small daily actions can improve mental health and prevent depression.")
    st.write("##### If you or someone you know struggles with depression, reach out for support always —> you're not alone !!!")

if "predicted_data" not in st.session_state:
    st.session_state["predicted_data"] = pd.DataFrame()

if page == "Upload Data":

    st.header("Upload Survey Data")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(data.head())
        
        cleaned_data = load_cleaning()
        st.write("### Cleaned Data Preview")
        st.dataframe(cleaned_data.head())
        
        preprocessor = load_preprocessor()
        model = load_model()
        
        preprocessed_data = preprocessor.transform(cleaned_data.copy())

        expected_features = 13
        if preprocessed_data.shape[1] > expected_features:
            preprocessed_data = preprocessed_data.iloc[:, :expected_features]

        if st.button("Predict Depression Risk"):
            predictions = model.predict(preprocessed_data)
            prediction_labels = [1 if pred > 0.5 else 0 for pred in predictions]
            confidence_scores = [round(float(pred), 2) for pred in predictions]

            st.write("#### Prediction Labels \n ##### 1 -> Depression Detected \n ##### 0 -> No Depression")
            
            result_df = cleaned_data.copy()
            result_df["Prediction"] = prediction_labels
            result_df["Confidence Score"] = confidence_scores

            st.session_state["predicted_data"] = result_df
            st.write("#### Prediction Results")
            st.dataframe(result_df.head())

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        
elif page == "Manual Entry":

    st.header("Manual Data Entry")

    model = load_model()
    preprocessor = load_preprocessor()
    
    working_status = st.selectbox("Working Professional or Student", ["Professional", "Student"])

    sleep = ["6-8 hours", "Less than 5 hours", "5-6 hours", "More than 8 hours"]

    city = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Ahmedabad", "Chennai", "Kolkata", "Pune", "Jaipur", "Lucknow",
            "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Vasai-Virar", "Ghaziabad", "Ludhiana",
            "Agra", "Nashik", "Faridabad", "Meerut", "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar", "Kalyan"]
    
    profession = ["Teacher", "Content Writer", "Architect", "Consultant", "HR Manager", "Pharmacist", "Doctor", "Business Analyst", "Entrepreneur",
                   "Chemist", "Educational Consultant", "Chef", "Data Scientist", "Researcher", "Lawyer", "Customer Support", "Marketing Manager", "Pilot", "Travel Consultant",
                   "Plumber", "Sales Executive", "Manager", "Judge", "Electrician", "Financial Analyst", "Software Engineer", "Civil Engineer", "UX/UI Designer", "Digital Marketer", 
                   "Accountant", "Finanancial Analyst", "Mechanical Engineer", "Graphic Designer", "Research Analyst", "Investment Banker", "Family Consultant", "Dev", "Analyst"]
    
    degree = ["Class 12", "B.Ed", "B.Arch", "B.Com", "BCA", "B.Pharm", "M.Ed", "MCA", "BBA", "BSc", "MSc", "LLM", "M.Tech", "B.Tech", "M.Pharm", "LLB", "BHM", "MBA", "BA",
              "ME", "MD", "MHM", "BE", "PhD", "M.Com", "MBBS", "MA", "M.Arch"]
    
    diet = ["Moderate", "Unhealthy", "Healthy"]
    
    input_data = {
        "Gender": st.selectbox("Select Gender", ["Male", "Female"]),
        "Age": st.slider("Select Age", 18, 80, 1),
        "City": st.selectbox("Select City", city),
        "Sleep Duration": st.selectbox("Select Sleep Duration", sleep),
        "Dietary Habits": st.selectbox("Select Dietary Habits", diet),
        "Have you ever had suicidal thoughts ?": st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"]),
        "Work/Study Hours": st.slider("Enter Work/Study Hours", 0, 24, 1),
        "Financial Stress": st.slider("Financial Stress", 0, 5, 1),
        "Family History of Mental Illness": st.selectbox("Family History of Mental Illness", ["Yes", "No"]),
    }
    
    if working_status == "Professional":
        input_data.update({
            "Profession": st.selectbox("Enter Profession", profession),
            "Work Pressure": st.slider("Work Pressure", 0, 10, 5),
            "Job Satisfaction": st.slider("Job Satisfaction", 0, 10, 5),
        })
    else:
        input_data.update({
            "Academic Pressure": st.slider("Academic Pressure", 0, 10, 5),
            "CGPA": st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.0),
            "Study Satisfaction": st.slider("Study Satisfaction", 0, 10, 5),
            "Degree": st.selectbox("Enter Degree", degree),
        })
    
    if st.button("Predict Depression Risk"):
        prediction, confidence = predict_depression(input_data, model, preprocessor)
        st.write(f"#### Prediction: \n {prediction}")
        st.write(f"#### Confidence Score: \n {confidence:.2f}")

elif page == "Visualizations":

    st.header("Data Insights & Visualizations")

    preprocessor = load_preprocessor()
    model = load_model()
    cleaned_data = load_cleaning()
        
    preprocessed_data = preprocessor.transform(cleaned_data.copy())

    expected_features = 13

    if preprocessed_data.shape[1] > expected_features:
        preprocessed_data = preprocessed_data.iloc[:, :expected_features]
    
    if not st.session_state["predicted_data"].empty:
        st.write("#### Predicted Data Overview")
        st.dataframe(st.session_state["predicted_data"].head())

        numeric_data = st.session_state["predicted_data"].select_dtypes(include=[np.number])

        if not numeric_data.empty:
            st.write("#### Correlation Heatmap (Predicted Data)")
            plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)

        else:
            st.warning("No numeric columns available for correlation heatmap.")

        st.write("#### Depression Distribution in Predictions")

        fig, ax = plt.subplots()
        sns.countplot(x="Prediction", data=st.session_state["predicted_data"], ax=ax)
        st.pyplot(fig)

    else:
        st.warning("No predicted data available. Please run predictions in the Upload Data section first.")

elif page == 'Bias_Evaluation':

    st.header("Bias Evaluation for Prediction Distribution")
    
    if "predicted_data" in st.session_state and not st.session_state["predicted_data"].empty:
        data = st.session_state["predicted_data"].copy()
    else:
        st.warning("No predicted data available. Please run predictions in the Upload Data section first.")
        st.stop()
    
    st.write("#### Gender Distribution in Predictions")
    
    if "Gender" in data.columns and "Prediction" in data.columns:

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Gender", hue="Prediction", data=data, ax=ax)
        plt.title("Depression Predictions by Gender")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        st.pyplot(fig)
        
        st.write("#### Depression Prediction Rates by Gender")
        gender_stats = data.groupby("Gender")["Prediction"].mean().reset_index()
        gender_stats.columns = ["Gender", "Depression Prediction Rate"]
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Gender", y="Depression Prediction Rate", data=gender_stats, ax=ax2)
        plt.title("Depression Prediction Rate by Gender")
        plt.ylim(0, 1)
        st.pyplot(fig2)
        
        st.write("#### Statistical Breakdown by Gender")
        st.dataframe(gender_stats)

    else:
        st.warning("Required columns 'Gender' or 'Prediction' not found in dataset.")
    
    if "Age" in data.columns and "Prediction" in data.columns:

        st.write("#### Age Distribution in Predictions")
        
        data["Age Group"] = pd.cut(data["Age"], 
                                  bins=[17, 25, 35, 45, 55, 85], 
                                  labels=["18-25", "26-35", "36-45", "46-55", "56+"])
        
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.countplot(x="Age Group", hue="Prediction", data=data, ax=ax3)
        plt.title("Depression Predictions by Age Group")
        plt.xlabel("Age Group")
        plt.ylabel("Count")
        st.pyplot(fig3)
        
        st.write("#### Depression Prediction Rates by Age Group")
        age_stats = data.groupby("Age Group")["Prediction"].mean().reset_index()
        age_stats.columns = ["Age Group", "Depression Prediction Rate"]
        
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Age Group", y="Depression Prediction Rate", data=age_stats, ax=ax4)
        plt.title("Depression Prediction Rate by Age Group")
        plt.ylim(0, 1)
        st.pyplot(fig4)
        
        st.write("#### Statistical Breakdown by Age Group")
        st.dataframe(age_stats)

    else:
        st.warning("Required column 'Age' not found in dataset.")
    
    bias_factors = ["City", "Sleep Duration", "Dietary Habits", "Work/Study Hours"]
    
    for factor in bias_factors:

        if factor in data.columns and "Prediction" in data.columns:
            st.write(f"#### {factor} Distribution in Predictions")
            
            if data[factor].nunique() > 10:

                top_categories = data[factor].value_counts().head(10).index
                filtered_data = data[data[factor].isin(top_categories)].copy()
                st.write(f"(Showing top 10 categories out of {data[factor].nunique()})")

            else:
                filtered_data = data.copy()
            
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            sns.countplot(x=factor, hue="Prediction", data=filtered_data, ax=ax5)
            plt.title(f"Depression Predictions by {factor}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig5)
            
            factor_stats = data.groupby(factor)["Prediction"].mean().reset_index()
            factor_stats.columns = [factor, "Depression Prediction Rate"]
            
            st.write(f"#### Statistical Breakdown by {factor}")
            st.dataframe(factor_stats)

    
