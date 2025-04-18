# Mental Health Prediction: AI-Powered Depression Risk Predictor

## Overview

The **Mental Health Prediction** web application uses advanced Artificial Intelligence (AI) techniques to predict the likelihood of depression based on survey responses. By leveraging a trained Neural Network model, the app analyzes user data and provides insights into mental health conditions, specifically depression risk.

This application is designed for use by individuals, healthcare professionals, and organizations who aim to assess mental health risks based on user-provided survey data. The system offers features for both manual input and CSV data upload for seamless prediction and analysis.

---

## Features

- **Upload Data** – Upload survey data in CSV format for comprehensive analysis and depression risk prediction.
- **Manual Entry** – Manually input survey responses and instantly get depression risk predictions.
- **Preprocessing & Cleaning** – Automatically cleans and preprocesses the uploaded data to ensure consistency and readiness for model prediction.
- **AI-Powered Predictions** – Utilizes a Neural Network model to accurately predict the likelihood of depression based on survey responses.
- **Data Visualizations** – Visualize insights such as correlation heatmaps and depression case distribution for better understanding of the data and results.

---

## Project Structure
📦 Mental Health Prediction  
┃   
┣ 📂 **Pickle**   
┃  ┣ 📜 `neural_network.keras` -> *Trained TensorFlow Model*  
┃  ┣ 📜 `preprocessor.pkl` -> *Preprocessing pipeline*  
┃  ┣ 📜 `cleaning.pkl` -> *Data cleaning logic*  
┃  
┣ 📂 **Data**   
┃  ┣ 📜 `train.csv` -> *Train dataset*  
┃  ┣ 📜 `test.csv` -> *Test dataset*  
┃  ┣ 📜 `sample_submission.csv` -> *Sample Survey dataset*  
┃  ┣ 📜 `cleaned_data.csv` -> *Cleaned Dataset*  
┃  ┣ 📜 `preprocessor_data.csv` -> *Preprocessed Dataset* 
┃  
┣ 📂 **Scripts**   
┃  ┣ 📜 `data_understanding.ipynb` -> *Helps in Finding and Understanding Dataset's Characteristic*  
┃  ┣ 📜 `data_processing.ipynb` -> *Prepares Data for Model Building and Prediction*   
┃  ┣ 📜 `EDA.ipynb` -> *Visualizes data for the understanding*  
┃  ┣ 📜 `model_building.ipynb` -> *Builds Neural Networks Model and Prepares for the Prediction*  
┃  
┣ 📜 app.py -> *Streamlit Application*  
┣ 📜 requirements.txt -> *Python dependencies*    
┣ 📜 environments.yml -> *conda dependencies*   
┣ 📜 README.md -> *Documentation*   


---

## Visualizations & Insights

- **Correlation Heatmaps** – Explore relationships between features using correlation heatmaps.
- **Depression Distribution** – Visualize the distribution of depression cases within the dataset.
- **Real-Time Predictions** – Get instant depression risk predictions for manually entered data or uploaded CSVs.

---

## Technologies Used

- **Streamlit** – A web application framework for building interactive data apps.
- **TensorFlow/Keras** – Deep learning libraries used to train the Neural Network model for depression prediction.
- **Pandas & NumPy** – Powerful data processing and transformation tools.
- **Matplotlib & Seaborn** – Visualization libraries for generating insights and graphical representations of the data.

---

## Installation & Setup

### Step 1: Clone the repository
git clone https://github.com/ShreeChander/Mental_Health_Survey.git


### Step 2: Create a virtual environment
conda create --name mental_health python=3.8 conda activate mental_health


### Step 3: Install dependencies
Install the necessary Python packages using the following command:
pip install -r requirements.txt

### Step 4: Run the Streamlit app
To run the app locally, use the following command:
streamlit run Streamlit.py

---

## How It Works

1. **Data Input** – Upload survey data (CSV format) or manually enter survey responses.
2. **Data Preprocessing** – The uploaded data is automatically cleaned and transformed using predefined preprocessing logic.
3. **Model Prediction** – The cleaned and preprocessed data is passed through a Neural Network model trained on historical survey data, which outputs the likelihood of depression.
4. **Visualizations** – The app displays insightful visualizations, including correlation heatmaps and depression distribution charts, to help users understand the results.
5. **Depression Risk** – The app provides a risk prediction based on the user data, assisting in early intervention.


---

## Final Thoughts

Depression is a serious but treatable condition, and early identification can significantly improve outcomes. This application helps in identifying potential depression risks, offering users a tool to take control of their mental health and seek professional support when needed. If you or someone you know is struggling with depression, please reach out to a mental health professional. Remember, you're not alone.

**Helpline (India):** 1-800-599-0019 (Available 24/7)
"""

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
