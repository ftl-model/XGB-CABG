
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('XGB.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature names
feature_names = [ "bmi", " EF ", "LVDs ", " preop_max_Cr ", "lymphocyte", "Triiodothyronine", "direct_bilirubin ", " BP_55_time", "Lac"]

## Streamlit user interface
st.title("LCOS Predictor")

bmi=st.number_input("BMI:",min_value=0.00,max_value=100.00,value=25.00)
EF=st.number_input("EF:",min_value=0,max_value=150,value=62)
LVDs=st.number_input("LVDs:",min_value=0.00,max_value=100.00,value=41.00)
preop_max_Cr=st.number_input("Creatinine (μmol/L):",min_value=0.00,max_value=500.00,value=141.40)
lymphocyte=st.number_input("Lymphocyte (109/L):",min_value=0.00,max_value=100.00,value=19.10)
Triiodothyronine=st.number_input("Triiodothyronine (nmol/L):",min_value=0.00,max_value=10.00,value=1.25)
direct_bilirubin=st.number_input("Direct bilirubin (μmol/L):",min_value=0.00,max_value=50.00,value=1.90)
BP_55_time=st.number_input("Time for MAP <55 (min):",min_value=0.00,max_value=1000.00,value=75.00)
Lac=st.number_input("Lactate (mmol/L):",min_value=0.00,max_value=50.00,value=6.70)

# Process inputs and make predictions
feature_values = [ bmi,  EF , LVDs ,  preop_max_Cr , lymphocyte, Triiodothyronine, direct_bilirubin ,  BP_55_time, Lac]
features = np.array([feature_values])

if st.button("Predict"):    
    # 标准化特征
    standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:       
        advice = (
            f"According to our model, you have a high risk of heart disease. "          
            f"The model predicts that your probability of having LCOS is {probability:.1f}%. "            
            "While this is just an estimate, it suggests that you may be at significant risk. "           
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )   
    else:        
        advice = (
            f"According to our model, you have a low risk of heart disease. "            
            f"The model predicts that your probability of not having heart disease is {probability:.1f}%. "            
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your heart health, "            
            "and to seek medical advice promptly if you experience any symptoms." 
        )
        
    st.write(advice)

# Calculate SHAP values and display force plot 
    st.subheader("SHAP Force Plot Explanation") 
    explainer = shap.TreeExplainer(model) 
   
    shap_values = explainer.shap_values(pd.DataFrame(standardized_features, columns=feature_names))
   # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    shap.force_plot(explainer.expected_value, shap_values[0], original_feature_values, matplotlib=True)   
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
