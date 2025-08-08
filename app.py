import pandas as pd

inp = pd.read_csv("Input.csv")

import joblib

model = joblib.load("Pickles/knn.pkl")
sc = joblib.load("Pickles/sc.pkl")

######################## UI ######################################

# pip install streamlit (In Anaconda Prompt or Command Prompt Connected With Python)

import streamlit as st

st.subheader(":green[Animal Adaptation Study and Prediction 🦝....]")
st.write("---")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("https://assets.ltkcontent.com/images/5359/adaptable-meerkats_27c5571306.jpg")
st.write("ML Applied on Taken Dataset:")
st.write("Below is the sample Dataset:")
st.dataframe(inp.head())
st.write("---")
st.subheader(":red[Enter Animal Details to Get a Adaptation Signal:]")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    spec = st.selectbox("Select Species:", inp.Species.unique())
with col2:
    obtime = st.selectbox("Select Observation Time:", inp.Observation_Time.unique())
with col3:
    loc = st.selectbox("Select Location Type:", inp.Location_Type.unique())

col4, col5, col6 = st.columns([1,1,1])
with col4:
    ndb = st.number_input("Enter Noise Level in DB:", min_value=inp.Noise_Level_dB.min(), max_value=inp.Noise_Level_dB.max())
with col5:
    hd = st.number_input("Enter Human Density:", min_value=inp.Human_Density.min(), max_value=inp.Human_Density.max())
with col6:
    fss = st.number_input("Enter Food Source Score:", min_value=inp.Food_Source_Score.min(), max_value=inp.Food_Source_Score.max())

col7, col8, col9 = st.columns([1,1,1])
with col4:
    sqs = st.number_input("Enter Shelter Quality Score:", min_value=inp.Shelter_Quality_Score.min(), max_value=inp.Shelter_Quality_Score.max())
with col5:
    bas = st.number_input("Enter Behaviour Anamoly Score:", min_value=inp.Behavior_Anomaly_Score.min(), max_value=inp.Behavior_Anomaly_Score.max())
with col6:
    edd = st.number_input("Enter Estimated Daily Distance in Km", min_value=inp.Estimated_Daily_Distance_km.min(), max_value=inp.Estimated_Daily_Distance_km.max())

if st.button("Predict"):
    row = pd.DataFrame([[spec,obtime,loc,ndb,hd,fss,sqs,bas,edd]], columns=inp.columns)
    st.write("Given Data:")
    st.dataframe(row)
    
    # Pre-Processing Code
    row.replace({"Fox":4,"Pigeon":3,"Squirrel":2,"Raccoon":1,
          "Morning":4,"Afternoon":3,"Evening":2,"Night":1,
          "Residential":4,"Commercial":3,"Park":2,"Industrial":1}, inplace=True)
    
    row = sc.transform(row)
    
    # Prediction
    probs = [round(prob,2) for prob in model.predict_proba(row)[0]]
    classprobs = {k:v for k, v in zip(model.classes_, probs)}
    st.write("Predicted Probabilities:")
    st.write(classprobs)
    out = model.predict(row)[0]
    st.subheader(f":green[Predicted Adaptation Signal: :red[{out}]]")
    st.balloons()