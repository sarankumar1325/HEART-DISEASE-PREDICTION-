import streamlit as st
import pickle
import pandas as pd

if __name__ == '__main__':
    # $ streamlit run main.py
    st.title('Probability of Heart Disease')
    age = st.number_input('Age')
    max_heart_reate = st.number_input('Max Heart Rate')
    sex = st.text_input('Sex [M|F]')
    exercise_angina = st.text_input('Exercise Angina [Y|N]')

    model = pickle.load(open('../models/production.sav', 'rb'))
    scaler = pickle.load(open('../models/min_max_scaler.sav', 'rb'))

    def predict(model, scaler, age, max_heart_reate, sex, exercise_angina):
        example = pd.DataFrame()
        example[['Age', 'MaxHR']] = scaler.transform(pd.DataFrame({'Age': [age], 'MaxHR': [max_heart_reate]}))
        example['Sex'] = [1 if sex == 'M' else 0]
        example['ExerciseAngina'] = [1 if exercise_angina == 'Y' else 0]
        return model.predict_proba(example)[0][1]

    st.write(f"You have a {predict(model, scaler, age, max_heart_reate, sex, exercise_angina)*100:.2f} % chance of having Heart Disease")
    
