import pandas as pd
import pickle
import gradio as gr

# MUST be defined before loading model
def create_features(X):
    X = X.copy()
    X['Age_Group'] = X['Age'] // 10
    X['Salary_Group'] = X['EstimatedSalary'] // 1000
    X['Age_Salary_Interaction'] = X['Age'] * X['EstimatedSalary']
    return X

# Load trained model
with open('model.pkl', 'rb') as file:
    best_model = pickle.load(file)

def predict_purchase(gender, age, salary):
    data = pd.DataFrame(
        [[gender, age, salary]],
        columns=['Gender', 'Age', 'EstimatedSalary']
    )
    
    prediction = best_model.predict(data)[0]
    
    return "Will Purchase ✅" if prediction == 1 else "Will Not Purchase ❌"

interface = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Estimated Salary")
    ],
    outputs=gr.Text(label="Prediction"),
    title="Social Network Ads Purchase Prediction",
    description="Enter user details to predict purchase behavior"
)

interface.launch()
