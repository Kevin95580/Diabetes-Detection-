# from fastapi import FastAPI, Form
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# import joblib
# import pandas as pd

# # Initialize the app
# app = FastAPI()

# # Serve the static directory for CSS
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Load the model, scaler, and feature list
# model = joblib.load('diabetes_model.pkl')
# scaler = joblib.load('scaler.pkl')
# features = joblib.load('model_features.pkl')

# # Home route: Form page
# @app.get("/", response_class=HTMLResponse)
# async def read_form():
#     with open("templates/index.html", "r") as file:
#         return HTMLResponse(content=file.read())

# # Prediction route
# @app.post("/predict/")
# async def predict(
#     gender: str = Form(...),
#     age: float = Form(...),
#     hypertension: int = Form(...),
#     heart_disease: int = Form(...),
#     smoking_history: str = Form(...),
#     bmi: float = Form(...),
#     HbA1c_level: float = Form(...),
#     blood_glucose_level: int = Form(...)
# ):
#     # Prepare the input data
#     input_data = {
#         'gender': gender,
#         'age': age,
#         'hypertension': hypertension,
#         'heart_disease': heart_disease,
#         'smoking_history': smoking_history,
#         'bmi': bmi,
#         'HbA1c_level': HbA1c_level,
#         'blood_glucose_level': blood_glucose_level
#     }

#     # Convert the input into a dataframe and one-hot encode it
#     input_df = pd.DataFrame([input_data])

#     # Apply the same preprocessing (one-hot encoding and scaling)
#     input_df = pd.get_dummies(input_df).reindex(columns=features, fill_value=0)
#     input_scaled = scaler.transform(input_df)

#     # Get prediction
#     prediction = model.predict(input_scaled)

#     result = "Diabetic" if prediction[0] == 1 else "Non-diabetic"

#     # Return the result
#     return HTMLResponse(content=f"<h2>Prediction: {result}</h2>")

######################################################################################################



from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import joblib
import pandas as pd

# Initialize the app
app = FastAPI()

# Mount static files for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the templates
templates = Jinja2Templates(directory="templates")

# Load the model, scaler, and feature list
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('model_features.pkl')

# Home route: Form page
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict/")
async def predict(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    smoking_history: str = Form(...),
    bmi: float = Form(...),
    HbA1c_level: float = Form(...),
    blood_glucose_level: int = Form(...)
):
    # Prepare the input data
    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }

    # Convert the input into a dataframe and one-hot encode it
    input_df = pd.DataFrame([input_data])

    # Apply the same preprocessing (one-hot encoding and scaling)
    input_df = pd.get_dummies(input_df).reindex(columns=features, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # Get prediction
    prediction = model.predict(input_scaled)

    result = "Diabetic" if prediction[0] == 1 else "Non-diabetic"

    # Render the result on a template
    return templates.TemplateResponse("result.html", {
        "request": request,
        "result": result,
        "input_data": input_data
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)