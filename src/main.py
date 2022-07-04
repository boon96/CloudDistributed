
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import pickle
from typing import Optional

# 2. Create the app object
app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }

@app.post('/get_weight')
def get_weight(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    return {"Weight is", variance}

