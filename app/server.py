
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import pickle
import pandas as pd
# 2. Create the app object
appServer = FastAPI()

# 3. Index route, opens automatically on http://127.0.0.1:8001
@appServer.get('/')
def index():
    return {'message': 'Hello, Server'}

# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@appServer.get('/get_weight')
def get_weight():
    return {'prediction': 0.0}
    

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8001
if __name__ == '__main__':
    uvicorn.run(appServer, host='127.0.0.1', port=8001)
    
#uvicorn app:app --reload