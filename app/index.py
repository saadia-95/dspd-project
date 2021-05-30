from pydantic import BaseModel
from fastapi import FastAPI
import pickle
from sklearn.metrics import mean_squared_error
import xgboost as xg

class rec_mod(BaseModel):
    id: int 
    visitors: float 
    genre: int 
    year: float 
   
    class Config:
        schema_extra = {
            "example": {
   
                "id": 599, 
                "visitors": 23.843750, 
                "genre": 4,
                "year": 2017
            
            }
        }


app = FastAPI()

@app.on_event("startup")
def load_model():
    global modelt
    #global modelx
    modelt = pickle.load(open("model_tree.pkl", "rb"))
    #modelx = pickle.load(open("model_xgbr.pkl", "rb"))


@app.get('/')
def get_root():

	return {"Hello": "Welcome to the DSPD Project. Please type /docs infront of the url to access methods. Thank you!"}


@app.post('/predicttree')
def get_prediction_tree(data: rec_mod):
    received = data.dict()
    id = received['id']
    visitors = received['visitors']
    genre = received['genre']
    year = received['year']
   
    pred_name = modelt.predict([[id, visitors, genre,
                                year]]).tolist()[0]

    return {'prediction': pred_name}

#@app.post('/predictxgbr')
#def get_prediction(data: rec_mod):
#    received = data.dict()
#    id = received['id']
#    visitors = received['visitors']
#    genre = received['genre']
#    year = received['year']
   
#    pred_name1 = modelx.predict([[id, visitors, genre,
#                                year]]).tolist()[0]

#    return {'prediction': pred_name1}


