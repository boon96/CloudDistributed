from fastapi import FastAPI
import cv2

app = FastAPI()

@app.get("/")
def read_root():
    return{"Hello": "World"}

# Request to predict
@app.get("/predict/{image}")
def prediction(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(f'API_Images/{image}')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print("No Face")
        return {"result" : "Not Face"}
    else:
        print("Got Face")
        return {"result" : "Face"}

