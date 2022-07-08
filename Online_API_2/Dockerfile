FROM python:3.7
RUN pip3 install fastapi uvicorn pydantic
RUN pip3 install opencv-python opencv-contrib-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./API_Images /app/API_Images
COPY Online_API.py /app
COPY haarcascade_frontalface_alt.xml /app
COPY ./models /app/models/
WORKDIR /app

EXPOSE 8002
CMD ["uvicorn", "Online_API:app", "--host", "0.0.0.0", "--port", "8002"]