FROM python:3.7
RUN pip3 install fastapi uvicorn pydantic
RUN pip3 install opencv-python opencv-contrib-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./Unfiltered_Images /app/Unfiltered_Images
COPY Label_Query.py /app
WORKDIR /app
CMD ["python", "Label_Query.py"]
EXPOSE 8001