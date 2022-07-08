FROM python:3.7
RUN pip3 install fastapi uvicorn pydantic
RUN pip3 install requests tensorflow matplotlib
RUN pip3 install opencv-python opencv-contrib-python
RUN pip3 install sklearn pydot graphviz
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./Images /app/Images
COPY ./logs /app/logs
COPY ./models /app/models
COPY integrated_stacking_model.py /app
COPY combine.py /app
COPY model_graph.png /app
WORKDIR /app
EXPOSE 8004
CMD ["python3", "./integrated_stacking_model.py"]