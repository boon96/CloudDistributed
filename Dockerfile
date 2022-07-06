FROM python
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY ./Query_API /app/Query_API
COPY ./data /app
COPY requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8000

