FROM python
COPY ./src /app/src
COPY requirements.txt /app
COPY BankNote_Authentication.csv /app
COPY BankNotes.py /app
COPY classifier.pkl /app
COPY ModelTraining.ipynb /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000

