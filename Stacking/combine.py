import requests
from subprocess import call
from pathlib import Path

def download_file(URL,filename):
    # 2. download the data behind the URL
    response = requests.get(URL)
    # 3. Open the response into a new file called instagram.ico
    open(f"Stacking/models/{filename}.h5", "wb").write(response.content)

download_file("http://127.0.0.1:8000/get_model","imageclassifier_1")
download_file("http://127.0.0.1:8002/get_model","imageclassifier_2")
