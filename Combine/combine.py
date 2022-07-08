import urllib.request
pdf_path = ""
def download_file(download_url, filename):
    response = urllib.request.urlopen(download_url)    
    file = open(filename + ".h5", 'wb')
    file.write(response.read())
    file.close()
 
download_file("http://127.0.0.1:8000/get_model", "Test")