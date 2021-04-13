
# client.py
import requests

resp = requests.post(".../predict",
                     files={"file": open('.../cat.jpg','rb')})  
