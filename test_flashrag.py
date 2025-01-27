import requests

# sending GET/POST etc requests using Query-String-Parameter
response = requests.get(url="http://127.0.0.1:8000/query?query='who am I?'")
print(response.json())
