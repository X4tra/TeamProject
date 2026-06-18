import requests

# Simplest GET request to the 'api' container
response = requests.get("http://dashboard:8000/api/test")
print(response.text)
