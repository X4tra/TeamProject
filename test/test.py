import requests

# Simplest GET request to the 'api' container
response = requests.get("http://api:8000")
print(response)
