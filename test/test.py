import requests

# Simplest GET request to the 'api' container
response = requests.get("http://dashboard:8000")
print(response)

with open("test_output.txt", "w") as f:
    f.write(f"Response from dashboard: {response.text}\n")
