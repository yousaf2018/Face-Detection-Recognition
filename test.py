import requests
from datetime import datetime

# Define the endpoint URL
url = "http://localhost:5000/"

# Define the data to be sent in the request body
data = {
    "_id": 9934939, 
    "name": "John Doe",
    "email": "johndoe@example.com",
    "attendance": 100,
    "date": datetime.now().isoformat()  # Sending current date and time
}

try:
    # Send POST request to the endpoint with the data
    response = requests.post(url, json=data)

    # Check the response status code
    if response.status_code == 201:
        print("User added successfully:")
        print(response.json())  # Print the response data
    else:
        print(f"Failed to add user. Status code: {response.status_code}")
        print(response.json())  # Print any error message from the server

except requests.exceptions.RequestException as e:
    print(f"Error occurred: {e}")
