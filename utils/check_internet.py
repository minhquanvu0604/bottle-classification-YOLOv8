import requests

def check_internet(url='http://www.google.com'):
    try:
        # Send a GET request to the specified URL
        response = requests.get(url, timeout=5)  # Timeout set to 5 seconds
        # If the request was successful, no error is raised
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        print(f"Internet connection is available. Status Code: {response.status_code}")
    except requests.RequestException as e:
        # Handle exceptions that might occur during the request
        print(f"No internet connection available. Error: {e}")

# Usage
check_internet()