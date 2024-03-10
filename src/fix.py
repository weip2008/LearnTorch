import requests

# GitHub file URL
url = 'https://github.com/weip2008/LearnTorch/commits/master/doc/torchBasics.ipynb'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the content of the file
    print(response.text)
else:
    # Print an error message if the request failed
    print(f"Failed to retrieve file content: {response.status_code}")