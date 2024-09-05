import requests

# URL of the MoveNet model
url = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

# Correctly handle redirects and download the file
response = requests.get(url, allow_redirects=True)

# Check if the download is successful by verifying the content-type
if 'tflite' not in response.headers.get('Content-Type', ''):
    print("Failed to download the model. The URL might have redirected to an HTML page.")
else:
    # Save the model locally if it's valid
    with open("movenet.tflite", "wb") as f:
        f.write(response.content)
    print("Model downloaded and saved successfully as movenet.tflite")
