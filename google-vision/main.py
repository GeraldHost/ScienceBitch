#! /usr/bin/env python
"""
set up your google credentials
export GOOGLE_APPLICATION_CREDENTIALS=[path_to_creds.json]
use this script
./main.py path_to_image
"""
import sys
import io
from google.cloud import vision

def read_image(client, path):
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)
    
    texts = response.text_annotations
    for text in texts:
        print(text.description)

if __name__ == "__main__":
    client = vision.ImageAnnotatorClient()
    path = sys.argv[1]
    read_image(client, path)



