import requests
import pandas as pd
import torch
import numpy as np
from utils import create_rect, plot_image_rect


def classify_vector(vector):
    url = "http://127.0.0.1:5000/classify"
    payload = {"vector": vector}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Predicted ROI:", response.json()["roi"])
    else:
        print("Error:", response.json()["error"])
    return response.json()["roi"]

if __name__ == "__main__":
    # Example usage
    df_test = pd.read_csv('Data/testData.csv')
    test_vector = df_test.loc[1,:].to_list()
    roi_pred = classify_vector(test_vector)
    plot_image_rect(test_vector, roi_pred)
  
    
    
    
