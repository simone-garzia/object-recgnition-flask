import requests
import pandas as pd
import torch
import numpy as np
from utils import create_rect, plot_image_rect, save_image_rect


def classify_vector(vector_list):
    url = "http://127.0.0.1:5000/classify"
    payload = {"vector_list": vector_list}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("Predicted {} ROI".format(len(response.json()["roi_list"])))
    else:
        print("Error:", response.json()["error"])
    return response.json()["roi_list"]

if __name__ == "__main__":
    # Example usage
    df_test = pd.read_csv('Data/testData.csv')
    test_vector_list = []
    test_vector_list = [df_test.loc[i, :].tolist() for i in range(len(df_test))]
    roi_list_pred = classify_vector(test_vector_list)
    save_image_rect(test_vector_list, roi_list_pred)
  
    
    
    
