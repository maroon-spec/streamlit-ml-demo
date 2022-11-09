import streamlit as st 
import numpy as np 
from PIL import Image
import base64
import io


#st.title('Title')
st.header('Databricks ML model serving demo')
st.image("/app/streamlit-ml-demo/images/serving.png", width=500)

# Copy and paste this code from the MLflow real-time inference UI. Make sure to save Bearer token from 
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-984752964297111.11.azuredatabricks.net/model-endpoint/daiwt2022_model/2/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# data upload interface
csv_file_buffer_single = st.file_uploader('CSVをアップロードしてください', type='json')
if csv_file_buffer_single is not None:
  df = pd.read_json(csv_file_buffer_single)
  #st.write(df)

  response = score_model(df) 
  df['prediction'] = response
  st.write(df[["customerID,prediction"]])
  #
  #result = ""
  #if df['prediction'][0] == 1:
  #  result = "解約"
  #else:
  #  result = "サービス継続"

  st.write(f'顧客の契約予測: {response}')
