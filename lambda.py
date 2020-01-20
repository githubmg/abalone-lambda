import json
import pickle
import boto3
from io import BytesIO
import numpy as np
import pandas as pd
from io import StringIO

s3 = boto3.resource('s3')
with BytesIO() as data:
    s3.Bucket("######").download_fileobj("preprocessing/le_sex.pkl", data)
    data.seek(0)    # move back to the beginning after writing
    le_sex = pickle.load(data)
    
with BytesIO() as data2:
    s3.Bucket("######").download_fileobj("preprocessing/ohe_sex.pkl", data2)
    data2.seek(0)    # move back to the beginning after writing
    ohe_sex = pickle.load(data2)
    
def lambda_handler(event, context):
    
    print('event:')
    print(event)
    if 'httpMethod' in event:
        event = event['queryStringParameters']
        
    df = pd.read_csv(StringIO(event['payload']),header=None)
    array = df[0]
    ohe_sex.drop=None
    df_abalone_sex = pd.DataFrame(
        ohe_sex.transform(le_sex.transform(array).reshape(-1, 1)).todense()
    )
    df = df.drop(0,axis=1)
    df = pd.concat([df,df_abalone_sex],axis=1)
    # call SageMaker with payload
    runtime = boto3.Session().client('sagemaker-runtime')
    sagemaker_payload = np.array2string(df.values[0],separator=',')[1:-1]
    print(sagemaker_payload)
    response = runtime.invoke_endpoint(EndpointName=event['endpoint_name'], ContentType='text/csv', Body=sagemaker_payload)
    result = json.loads(response['Body'].read().decode())
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
