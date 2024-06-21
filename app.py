import os 
import json
import sys 
import boto3

bedrock=boto3.client(service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('aws_access_key_id'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_id'))
prompt="""
you are a cricket expert now jsut tell me when RCB
will win the IPL?
"""

payload={
    "prompt": "[INST]" + prompt+ "[/INST]",
    "max_gen_len":512,
    "temperature":0.3,
    "top_p":0.9

}
body=json.dumps(payload)


response=bedrock.invoke_model(
    modelId="meta.llama2-70b-chat-v1",
    body=body,
    accept='application/json',
    contentType='application/json'
)
request_body=json.loads(response.get("body").read())
response=request_body["generation"]
print(response)

