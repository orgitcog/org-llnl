import boto3
from decimal import Decimal
import json
import urllib

#print('Loading function')

rekognition = boto3.client('rekognition')

# --------------- Helper Functions to call Rekognition APIs ------------------

def get_object_name(bucket, key):
    response = boto3.client('s3').get_object_tagging(
        Bucket=bucket,
        Key=key
    )
    
    name = key.replace(".png","").replace("_", " ")
    if('TagSet' in response):
        for tag in response['TagSet']:
            if(tag["Key"] == "Name" or tag["Key"] == "name"):
                name = tag["Value"]
    return name
    
def add_face(bucket, key):
    try:
        response = rekognition.create_collection(CollectionId='faces')
    except Exception as e:
        pass

    response = rekognition.index_faces(Image={"S3Object": {"Bucket": bucket, "Name": key}}, CollectionId="faces")

    if("FaceRecords" in response):
        faceid = response["FaceRecords"][0]["Face"]["FaceId"]
        table = boto3.resource('dynamodb').Table('index-face')
        table.put_item(Item={'key': key, 'name': get_object_name(bucket,key), 'faceid': faceid})
    return response

def delete_face(bucket, key):
    table = boto3.resource('dynamodb').Table('index-face')
    item = table.get_item(Key={'key': key})
    faces=[]
    print(json.dumps(item))
    faces.append(item["Item"]["faceid"])
    response = rekognition.delete_faces(CollectionId='faces',FaceIds=faces)
    
    # Sample code to write response to DynamoDB table 'MyTable' with 'PK' as Primary Key.
    # Note: role used for executing this Lambda function should have write access to the table.
    table = boto3.resource('dynamodb').Table('index-face')
    table.delete_item(Key={'key': key})
    return response


# --------------- Main handler ------------------


def lambda_handler(event, context):
    '''Demonstrates S3 trigger that uses
    Rekognition APIs to index faces in S3 Object.
    '''
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    eventtype = event['Records'][0]['eventName']
    name = "John Doe"
    key = event['Records'][0]['s3']['object']['key']
    try:
        if(eventtype == "ObjectCreated:Put"):
            # Calls rekognition API to index faces in S3 object and add dynamodb record
            response = add_face(bucket, key)

        if(eventtype == "ObjectRemoved:Delete"):
            # Calls rekognition API to index faces in S3 object and add dynamodb record
            response = delete_face(bucket, key)

        # Print response to console.
        print(response)

        return response
    except Exception as e:
        print(e)
        print("Error processing object {} from bucket {}. ".format(key, bucket) +
              "Make sure your object and bucket exist and your bucket is in the same region as this function.")
