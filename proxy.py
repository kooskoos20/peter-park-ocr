import json
import boto3

client = boto3.client('lambda')

def handler(event, context):
    try:
        body = json.loads(event['body'])
        image_data = body.get("image")

        if not image_data:
            return {"statusCode": 400, "body": json.dumps({"error": "Image data required."})}

        response = client.invoke(
            FunctionName="text-recognition-service-dev-textRecognition",
            InvocationType='RequestResponse',
            Payload=json.dumps({"body": json.dumps({"image": image_data})})
        )
        result = json.load(response['Payload'])
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
