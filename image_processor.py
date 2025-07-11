import json
import base64
import io
import boto3
from PIL import Image
from google.cloud import vision

# Initialize clients outside the handler
rekognition_client = boto3.client('rekognition')
# The Vision client will automatically find the credentials from the environment
vision_client = vision.ImageAnnotatorClient()


def recognize_text_handler(event, context):
    image_b64 = event['image']
    image_bytes = base64.b64decode(image_b64)

    # --- Pre-processing (Example: Convert to JPEG) ---
    # AWS Rekognition works best with JPEG or PNG
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            processed_image_bytes = buffer.getvalue()
    except Exception as e:
        print(f"Could not process image: {e}")
        processed_image_bytes = image_bytes


    # --- Method 1: AWS Rekognition ---
    rek_response = rekognition_client.detect_text(Image={'Bytes': processed_image_bytes})
    rek_text_detections = rek_response.get('TextDetections', [])
    rek_avg_confidence = 0
    rek_result_text = ""
    # Understand later how this is being calculated.
    if rek_text_detections:
        rek_avg_confidence = sum(d['Confidence'] for d in rek_text_detections if d['Type'] == 'LINE') / len([d for d in rek_text_detections if d['Type'] == 'LINE'])
        rek_result_text = " ".join([d['DetectedText'] for d in rek_text_detections if d['Type'] == 'LINE'])


    # --- Method 2: Google Cloud Vision ---
    vision_image = vision.Image(content=processed_image_bytes)
    vision_response = vision_client.text_detection(image=vision_image)
    vision_texts = vision_response.text_annotations
    google_avg_confidence = 0
    google_result_text = ""
    if vision_texts:
        # Google Vision's first result is the full text block, others are individual words.
        # Confidence is not directly provided per block, so we use the score of the full text.
        google_avg_confidence = vision_texts[0].score * 100 # Convert to percentage-like scale
        google_result_text = vision_texts[0].description.replace('\n', ' ')


    # --- Confidence Comparison ---
    if rek_avg_confidence > google_avg_confidence:
        final_result = {
            'source': 'AWS Rekognition',
            'text': rek_result_text,
            'confidence': rek_avg_confidence
        }
    else:
        final_result = {
            'source': 'Google Cloud Vision',
            'text': google_result_text,
            'confidence': google_avg_confidence
        }

    # In a real-world scenario, you'd send this result somewhere (e.g., a webhook, S3, database).
    # For now, we'll just print it to the logs.
    print(json.dumps(final_result))

    return {'status': 'success', 'result': final_result}


# Local testing
# (Your existing imports and function definitions for proxy_handler and recognize_text_handler go here)


# ===================================================================
#  Direct invocation block for local testing
# ===================================================================
if __name__ == "__main__":
    import base64

    # 1. Encode a local image to base64
    #    Replace 'path/to/your/test_image.png' with your image file
    try:
        with open("path/to/your/test_image.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print("Error: Test image not found. Please update the path.")
        exit()

    # 2. Create the 'event' dictionary that the handler expects
    test_event = {
        "image": encoded_string
    }

    # 3. Call the handler function directly and print the output
    print("🚀 Starting local test...")
    result = recognize_text_handler(test_event, None) # context can be None
    print("\n✅ Test finished. Result:")
    print(result)