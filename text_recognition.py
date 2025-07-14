import json
import base64
import boto3
import numpy as np
from PIL import Image
import io
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/opt/bin/tesseract"
textract = boto3.client('textract')

def method_textract(base64_image):
    image_bytes = base64.b64decode(base64_image)
    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    lines = []
    confidences = []
    for block in response['Blocks']:
        if block['BlockType'] == 'LINE':
            lines.append(block['Text'])
            confidences.append(block.get('Confidence', 0))

    text = "\n".join(lines)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    return text, avg_confidence / 100  # Textract confidence is percent

def method_tesseract(base64_image):
    img_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(img_bytes))

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    texts = []
    confidences = []

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:
            texts.append(data['text'][i])
            confidences.append(int(data['conf'][i]))

    text = " ".join(texts)
    avg_confidence = (sum(confidences) / len(confidences) / 100) if confidences else 0
    
    return text, avg_confidence

def handler(event, context):
    try:
        body = json.loads(event['body'])
        base64_image = body.get('image')
        if not base64_image:
            return {'error': 'Missing "image" field'}

        text_textract, conf_textract = method_textract(base64_image)
        text_easyocr, conf_easyocr = method_tesseract(base64_image)

        # Compare confidences and return best
        if conf_textract >= conf_easyocr:
            best_text = text_textract
            best_confidence = conf_textract
            best_method = 'textract'
        else:
            best_text = text_easyocr
            best_confidence = conf_easyocr
            best_method = 'easyocr'

        return {
            'text': best_text,
            'confidence': best_confidence,
            'method_used': best_method
        }

    except Exception as e:
        return {'error': str(e)}
