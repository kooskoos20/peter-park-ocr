import json
import base64
import boto3
import easyocr
import numpy as np
from PIL import Image
import io
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

textract = boto3.client('textract')
reader = easyocr.Reader(['en'], gpu=False)

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

def method_easyocr(base64_image):
    img_bytes = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(img_bytes))
    img_np = np.array(image)

    results = reader.readtext(img_np)

    texts = [res[1] for res in results]
    confidences = [res[2] for res in results]

    text = "\n".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    return text, avg_confidence

def handler(event, context):
    try:
        base64_image = event.get('image')
        if not base64_image:
            return {'error': 'Missing "image" field'}

        text_textract, conf_textract = method_textract(base64_image)
        text_easyocr, conf_easyocr = method_easyocr(base64_image)

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
