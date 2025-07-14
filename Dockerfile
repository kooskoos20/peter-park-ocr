FROM public.ecr.aws/lambda/python:3.8

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt -t /var/task

# Copy your function code
COPY text_recognition.py .

# Copy Tesseract layer contents to /opt
COPY amazonlinux-2/ /opt

CMD ["text_recognition.handler"]
