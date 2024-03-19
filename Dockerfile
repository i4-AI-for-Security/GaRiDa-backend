FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update
RUN apt-get install -y \
    'python'\
    'ffmpeg'\
    'libsm6'\
    'libxext6'\
    'build-essential'\
    'pkg-config'\
    'libhdf5-dev'

COPY requirements_mediapipe.txt  .
RUN pip install -r requirements_mediapipe.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
