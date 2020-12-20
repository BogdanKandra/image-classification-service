FROM tensorflow/tensorflow:2.3.1 AS base

RUN mkdir -p /app/src
RUN mkdir -p /app/static
WORKDIR /app

COPY model.h5 ./
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/

CMD python ./src/app.py