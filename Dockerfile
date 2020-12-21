FROM tensorflow/tensorflow:2.3.1 AS base

RUN mkdir -p /app/src
RUN mkdir -p /app/static
WORKDIR /app

COPY src/ ./src/
COPY static/ ./static/
COPY requirements.txt model.h[5] ./

RUN pip install -r requirements.txt
RUN python ./src/create_model.py

# CMD python ./src/app.py
CMD ["gunicorn", "-c", "src/gunicorn.conf.py", "src.app:APP"]