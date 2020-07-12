FROM python:3.7

RUN mkdir /metrics-api

COPY . /metrics-api

WORKDIR /metrics-api

RUN pip install -r requirements.txt

EXPOSE 8001

CMD [ "python", "/metrics-api/server.py" ]