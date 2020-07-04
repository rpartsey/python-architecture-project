FROM python:3.7

RUN mkdir /ml-api

COPY . /ml-api

WORKDIR /ml-api

RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

EXPOSE 8000

CMD [ "python", "/ml-api/server.py" ]
