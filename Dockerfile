FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /reject_accept_api
WORKDIR $APP_HOME
COPY . ./

RUN pip install pycaret fastapi uvicorn gunicorn

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 reject_accept_api:app
