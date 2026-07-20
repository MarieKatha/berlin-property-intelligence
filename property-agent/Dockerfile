FROM python:3.12

# Install requirements
COPY property-agent/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY property-agent property-agent

WORKDIR /property-agent/src

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
