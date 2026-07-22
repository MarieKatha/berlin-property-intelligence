FROM python:3.12

# Install requirements
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY api api
COPY MLlogic-sales MLlogic-sales
COPY MLlogic-new-construction MLlogic-new-construction
COPY MLlogic_rent MLlogic_rent

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
