FROM python:3.12

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY api_sales api_sales
COPY MLlogic-sales MLlogic-sales

CMD uvicorn api_sales.fast:app --host 0.0.0.0 --port $PORT
