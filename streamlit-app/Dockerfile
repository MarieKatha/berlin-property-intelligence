FROM python:3.12

# Install requirements
COPY streamlit-app/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit-app streamlit-app

# streamlit server starten
CMD streamlit run streamlit-app/streamlit_agent_app.py \
	--server.port=$PORT \
	--server.address=0.0.0.0 \
	--server.headless=true
