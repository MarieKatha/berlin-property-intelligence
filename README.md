# ImmobilienScout24 Scraper Service

Scrapes real estate listings from [ImmobilienScout24](https://www.immobilienscout24.de) and returns structured data in JSON format, ready to feed into the Berlin Property Intelligence prediction models.

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Pip

### Installation

1. Clone/navigate to this directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Playwright browsers (one-time setup):
   ```bash
   playwright install chromium
   ```

5. Copy `.env.example` to `.env` and configure as needed:
   ```bash
   cp .env.example .env
   ```

6. Run the server:
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```

7. Open [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API interactively.

## Docker Setup (Local)

### Build the image

```bash
docker build -t berlin-property-scraper:latest .
```

### Run the container standalone

```bash
docker run -it --rm -p 8000:8000 --env-file .env berlin-property-scraper:latest
```

Then visit [http://localhost:8000/docs](http://localhost:8000/docs).

## Integration with Agentic API (Docker Compose)

If integrating with the existing Berlin Property Intelligence API (models + NLP agent), use `docker-compose.yml` from the parent repo to spin up all services together:

```yaml
version: '3.8'

services:
  scraper:
    build: ./scraper
    container_name: berlin-property-scraper
    ports:
      - "8001:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    networks:
      - berlin-property-network
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  sales-model:
    build: ./MLlogic-sales
    container_name: berlin-property-sales-model
    ports:
      - "8002:8000"
    networks:
      - berlin-property-network

  construction-model:
    build: ./MLlogic-new-construction
    container_name: berlin-property-construction-model
    ports:
      - "8003:8000"
    networks:
      - berlin-property-network

  rental-model:
    build: ./MLlogic_rent
    container_name: berlin-property-rental-model
    ports:
      - "8004:8000"
    networks:
      - berlin-property-network

  agent:
    build: ./agent  # Your NLP/agentic orchestrator
    container_name: berlin-property-agent
    ports:
      - "8000:8000"
    depends_on:
      - scraper
      - sales-model
      - construction-model
      - rental-model
    environment:
      - SCRAPER_URL=http://scraper:8000
      - SALES_MODEL_URL=http://sales-model:8000
      - CONSTRUCTION_MODEL_URL=http://construction-model:8000
      - RENTAL_MODEL_URL=http://rental-model:8000
    networks:
      - berlin-property-network

networks:
  berlin-property-network:
    driver: bridge
```

Then:

```bash
docker-compose up -d
```

The scraper will be reachable internally at `http://scraper:8000` from other containers.

## API Endpoints

### `GET /scrape`

Scrapes a single ImmobilienScout24 listing.

**Parameters:**
- `url` (string, required): Full URL of the listing, e.g., `https://www.immobilienscout24.de/expose/169437771`

**Response:**
```json
{
  "url": "https://www.immobilienscout24.de/expose/169437771",
  "title": "Charmante 2-Zimmer-Wohnung im Herzen von Neukölln",
  "ortsteil": "Neukölln",
  "bezirk": "Neukölln",
  "address": "Neukölln (Ortsteil), 12045 Berlin",
  "rooms": "2",
  "area_m2": "63.8",
  "energy_class": "B",
  "has_lift": "Nein",
  "raw_data": { ... }
}
```

**Example:**
```bash
curl "http://localhost:8000/scrape?url=https://www.immobilienscout24.de/expose/169437771"
```

### `GET /health`

Health check for container orchestration.

```bash
curl http://localhost:8000/health
```

### `GET /docs`

Interactive Swagger UI for testing endpoints.

## Configuration

Environment variables (see `.env.example`):

- **ENVIRONMENT**: `development` or `production`
- **DEBUG**: Log level verbosity
- **SCRAPER_TIMEOUT_SECONDS**: Max time to wait for page load (default: 30)
- **SCRAPER_HEADLESS**: Run browser in headless mode (default: True)
- **SCRAPER_WAIT_FOR_SELECTOR_TIMEOUT_MS**: Timeout for waiting for specific DOM elements
- **RATE_LIMIT_PER_MINUTE**: Requests per minute (0 = unlimited)
- **PROXY_URL**: Optional HTTP proxy for scraping
- **LOG_LEVEL**: `INFO`, `DEBUG`, `WARNING`, `ERROR`

## Project Structure

```
.
├── api.py                 # FastAPI app and endpoints
├── scraper.py            # Core scraping logic (Playwright)
├── model.py              # Pydantic models / schemas (if used)
├── utils.py              # Helper functions
├── exceptions.py         # Custom exceptions
├── __init__.py           # Package marker
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker build config
├── .dockerignore         # Docker build exclusions
├── .env.example          # Template for environment variables
└── README.md             # This file
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
black .
flake8 .
```

### Updating Dependencies

```bash
pip install -U -r requirements.txt
pip freeze > requirements.txt
```

## Troubleshooting

### Playwright Installation Issues

If `playwright install chromium` fails:

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install -y libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0

# Then try Playwright install again
playwright install chromium
```

### Scraping Timeouts

- Increase `SCRAPER_TIMEOUT_SECONDS` in `.env`
- Check network connectivity to is24.de
- IS24 may have blocks; consider adding a proxy via `PROXY_URL`

### Rate Limiting / Blocking

ImmobilienScout24 may block aggressive scraping. Recommendations:

- Add delays between requests
- Use rotating proxies (configure via `PROXY_URL`)
- Implement exponential backoff for retries
- Cache results to avoid re-scraping the same listing

## Integration with Agentic API

The scraper is designed to be called by your NLP agent to:

1. **Fetch live data**: Agent receives a user query like "Find 2-room flats under €1200/month in Kreuzberg"
2. **Scrape listings**: Agent calls `/scrape` for each candidate URL
3. **Enrich with predictions**: Agent pipes structured output into `/predict_rentals` etc.
4. **Return results**: Agent presents comparisons (market price vs. prediction)

Example agent flow:

```python
# Pseudo-code in your agentic orchestrator
urls = search_is24_by_filters(rooms=2, max_rent=1200, ortsteil="Kreuzberg")

for url in urls:
    scraped = requests.get("http://scraper:8000/scrape", params={"url": url})
    listing = scraped.json()
    
    prediction = requests.get(
        "http://rental-model:8000/predict_rentals",
        params={
            "ortsteil": listing["ortsteil"],
            "area_m2": float(listing["area_m2"]),
            "condition": "...",  # infer from scraped data
        }
    )
    
    print(f"{listing['title']} - Market: €{listing['rent']}, Predicted: €{prediction['predicted_rent_eur']}")
```

## License

Respect IS24's Terms of Service. Scraping may be rate-limited or blocked if abused.

## Contact

For issues or questions about the Berlin Property Intelligence project, refer to the main repo's documentation.
