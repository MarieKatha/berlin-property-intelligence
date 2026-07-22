"""FastAPI app exposing the ImmobilienScout24 scraper as a service.

Endpoint:
  /scrape              -- Scrape a single IS24 listing URL and return structured data

The scraper is designed to parse real estate listings from ImmobilienScout24
(is24.de) and extract key fields like area, rooms, condition, energy class, etc.
Output is a flat JSON dict matching the structure expected by downstream
prediction models (predict_sales, predict_construction, predict_rentals).
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Only immobilienscout24.de -- the scraper's extraction logic is built
# specifically around that site's markup and would just return empty
# fields for anything else. Rejecting other hosts up front also keeps this
# endpoint from being usable as an open SSRF-style URL fetcher once public.
ALLOWED_HOSTS = {"www.immobilienscout24.de", "immobilienscout24.de"}

# Configure logging to stdout (Docker-friendly)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Berlin Property Intelligence - ImmobilienScout24 Scraper",
    description="Scrapes and structures real estate listings from IS24 for use with prediction models",
    version="1.0.0",
)


class ScrapingResponse(BaseModel):
    """Response from a successful scraping operation."""

    url: str
    title: Optional[str] = None
    ortsteil: Optional[str] = None
    bezirk: Optional[str] = None
    address: Optional[str] = None
    rooms: Optional[str] = None
    area_m2: Optional[str] = None
    energy_class: Optional[str] = None
    has_lift: Optional[str] = None
    # Add any other fields your scraper extracts
    raw_data: Optional[dict] = None  # For debugging / transparency


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    """Root endpoint with usage instructions."""
    example_url = "https://www.immobilienscout24.de/expose/169437771"
    scrape_link = f"/scrape?url={example_url}"
    docs_link = "/docs"
    return (
        "Welcome to the Berlin Property Intelligence Scraper API.<br>"
        f'Example scrape: <a href="{scrape_link}">/scrape?url=...</a><br>'
        f'Interactive API docs: <a href="{docs_link}">{docs_link}</a>'
    )


@app.get("/health")
def health() -> dict:
    """Health check endpoint for container orchestration."""
    return {"status": "ok", "service": "berlin-property-scraper"}


@app.get("/scrape", response_model=ScrapingResponse)
async def scrape(
    url: str = Query(
        ...,
        description="Full URL of an ImmobilienScout24 listing (e.g., https://www.immobilienscout24.de/expose/123456)",
    ),
) -> ScrapingResponse:
    """Scrapes a single ImmobilienScout24 listing and returns structured data.

    The response includes fields like area_m2, rooms, condition, energy_class, etc.
    that can be fed directly into the prediction models
    (/predict_sales, /predict_construction, /predict_rentals).

    **Rate Limiting:** Scraping is I/O-heavy (Playwright browser automation).
    Use responsibly and cache results when possible.

    **Error Handling:** If the URL is unreachable, the page structure has changed,
    or required fields are missing, the endpoint returns a 422 (validation error)
    or 503 (service unavailable) with details.
    """
    logger.info(f"Scraping request for URL: {url}")

    hostname = urlparse(url).hostname or ""
    if hostname not in ALLOWED_HOSTS:
        raise HTTPException(
            status_code=422,
            detail="Only immobilienscout24.de listing URLs are supported.",
        )

    try:
        # Import here to avoid circular imports and get fresh instance
        from .scraper import IS24Scraper

        # Create scraper with headless=True to prevent browser window
        scraper = IS24Scraper(headless=True, timeout=60000)
        raw_data = await scraper.scrape(url)

        if raw_data is None:
            raise ValueError(f"Scraper returned None for {url}")

    except TimeoutError as e:
        logger.error(f"Timeout scraping {url}: {e}")
        raise HTTPException(
            status_code=503,
            detail="Scraping timed out. The page may be slow or unresponsive. Try again later.",
        )
    except ValueError as e:
        logger.error(f"Validation error scraping {url}: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Failed to extract required fields from the listing: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during scraping: {str(e)}",
        )

    # Construct response from raw_data
    # Assuming raw_data is a dict with keys: url, title, ortsteil, bezirk, address, rooms, area_m2, energy_class, has_lift, etc.
    response = ScrapingResponse(
        url=str(url),
        title=raw_data.get("title"),
        ortsteil=raw_data.get("ortsteil"),
        bezirk=raw_data.get("bezirk"),
        address=raw_data.get("address"),
        rooms=raw_data.get("rooms"),
        area_m2=raw_data.get("area_m2"),
        energy_class=raw_data.get("energy_class"),
        has_lift=raw_data.get("has_lift"),
        raw_data=raw_data,  # Include full dict for transparency (remove in production if sensitive)
    )

    logger.info(f"Successfully scraped {url}")
    return response
