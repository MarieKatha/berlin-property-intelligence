"""Data models for scraped properties"""

from dataclasses import dataclass, asdict
import json
from typing import Dict, Any, Optional


@dataclass
class PropertyData:
    """Data class for property information"""
    url: str
    title: str = ""
    ortsteil: str = ""
    bezirk: str = ""
    address: str = ""
    rooms: str = ""
    area_m2: str = ""
    floor: str = ""
    building_era: str = ""
    energy_class: str = ""
    condition: str = ""
    has_lift: str = ""
    has_balcony: str = ""
    furnished: str = ""
    kaltmiete: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding empty values"""
        return {k: v for k, v in asdict(self).items() if v not in [None, "", 0]}

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyData":
        """Create PropertyData from dictionary"""
        return cls(**data)


# Pydantic models for API
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
from typing import List, Optional as OptionalType


class ScrapeRequest(BaseModel):
    """Request model for scraping a single property"""
    url: HttpUrl = Field(..., description="ImmobilienScout24 property URL")
    headless: bool = Field(True, description="Run browser in headless mode")
    timeout: int = Field(60000, description="Timeout in milliseconds")


class BatchScrapeRequest(BaseModel):
    """Request model for batch scraping"""
    urls: List[HttpUrl] = Field(..., description="List of property URLs")
    headless: bool = Field(True, description="Run browser in headless mode")
    max_concurrent: int = Field(5, description="Maximum concurrent scrapes")


class ScrapeResponse(BaseModel):
    """Response model for scrape results"""
    success: bool
    data: OptionalType[Dict[str, Any]] = None
    error: OptionalType[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchScrapeResponse(BaseModel):
    """Response model for batch scrape results"""
    success: bool
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total: int = 0
    successful: int = 0
    failed: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    service: str
