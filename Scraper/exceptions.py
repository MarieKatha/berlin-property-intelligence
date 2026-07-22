"""Custom exceptions for the scraper"""


class ScraperError(Exception):
    """Base scraper exception"""
    pass


class PropertyNotFoundError(ScraperError):
    """Property not found on ImmobilienScout24"""
    pass


class ScrapeTimeoutError(ScraperError):
    """Scrape operation timed out"""
    pass


class InvalidURLError(ScraperError):
    """Invalid URL provided"""
    pass


class AntiBotDetectedError(ScraperError):
    """Anti-bot detection triggered"""
    pass
