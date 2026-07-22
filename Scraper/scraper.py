"""ImmobilienScout24 scraper with anti-detection"""

import asyncio
import random
import re
import json
import logging
from typing import Optional

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Handle imports flexibly (works both as package and standalone)
try:
    from models import PropertyData
    from exceptions import PropertyNotFoundError, ScrapeTimeoutError, AntiBotDetectedError
    from utils import get_berlin_postcode_map
except ImportError:
    try:
        from .models import PropertyData
        from .exceptions import PropertyNotFoundError, ScrapeTimeoutError, AntiBotDetectedError
        from .utils import get_berlin_postcode_map
    except ImportError:
        # Fallback: define minimal stubs if imports fail
        PropertyData = dict
        get_berlin_postcode_map = lambda: {}
        PropertyNotFoundError = Exception
        ScrapeTimeoutError = Exception
        AntiBotDetectedError = Exception

logger = logging.getLogger(__name__)


class IS24Scraper:
    """ImmobilienScout24 scraper with anti-detection"""

    def __init__(self, headless: bool = True, timeout: int = 60000):
        self.headless = headless
        self.timeout = timeout
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        ]
        self.postcode_to_bezirk = get_berlin_postcode_map()

    def _extract_feature(self, soup: BeautifulSoup, keys: list) -> str:
        """Extract a feature from the page using multiple keys"""
        # Method 1: Look in definition lists
        for dl in soup.find_all('dl'):
            for dt in dl.find_all('dt'):
                label = dt.text.strip().lower()
                for key in keys:
                    if key.lower() in label:
                        dd = dt.find_next('dd')
                        if dd:
                            return dd.text.strip()

        # Method 2: Look in grid items
        grid_items = soup.find_all('div', class_=lambda x: x and ('grid' in str(x).lower() or 'detail' in str(x).lower()))
        for item in grid_items:
            labels = item.find_all(['dt', 'span', 'div'], class_=lambda x: x and ('label' in str(x).lower() or 'title' in str(x).lower()))
            for label_elem in labels:
                label_text = label_elem.text.strip().lower()
                for key in keys:
                    if key.lower() in label_text:
                        value_elem = label_elem.find_next(['dd', 'div', 'span'])
                        if value_elem:
                            return value_elem.text.strip()

        # Method 3: Search in text
        all_text = soup.get_text()
        for key in keys:
            escaped_key = re.escape(key)
            pattern = re.compile(f'{escaped_key}[:\\s]*([^\\n]+)', re.IGNORECASE)
            match = re.search(pattern, all_text)
            if match:
                return match.group(1).strip()

        return ""

    def _extract_clean_condition(self, soup: BeautifulSoup) -> str:
        """Extract and clean the condition field"""
        condition = self._extract_feature(soup, ['zustand', 'condition', 'erhaltung', 'sanierungsbedarf', 'obj_condition'])

        if condition:
            if condition.startswith('{') or '"obj_condition"' in condition:
                try:
                    data = json.loads(condition)
                    for key in ['obj_condition', 'zustand', 'condition', 'erhaltung']:
                        if key in data:
                            value = data[key]
                            if value and value != 'no_information':
                                return self._map_condition_value(value)
                    return "Information available"
                except:
                    pass

            condition_lower = condition.lower()
            condition_map = {
                'sanierungsbedarf': 'Sanierungsbedarf',
                'sanierungsbedürftig': 'Sanierungsbedarf',
                'renovierungsbedarf': 'Renovierungsbedarf',
                'modernisiert': 'Modernisiert',
                'gepflegt': 'Gepflegt',
                'gut': 'Gut',
                'sehr gut': 'Sehr gut',
                'neuwertig': 'Neuwertig',
                'neu': 'Neubau',
                'neubau': 'Neubau',
                'renoviert': 'Renoviert',
                'kernsaniert': 'Kernsaniert',
                'altbau': 'Altbau',
                'denkmalgeschützt': 'Denkmalgeschützt',
            }

            for key, value in condition_map.items():
                if key in condition_lower:
                    return value

            condition = re.sub(r'[{}\[\]"\\]', '', condition)
            condition = re.sub(r'obj_\w+:', '', condition)
            condition = re.sub(r'no_information', '', condition)
            condition = ' '.join(condition.split())

            if len(condition) > 100 or condition.startswith('{'):
                return "Information available"

            return condition if condition else ""

        return ""

    def _map_condition_value(self, value: str) -> str:
        """Map condition values from IS24 to clean labels"""
        value = value.lower()
        if 'sanierungsbedarf' in value:
            return 'Sanierungsbedarf'
        elif 'modernisiert' in value:
            return 'Modernisiert'
        elif 'gepflegt' in value:
            return 'Gepflegt'
        elif 'neuwertig' in value:
            return 'Neuwertig'
        elif 'kernsaniert' in value:
            return 'Kernsaniert'
        elif 'renoviert' in value:
            return 'Renoviert'
        elif 'gut' in value:
            return 'Gut'
        elif 'sehr gut' in value:
            return 'Sehr gut'
        else:
            return value.capitalize()

    def _extract_ortsteil_from_address(self, address: str) -> tuple:
        """Extract Ortsteil and Bezirk from address"""
        ortsteil = ""
        bezirk = ""

        if not address:
            return ortsteil, bezirk

        postcode_match = re.search(r'\b(\d{5})\b', address)
        postcode = postcode_match.group(1) if postcode_match else None

        parts = address.split(',')
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) >= 2:
            for part in parts[1:]:
                if re.search(r'\d{5}', part):
                    if postcode:
                        bezirk = self.postcode_to_bezirk.get(postcode, "")
                    if len(parts) >= 3 and 'berlin' not in parts[-2].lower():
                        ortsteil = parts[-2]
                    elif len(parts) >= 2 and ',' in address:
                        prev_part = parts[-2] if len(parts) >= 2 else ""
                        if prev_part and not re.search(r'\d{5}', prev_part):
                            ortsteil = prev_part
                else:
                    if not ortsteil and 'berlin' not in part.lower():
                        ortsteil = part

        if not bezirk and postcode:
            bezirk = self.postcode_to_bezirk.get(postcode, "")

        if not ortsteil:
            ortsteil_match = re.search(r'Berlin[,\s-]+([A-ZÄÖÜ][a-zäöüß]+)', address)
            if ortsteil_match:
                ortsteil = ortsteil_match.group(1)

        if ortsteil and 'ortsteil' in ortsteil.lower():
            ortsteil = re.sub(r'\(?Ortsteil\)?\s*', '', ortsteil, flags=re.IGNORECASE).strip()

        return ortsteil, bezirk

    async def scrape(self, url: str) -> Optional[dict]:
        """Scrape property features from ImmobilienScout24"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
            )

            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=random.choice(self.user_agents),
                locale='de-DE',
                timezone_id='Europe/Berlin',
                # Corporate proxies (e.g. Zscaler) re-sign HTTPS with their own
                # root CA, which Chromium doesn't trust -- ignore just the TLS
                # validation, not a statement about the target site's own cert.
                ignore_https_errors=True,
                extra_http_headers={
                    'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
                    'DNT': '1',
                }
            )

            page = await context.new_page()

            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                window.chrome = { runtime: {} };
            """)

            try:
                logger.info(f"🔍 Scraping: {url}")

                await page.goto(url, wait_until='domcontentloaded', timeout=self.timeout)
                await asyncio.sleep(random.uniform(2, 4))

                for _ in range(random.randint(5, 10)):
                    scroll_amount = random.randint(200, 700)
                    await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                    await asyncio.sleep(random.uniform(0.3, 0.8))

                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(random.uniform(1, 2))

                html = await page.content()
                soup = BeautifulSoup(html, 'html.parser')

                # Title
                title_elem = soup.find('h1')
                title = title_elem.text.strip() if title_elem else ""

                # Address
                address = ""
                address_elem = soup.find('div', attrs={'data-testid': 'address'})
                if not address_elem:
                    address_elem = soup.find('div', class_=lambda x: x and 'address' in str(x).lower())
                if address_elem:
                    address = address_elem.text.strip()

                ortsteil, bezirk = self._extract_ortsteil_from_address(address)

                # Features
                rooms = self._extract_feature(soup, ['zimmer', 'rooms', 'anzahl zimmer'])
                if rooms:
                    match = re.search(r'(\d+[,.]?\d*)', rooms)
                    rooms = match.group(1).replace(',', '.') if match else ""

                area_m2 = self._extract_feature(soup, ['wohnfläche', 'fläche', 'area', 'living area', 'größe'])
                if area_m2:
                    match = re.search(r'(\d+[,.]?\d*)\s*m²', area_m2)
                    if not match:
                        match = re.search(r'(\d+[,.]?\d*)', area_m2)
                    area_m2 = match.group(1).replace(',', '.') if match else ""

                floor = self._extract_feature(soup, ['etage', 'floor', 'geschoss', 'stockwerk'])
                if floor:
                    match = re.search(r'(\d+\.?\s*[Oo]G|\d+\.\s*[Ee]tage|EG|DG|KG|Hochparterre)', floor)
                    floor = match.group(0).strip() if match else ""

                building_era = self._extract_feature(soup, ['baujahr', 'jahr', 'building age', 'erbauung'])
                if building_era:
                    match = re.search(r'(\d{4})', building_era)
                    building_era = match.group(1) if match else ""

                energy_class = self._extract_feature(soup, ['energieklasse', 'energy class', 'energetisch', 'verbrauch'])
                if energy_class:
                    match = re.search(r'([A-G]\+?)', energy_class)
                    energy_class = match.group(1) if match else ""

                condition = self._extract_clean_condition(soup)

                # Boolean features
                has_lift = ""
                lift_value = self._extract_feature(soup, ['aufzug', 'lift', 'fahrstuhl'])
                if lift_value:
                    if any(x in lift_value.lower() for x in ['ja', 'vorhanden', '✓']):
                        has_lift = "Ja"
                    elif any(x in lift_value.lower() for x in ['nein', 'nicht', '✗']):
                        has_lift = "Nein"
                if not has_lift:
                    all_text = soup.get_text().lower()
                    if 'aufzug' in all_text:
                        if 'kein aufzug' not in all_text and 'ohne aufzug' not in all_text:
                            has_lift = "Ja"
                        else:
                            has_lift = "Nein"
                    if not has_lift:
                        for script in soup.find_all('script'):
                            if 'obj_lift' in str(script):
                                if '"obj_lift":"y"' in str(script):
                                    has_lift = "Ja"
                                elif '"obj_lift":"n"' in str(script):
                                    has_lift = "Nein"

                has_balcony = ""
                balcony_value = self._extract_feature(soup, ['balkon', 'balcony', 'terrasse'])
                if balcony_value:
                    if any(x in balcony_value.lower() for x in ['ja', 'vorhanden', '✓']):
                        has_balcony = "Ja"
                    elif any(x in balcony_value.lower() for x in ['nein', 'nicht', '✗']):
                        has_balcony = "Nein"
                if not has_balcony:
                    all_text = soup.get_text().lower()
                    if 'balkon' in all_text:
                        has_balcony = "Nein" if any(x in all_text for x in ['kein balkon', 'ohne balkon']) else "Ja"

                furnished = ""
                furnished_value = self._extract_feature(soup, ['möbliert', 'furnished', 'einrichtung'])
                if furnished_value:
                    if any(x in furnished_value.lower() for x in ['ja', 'möbliert', '✓']):
                        furnished = "Ja"
                    elif any(x in furnished_value.lower() for x in ['nein', 'unmöbliert', '✗']):
                        furnished = "Nein"
                if not furnished:
                    all_text = soup.get_text().lower()
                    if 'möbliert' in all_text:
                        furnished = "Nein" if any(x in all_text for x in ['unmöbliert', 'nicht möbliert']) else "Ja"

                # Create result
                result = {
                    'url': url,
                    'title': title,
                    'ortsteil': ortsteil,
                    'bezirk': bezirk,
                    'address': address,
                    'rooms': rooms,
                    'area_m2': area_m2,
                    'floor': floor,
                    'building_era': building_era,
                    'energy_class': energy_class,
                    'condition': condition,
                    'has_lift': has_lift,
                    'has_balcony': has_balcony,
                    'furnished': furnished,
                }

                logger.info(f"✅ Success: {title[:50]}...")
                return result

            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await page.screenshot(path='error_debug.png')
                    logger.info("📸 Screenshot saved as 'error_debug.png'")
                except:
                    pass
                return None

            finally:
                await browser.close()


async def scrape_listing(url: str) -> dict:
    """Convenience wrapper for FastAPI integration"""
    scraper = IS24Scraper()
    result = await scraper.scrape(url)
    if result is None:
        raise ValueError(f"Failed to scrape {url}")
    return result


if __name__ == "__main__":
    # Quick test: python scraper.py <url>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scraper.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    result = asyncio.run(scrape_listing(url))
    import json
    print(json.dumps(result, indent=2))
