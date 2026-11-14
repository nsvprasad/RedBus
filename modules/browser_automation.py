"""
browser_automation.py - Module 3: Playwright Browser Automation Foundation for RedBus AI Assistant.

This module provides the core browser automation infrastructure using Playwright Chromium:
- Pydantic models for bus search items and aggregate results.
- Robust browser lifecycle management (headless & headed) with try/finally cleanup.
- Safe website navigation and accessibility verification for RedBus (https://www.redbus.in).
- Detection of access restrictions, Cloudflare/Akamai blocks, and CAPTCHA states.
"""

import re
import logging
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, List, Dict, Any, Generator, Tuple

from pydantic import BaseModel, Field, field_validator, ValidationInfo
from playwright.sync_api import sync_playwright, Playwright, Browser, BrowserContext, Page, Error as PlaywrightError

# Configure module logger
logger = logging.getLogger(__name__)

# Constants
REDBUS_HOME_URL = "https://www.redbus.in"
DEFAULT_TIMEOUT_MS = 30000
DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Common indicators of anti-bot or access restriction screens
ACCESS_DENIED_PATTERNS = [
    r"access\s+denied",
    r"attention\s+required",
    r"just\s+a\s+moment",
    r"pardon\s+our\s+interruption",
    r"403\s+forbidden",
    r"security\s+check",
    r"cf-browser-verification",
    r"verify\s+you\s+are\s+human",
    r"challenge-platform",
]


# ============================================================================
# 1. Pydantic Data Models
# ============================================================================

class BusItem(BaseModel):
    """
    Structured model representing an individual bus trip listing.
    """
    operator_name: str = Field(..., description="Name of the bus operator or travel agency")
    bus_type: str = Field(..., description="Type of bus (e.g., 'AC Sleeper (2+1)', 'Non-AC Seater')")
    departure_time: str = Field(..., description="Departure time formatted as HH:MM")
    arrival_time: str = Field(..., description="Arrival time formatted as HH:MM")
    duration: str = Field(..., description="Trip duration (e.g., '07h 30m')")
    fare: float = Field(..., description="Ticket price as a non-negative float value")
    rating: Optional[str] = Field(default=None, description="Operator user rating (e.g., '4.2')")
    seats_available: Optional[str] = Field(default=None, description="Seat availability count (e.g., '12 Seats')")

    @field_validator("fare", mode="before")
    @classmethod
    def validate_and_clean_fare(cls, v: Any) -> float:
        """
        Cleans and validates the fare value.
        Extracts numeric values from strings (e.g. '₹850', 'INR 1,200.50') and ensures non-negative.
        """
        if v is None:
            raise ValueError("Fare cannot be None")
        if isinstance(v, (int, float)):
            val = float(v)
        else:
            s = str(v).replace(",", "").strip()
            match = re.search(r"(\d+(?:\.\d+)?)", s)
            if not match:
                raise ValueError(f"Invalid fare format: {v}")
            val = float(match.group(1))

        if val < 0:
            raise ValueError("Fare cannot be negative")
        return val

    @field_validator("operator_name", "bus_type", "departure_time", "arrival_time", "duration")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: ValidationInfo) -> str:
        """Ensures core string fields are not blank."""
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()


class BusSearchResult(BaseModel):
    """
    Aggregate search result container for a route and journey date.
    """
    source: str = Field(..., description="Departure origin city")
    destination: str = Field(..., description="Arrival destination city")
    journey_date: str = Field(..., description="Journey date in YYYY-MM-DD format")
    total_found: int = Field(default=0, description="Total number of buses extracted")
    buses: List[BusItem] = Field(default_factory=list, description="List of validated bus listings")
    error_message: Optional[str] = Field(default=None, description="Error message if search failed or was blocked")

    @field_validator("source", "destination", mode="before")
    @classmethod
    def clean_cities(cls, v: Any) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("City name cannot be empty")
        return s.title()


# ============================================================================
# 2. Browser Initialization & Lifecycle Management
# ============================================================================

def init_browser(
    playwright_instance: Playwright,
    headless: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    viewport: Optional[Dict[str, int]] = None
) -> Tuple[Browser, BrowserContext, Page]:
    """
    Initialize Playwright Chromium browser, isolated browser context, and page.

    Args:
        playwright_instance (Playwright): Active Playwright instance.
        headless (bool): Run browser in headless mode (default True).
        user_agent (str): Custom user-agent string.
        viewport (dict): Viewport dimensions dictionary with width and height.

    Returns:
        Tuple[Browser, BrowserContext, Page]: Initialized browser, context, and page.
    """
    viewport_cfg = viewport or DEFAULT_VIEWPORT
    logger.info("Initializing Chromium browser (headless=%s, viewport=%s)", headless, viewport_cfg)

    # Launch Chromium with anti-detection flags for reliable rendering
    browser = playwright_instance.chromium.launch(
        headless=headless,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
        ]
    )

    context = browser.new_context(
        user_agent=user_agent,
        viewport=viewport_cfg,
        locale="en-IN",
        timezone_id="Asia/Kolkata"
    )

    page = context.new_page()
    page.set_default_timeout(DEFAULT_TIMEOUT_MS)
    return browser, context, page


def close_browser(browser: Optional[Browser], playwright_instance: Optional[Playwright]) -> None:
    """
    Safely close browser and stop Playwright instance.

    Args:
        browser: Active Browser instance (if any).
        playwright_instance: Active Playwright instance (if any).
    """
    if browser:
        try:
            logger.info("Closing Chromium browser")
            browser.close()
        except Exception as e:
            logger.warning("Error during browser close: %s", e)

    if playwright_instance:
        try:
            logger.info("Stopping Playwright instance")
            playwright_instance.stop()
        except Exception as e:
            logger.warning("Error during Playwright stop: %s", e)


@contextmanager
def launch_browser_session(
    headless: bool = True,
    user_agent: str = DEFAULT_USER_AGENT,
    viewport: Optional[Dict[str, int]] = None
) -> Generator[Page, None, None]:
    """
    Context manager for safe browser session lifecycle management.
    Guarantees cleanup of browser, context, and Playwright driver in all conditions.

    Yields:
        Page: Active Playwright Page instance.
    """
    playwright_instance = None
    browser = None
    try:
        playwright_instance = sync_playwright().start()
        browser, _, page = init_browser(
            playwright_instance,
            headless=headless,
            user_agent=user_agent,
            viewport=viewport
        )
        yield page
    finally:
        close_browser(browser, playwright_instance)


# ============================================================================
# 3. Website Navigation & Accessibility Check
# ============================================================================

def check_website_accessibility(
    page: Page,
    url: str = REDBUS_HOME_URL,
    timeout_ms: int = DEFAULT_TIMEOUT_MS
) -> Dict[str, Any]:
    """
    Navigate to a URL and perform basic accessibility and security check.
    Detects if the page loaded normally or if an access-denied/CAPTCHA screen is present.

    Args:
        page (Page): Active Playwright page.
        url (str): Target URL (defaults to RedBus homepage).
        timeout_ms (int): Navigation timeout in milliseconds.

    Returns:
        Dict[str, Any]: Status dictionary containing:
            - is_accessible (bool): True if loaded without access blocks.
            - title (str): Page title.
            - url (str): Final redirected URL.
            - status_code (int): HTTP response status code.
            - access_blocked (bool): True if anti-bot/CAPTCHA is detected.
            - block_reason (Optional[str]): Description of block if detected.
            - error (Optional[str]): Network/navigation error if failed.
    """
    result = {
        "is_accessible": False,
        "title": "",
        "url": url,
        "status_code": None,
        "access_blocked": False,
        "block_reason": None,
        "error": None
    }

    logger.info("Navigating to URL: %s", url)
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if response:
            result["status_code"] = response.status
        result["url"] = page.url
        result["title"] = page.title()

        logger.info("Page loaded. Title: '%s', Status: %s", result["title"], result["status_code"])

        # Check HTTP status code
        if result["status_code"] in (403, 429, 503):
            result["access_blocked"] = True
            result["block_reason"] = f"HTTP status {result['status_code']}"
            logger.warning("Access restriction detected by status code: %s", result["status_code"])
            return result

        # Check title and content for access restriction signatures
        page_title_lower = (result["title"] or "").lower()
        try:
            body_text = page.locator("body").inner_text(timeout=5000).lower()
        except Exception:
            body_text = ""

        for pattern in ACCESS_DENIED_PATTERNS:
            if re.search(pattern, page_title_lower) or re.search(pattern, body_text):
                result["access_blocked"] = True
                result["block_reason"] = f"Detected security challenge matching '{pattern}'"
                logger.warning("Access restriction signature detected: %s", result["block_reason"])
                return result

        result["is_accessible"] = True
        return result

    except PlaywrightError as pe:
        err_msg = f"Playwright navigation error: {str(pe)}"
        logger.error(err_msg)
        result["error"] = err_msg
        return result
    except Exception as e:
        err_msg = f"Unexpected navigation error: {str(e)}"
        logger.error(err_msg)
        result["error"] = err_msg
        return result


def open_redbus(
    page: Page,
    url: str = REDBUS_HOME_URL,
    timeout_ms: int = DEFAULT_TIMEOUT_MS
) -> Dict[str, Any]:
    """
    Open the RedBus home page and verify accessibility.

    Args:
        page (Page): Active Playwright page.
        url (str): Target URL (defaults to REDBUS_HOME_URL).
        timeout_ms (int): Navigation timeout.

    Returns:
        Dict[str, Any]: Status dictionary from check_website_accessibility.
    """
    logger.info("Opening RedBus website at %s", url)
    return check_website_accessibility(page, url=url, timeout_ms=timeout_ms)


# ============================================================================
# 4. RedBus Search Form Interaction
# ============================================================================

SRC_INPUT_SELECTORS = [
    "#srcinput",
    "input[role='combobox'][id*='src']",
    "input#src",
    "div[class*='src'] input",
]

DEST_INPUT_SELECTORS = [
    "#destinput",
    "input[role='combobox'][id*='dest']",
    "input#dest",
    "div[class*='dest'] input",
]

SUGGESTION_SELECTORS = [
    "[role='option']",
    "li[role='option']",
    "div[role='listbox'] li",
    "ul[role='listbox'] li",
    "ul.autoFill li",
    "div[class*='suggestion']",
]

DATE_PICKER_TRIGGER_SELECTORS = [
    "div[aria-label='Select date of journey']",
    "div.dateInputWrapper___874b87",
    "div[class*='dateInputWrapper']",
    "div:has-text('Date of Journey')",
    "#onwardCal",
    "div[class*='doj']",
]

SEARCH_BUTTON_SELECTORS = [
    "button[aria-label='Search buses']",
    "button:has-text('Search buses')",
    "button#search_btn",
    "button#search_button",
    "button.searchButtonWrapper___b89420",
]


def _find_first_matching_locator(page: Page, selectors: List[str], timeout_ms: int = 5000):
    """Helper to locate the first available element among candidate selectors."""
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=1000):
                return loc, sel
        except Exception:
            continue
    return None, None


def enter_source_city(page: Page, city: str, timeout_ms: int = 10000) -> bool:
    """
    Locate source input, enter city name, wait for suggestions, and select matching item.

    Args:
        page (Page): Active Playwright page.
        city (str): Name of origin city.
        timeout_ms (int): Operation timeout.

    Returns:
        bool: True if source city was entered and verified.
    """
    if not city or not city.strip():
        logger.error("Source city cannot be empty")
        return False

    city_clean = city.strip()
    logger.info("Entering source city: '%s'", city_clean)

    input_loc, _ = _find_first_matching_locator(page, SRC_INPUT_SELECTORS, timeout_ms=timeout_ms)
    if not input_loc:
        logger.error("Could not find source input element on page")
        return False

    try:
        input_loc.click()
        input_loc.fill("")
        input_loc.press_sequentially(city_clean, delay=50)

        # Wait for autosuggestions to populate
        selected = False
        for sugg_sel in SUGGESTION_SELECTORS:
            try:
                sugg_loc = page.locator(sugg_sel)
                if sugg_loc.first.is_visible(timeout=2500):
                    # Try matching item text or pick top suggestion
                    matching_opt = page.locator(f"{sugg_sel}:has-text('{city_clean}')").first
                    if matching_opt.is_visible(timeout=1000):
                        matching_opt.click()
                    else:
                        sugg_loc.first.click()
                    selected = True
                    logger.info("Selected suggestion using selector: '%s'", sugg_sel)
                    break
            except Exception:
                continue

        if not selected:
            # Fallback: navigate suggestions with keyboard ArrowDown + Enter
            logger.info("Using keyboard navigation fallback for source autosuggestion")
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")

        # Verification
        val = input_loc.input_value()
        logger.info("Source input value verified as: '%s'", val)
        return bool(val.strip())

    except Exception as e:
        logger.error("Failed to enter source city '%s': %s", city_clean, e)
        return False


def enter_destination_city(page: Page, city: str, timeout_ms: int = 10000) -> bool:
    """
    Locate destination input, enter city name, wait for suggestions, and select matching item.

    Args:
        page (Page): Active Playwright page.
        city (str): Name of destination city.
        timeout_ms (int): Operation timeout.

    Returns:
        bool: True if destination city was entered and verified.
    """
    if not city or not city.strip():
        logger.error("Destination city cannot be empty")
        return False

    city_clean = city.strip()
    logger.info("Entering destination city: '%s'", city_clean)

    input_loc, _ = _find_first_matching_locator(page, DEST_INPUT_SELECTORS, timeout_ms=timeout_ms)
    if not input_loc:
        logger.error("Could not find destination input element on page")
        return False

    try:
        input_loc.click()
        input_loc.fill("")
        input_loc.press_sequentially(city_clean, delay=50)

        # Wait for autosuggestions to populate
        selected = False
        for sugg_sel in SUGGESTION_SELECTORS:
            try:
                sugg_loc = page.locator(sugg_sel)
                if sugg_loc.first.is_visible(timeout=2500):
                    matching_opt = page.locator(f"{sugg_sel}:has-text('{city_clean}')").first
                    if matching_opt.is_visible(timeout=1000):
                        matching_opt.click()
                    else:
                        sugg_loc.first.click()
                    selected = True
                    logger.info("Selected suggestion using selector: '%s'", sugg_sel)
                    break
            except Exception:
                continue

        if not selected:
            logger.info("Using keyboard navigation fallback for destination autosuggestion")
            page.keyboard.press("ArrowDown")
            page.keyboard.press("Enter")

        # Verification
        val = input_loc.input_value()
        logger.info("Destination input value verified as: '%s'", val)
        return bool(val.strip())

    except Exception as e:
        logger.error("Failed to enter destination city '%s': %s", city_clean, e)
        return False


def select_journey_date(page: Page, journey_date: str, timeout_ms: int = 10000) -> bool:
    """
    Select the journey date in the RedBus date picker or quick buttons.

    Args:
        page (Page): Active Playwright page.
        journey_date (str): Date formatted as YYYY-MM-DD.
        timeout_ms (int): Operation timeout.

    Returns:
        bool: True if journey date was selected and verified.
    """
    if not journey_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", journey_date.strip()):
        logger.error("Invalid journey date format '%s'. Expected YYYY-MM-DD", journey_date)
        return False

    try:
        target_dt = datetime.strptime(journey_date.strip(), "%Y-%m-%d").date()
    except ValueError as ve:
        logger.error("Failed to parse journey date '%s': %s", journey_date, ve)
        return False

    logger.info("Selecting journey date: %s", target_dt)

    # Calculate IST current dates
    try:
        today_ist = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    except Exception:
        today_ist = date.today()
    tomorrow_ist = today_ist + timedelta(days=1)

    # Fast-path 1: Quick 'Today' button
    if target_dt == today_ist:
        for today_sel in ["button[aria-label='Search for Today']", "button:has-text('Today')"]:
            try:
                today_btn = page.locator(today_sel).first
                if today_btn.is_visible(timeout=1000):
                    today_btn.click()
                    logger.info("Selected date using quick 'Today' button")
                    return True
            except Exception:
                pass

    # Fast-path 2: Quick 'Tomorrow' button
    if target_dt == tomorrow_ist:
        for tomorrow_sel in ["button[aria-label='Search for Tomorrow']", "button:has-text('Tomorrow')"]:
            try:
                tomorrow_btn = page.locator(tomorrow_sel).first
                if tomorrow_btn.is_visible(timeout=1000):
                    tomorrow_btn.click()
                    logger.info("Selected date using quick 'Tomorrow' button")
                    return True
            except Exception:
                pass

    # Calendar dialog interaction
    trigger_loc, _ = _find_first_matching_locator(page, DATE_PICKER_TRIGGER_SELECTORS, timeout_ms=5000)
    if not trigger_loc:
        logger.error("Could not find calendar trigger element on page")
        return False

    try:
        trigger_loc.click()

        target_day = str(target_dt.day)
        target_month_short = target_dt.strftime("%b")
        target_month_full = target_dt.strftime("%B")
        target_year = str(target_dt.year)

        # Month header check and next month arrow
        for _ in range(12):  # Limit forward navigation to 12 months
            header_text = ""
            for header_sel in ["[class*='monthTitle']", "[class*='MonthYear']", "div[role='heading']", ".month"]:
                try:
                    h_loc = page.locator(header_sel).first
                    if h_loc.is_visible(timeout=500):
                        header_text = h_loc.inner_text().strip()
                        break
                except Exception:
                    pass

            if (target_month_short in header_text or target_month_full in header_text) and target_year in header_text:
                break

            # Click next month if not matching
            next_btn = page.locator("[class*='next'], [aria-label*='Next'], i.icon-arrow-right").first
            if next_btn.is_visible(timeout=500):
                next_btn.click()
            else:
                break

        # Select day cell
        day_selectors = [
            f"span[class*='DayTiles']:has-text('{target_day}')",
            f"div[role='gridcell']:has-text('{target_day}')",
            f"div[class*='day']:has-text('{target_day}')",
            f"span:has-text('{target_day}')",
        ]

        day_selected = False
        for day_sel in day_selectors:
            try:
                day_loc = page.locator(day_sel).first
                if day_loc.is_visible(timeout=1000):
                    day_loc.click()
                    day_selected = True
                    logger.info("Selected calendar day cell using: %s", day_sel)
                    break
            except Exception:
                continue

        return day_selected

    except Exception as e:
        logger.error("Failed to select calendar date '%s': %s", journey_date, e)
        return False


def search_redbus(
    source: str,
    destination: str,
    journey_date: str,
    headless: bool = True,
    max_results: int = 20,
    details: Optional[Any] = None
) -> BusSearchResult:
    """
    Execute full RedBus search workflow:
    1. Launch browser.
    2. Open RedBus homepage and verify accessibility.
    3. Enter source and destination cities.
    4. Select journey date.
    5. Click Search Buses.
    6. Verify results page loading.
    7. Extract bus listings and apply user preference filters.

    Args:
        source (str): Origin city.
        destination (str): Destination city.
        journey_date (str): Date strictly in YYYY-MM-DD format.
        headless (bool): Run browser in headless mode (default True).
        max_results (int): Maximum buses to extract (default 20).
        details: Optional BookingDetails or dict with user travel preferences.

    Returns:
        BusSearchResult: Search result object with status or error message and extracted buses.
    """
    # 1. Input validation
    if not source or not str(source).strip():
        return BusSearchResult(
            source=source or "Unknown",
            destination=destination or "Unknown",
            journey_date=journey_date or "",
            error_message="Source city cannot be empty."
        )

    if not destination or not str(destination).strip():
        return BusSearchResult(
            source=source,
            destination=destination or "Unknown",
            journey_date=journey_date or "",
            error_message="Destination city cannot be empty."
        )

    if not journey_date or not re.match(r"^\d{4}-\d{2}-\d{2}$", str(journey_date).strip()):
        return BusSearchResult(
            source=source,
            destination=destination,
            journey_date=journey_date or "",
            error_message=f"Invalid journey date '{journey_date}'. Expected format YYYY-MM-DD."
        )

    result = BusSearchResult(
        source=source.strip(),
        destination=destination.strip(),
        journey_date=journey_date.strip()
    )

    logger.info("Starting RedBus search: %s -> %s on %s (headless=%s)", result.source, result.destination, result.journey_date, headless)

    try:
        with launch_browser_session(headless=headless) as page:
            # 2. Open website and check access
            access = open_redbus(page)
            if access.get("access_blocked"):
                result.error_message = f"RedBus access restricted: {access.get('block_reason')}"
                logger.warning(result.error_message)
                return result

            if access.get("error"):
                result.error_message = f"Could not connect to RedBus: {access.get('error')}"
                logger.warning(result.error_message)
                return result

            # 3. Enter source city
            if not enter_source_city(page, result.source):
                result.error_message = f"Could not enter source city '{result.source}' on RedBus."
                return result

            # 4. Enter destination city
            if not enter_destination_city(page, result.destination):
                result.error_message = f"Could not enter destination city '{result.destination}' on RedBus."
                return result

            # 5. Select journey date
            if not select_journey_date(page, result.journey_date):
                result.error_message = f"Could not select journey date '{result.journey_date}' on RedBus."
                return result

            # 6. Click search button
            search_btn, _ = _find_first_matching_locator(page, SEARCH_BUTTON_SELECTORS, timeout_ms=5000)
            if not search_btn:
                result.error_message = "Could not find 'Search buses' button on RedBus."
                return result

            logger.info("Clicking Search buses button")
            search_btn.click()

            # 7. Wait for results page navigation
            try:
                page.wait_for_url(lambda u: "/bus-tickets/" in u or "search" in u, timeout=15000)
                logger.info("Search results page loaded successfully: %s", page.url)
            except Exception as nav_err:
                current_url = page.url
                if "/bus-tickets/" in current_url or "search" in current_url:
                    logger.info("Navigation confirmed via URL: %s", current_url)
                else:
                    logger.warning("Results page transition timeout: %s (current URL: %s)", nav_err, current_url)

            # 8. Extract and filter bus listings
            extracted_buses = extract_bus_results(page, max_results=max_results)
            filtered_buses = filter_bus_results(extracted_buses, details=details)

            result.buses = filtered_buses
            result.total_found = len(filtered_buses)
            return result

    except Exception as e:
        err_msg = f"RedBus search automation failed: {str(e)}"
        logger.error(err_msg)
        result.error_message = err_msg
        return result


# ============================================================================
# 5. Bus Results Extraction & Preference Filtering
# ============================================================================

BUS_CARD_SELECTORS = [
    "div.bus-item-details",
    "div[class*='bus-item']",
    "li.row-sec",
    "div.search-container div.bus-item",
    "div[class*='busCard']",
    "tr.tableRow",
]

OPERATOR_SELECTORS = [
    "div.travels",
    "div[class*='travels']",
    "span.travels",
    "div.operator-name",
    "h3",
    "strong",
]

BUS_TYPE_SELECTORS = [
    "div.bus-type",
    "div[class*='bus-type']",
    "span[class*='bus-type']",
    "div.busType",
    "div[class*='bus-spec']",
]

DEPARTURE_TIME_SELECTORS = [
    "div.dp-time",
    "span.dp-time",
    "div[class*='dp-time']",
    "span[class*='dp-time']",
    "div[class*='departureTime']",
]

ARRIVAL_TIME_SELECTORS = [
    "div.bp-time",
    "span.bp-time",
    "div[class*='bp-time']",
    "span[class*='bp-time']",
    "div[class*='arrivalTime']",
]

DURATION_SELECTORS = [
    "div.dur",
    "span.dur",
    "div[class*='dur']",
    "div[class*='duration']",
]

FARE_SELECTORS = [
    "span.f-19",
    "div.fare span",
    "div[class*='fare'] span",
    "span[class*='fare']",
    "span[class*='price']",
    "span[class*='seat-fare']",
]

RATING_SELECTORS = [
    "div.rating span",
    "div[class*='rating'] span",
    "span[class*='rating']",
    "div.rb-rating",
]

SEATS_SELECTORS = [
    "div.seat-left",
    "div[class*='seat-left']",
    "div[class*='seats-left']",
    "span[class*='seat-left']",
]


def _extract_text_by_selectors(locator, selectors: List[str]) -> Optional[str]:
    """Helper to extract text from the first matching sub-selector within a locator."""
    for sel in selectors:
        try:
            sub = locator.locator(sel).first
            text = sub.inner_text().strip()
            if isinstance(text, str) and text:
                return text
        except Exception:
            continue
    return None


def extract_bus_results(page: Page, max_results: int = 20) -> List[BusItem]:
    """
    Extract bus listings from the search results page.

    Args:
        page (Page): Active Playwright page with search results.
        max_results (int): Maximum number of buses to extract (default 20).

    Returns:
        List[BusItem]: List of validated BusItem instances.
    """
    buses: List[BusItem] = []

    # 1. Check for empty-result indicators
    try:
        empty_indicators = [
            "div.no-buses",
            "div[class*='noBuses']",
            "div:has-text('No buses found')",
            "div:has-text('Oops! No buses found')",
        ]
        for empty_sel in empty_indicators:
            empty_loc = page.locator(empty_sel).first
            if empty_loc.is_visible(timeout=500):
                logger.info("Search result indicates no buses available for this route.")
                return buses
    except Exception:
        pass

    # 2. Find matching bus card container
    card_loc = None
    for sel in BUS_CARD_SELECTORS:
        try:
            candidates = page.locator(sel)
            c_count = candidates.count()
            if isinstance(c_count, int) and c_count > 0:
                card_loc = candidates
                logger.info("Located %d bus cards using selector '%s'", c_count, sel)
                break
        except Exception:
            continue

    if not card_loc:
        logger.info("No bus cards found on current page.")
        return buses

    # Smooth scroll to ensure cards in viewport are rendered
    try:
        page.evaluate("window.scrollBy(0, 600)")
    except Exception:
        pass

    try:
        total_cards = card_loc.count()
        count = min(total_cards, max_results) if isinstance(total_cards, int) else 0
    except Exception:
        count = 0

    for i in range(count):
        try:
            card = card_loc.nth(i)
            raw_text = card.inner_text()
            card_text = raw_text.strip() if isinstance(raw_text, str) else ""
            if not card_text:
                continue

            # Extract operator name
            operator = _extract_text_by_selectors(card, OPERATOR_SELECTORS)
            if not operator:
                lines = [l.strip() for l in card_text.split("\n") if l.strip()]
                operator = lines[0] if lines else "Private Bus Operator"

            # Extract bus type
            bus_type = _extract_text_by_selectors(card, BUS_TYPE_SELECTORS)
            if not bus_type:
                if "ac" in card_text.lower():
                    bus_type = "AC Sleeper" if "sleeper" in card_text.lower() else "AC Seater"
                else:
                    bus_type = "Non-AC Sleeper" if "sleeper" in card_text.lower() else "Non-AC Seater"

            # Extract departure and arrival times
            dep_time = _extract_text_by_selectors(card, DEPARTURE_TIME_SELECTORS)
            arr_time = _extract_text_by_selectors(card, ARRIVAL_TIME_SELECTORS)

            times = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", card_text)
            if not dep_time and len(times) >= 1:
                dep_time = f"{times[0][0].zfill(2)}:{times[0][1]}"
            if not arr_time and len(times) >= 2:
                arr_time = f"{times[1][0].zfill(2)}:{times[1][1]}"

            if not dep_time:
                dep_time = "21:00"
            if not arr_time:
                arr_time = "06:00"

            # Extract duration
            duration = _extract_text_by_selectors(card, DURATION_SELECTORS)
            if not duration:
                dur_match = re.search(r"(\d+\s*h(?:rs?)?\s*(?:\d+\s*m(?:ins?)?)?)", card_text, re.I)
                duration = dur_match.group(1) if dur_match else "08h 00m"

            # Extract fare
            fare_text = _extract_text_by_selectors(card, FARE_SELECTORS)
            if not fare_text:
                fare_match = re.search(r"(?:₹|Rs\.?|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)", card_text, re.I)
                if not fare_match:
                    fare_match = re.search(r"\b(\d{3,5})\b", card_text)
                fare_text = fare_match.group(1) if fare_match else "0"

            try:
                fare_val = BusItem.validate_and_clean_fare(fare_text)
            except ValueError:
                continue

            # Extract rating & seats (optional)
            rating = _extract_text_by_selectors(card, RATING_SELECTORS)
            if not rating:
                rat_match = re.search(r"\b([1-5]\.\d)\b", card_text)
                rating = rat_match.group(1) if rat_match else None

            seats = _extract_text_by_selectors(card, SEATS_SELECTORS)
            if not seats:
                seat_match = re.search(r"(\d+\s*Seats?(?:\s*available|\s*left)?)", card_text, re.I)
                seats = seat_match.group(1) if seat_match else None

            bus_item = BusItem(
                operator_name=operator,
                bus_type=bus_type,
                departure_time=dep_time,
                arrival_time=arr_time,
                duration=duration,
                fare=fare_val,
                rating=rating,
                seats_available=seats
            )
            buses.append(bus_item)

        except Exception as item_err:
            logger.warning("Error parsing individual bus card %d: %s", i, item_err)
            continue

    logger.info("Successfully extracted %d bus listings.", len(buses))
    return buses


def filter_bus_results(buses: List[BusItem], details: Optional[Any] = None) -> List[BusItem]:
    """
    Filter bus listings based on user travel preferences:
    - bus_type: AC vs Non-AC
    - seating_preference: Sleeper vs Seater
    - max_budget: maximum fare limit
    - preferred_time: morning, afternoon, evening, night

    Args:
        buses (List[BusItem]): Raw extracted bus listings.
        details: BookingDetails instance, dict, or object with travel preferences.

    Returns:
        List[BusItem]: Filtered and price-sorted bus listings.
    """
    if not buses or not details:
        return sorted(buses, key=lambda b: b.fare)

    # Safely extract preference attributes
    if isinstance(details, dict):
        bus_type_pref = details.get("bus_type")
        seating_pref = details.get("seating_preference")
        max_budget = details.get("max_budget")
        pref_time = details.get("preferred_time")
    else:
        bus_type_pref = getattr(details, "bus_type", None)
        seating_pref = getattr(details, "seating_preference", None)
        max_budget = getattr(details, "max_budget", None)
        pref_time = getattr(details, "preferred_time", None)

    filtered = list(buses)

    # 1. AC / Non-AC filter
    if bus_type_pref:
        bus_type_lower = bus_type_pref.strip().lower()
        if "non" in bus_type_lower:
            filtered = [b for b in filtered if "non" in b.bus_type.lower()]
        elif "ac" in bus_type_lower:
            filtered = [b for b in filtered if "ac" in b.bus_type.lower() and "non" not in b.bus_type.lower()]

    # 2. Seating preference (Sleeper / Seater)
    if seating_pref:
        seating_lower = seating_pref.strip().lower()
        if "sleeper" in seating_lower:
            filtered = [b for b in filtered if "sleeper" in b.bus_type.lower()]
        elif "seater" in seating_lower or "sitting" in seating_lower:
            filtered = [b for b in filtered if "seater" in b.bus_type.lower() or "sitting" in b.bus_type.lower()]

    # 3. Maximum budget limit
    if max_budget is not None:
        try:
            budget_val = float(max_budget)
            if budget_val > 0:
                filtered = [b for b in filtered if b.fare <= budget_val]
        except (ValueError, TypeError):
            pass

    # 4. Preferred departure time
    if pref_time:
        p_time = str(pref_time).strip().lower()
        time_filtered = []
        for b in filtered:
            try:
                hour = int(b.departure_time.split(":")[0])
                if "morning" in p_time and (5 <= hour < 12):
                    time_filtered.append(b)
                elif "afternoon" in p_time and (12 <= hour < 17):
                    time_filtered.append(b)
                elif "evening" in p_time and (17 <= hour < 21):
                    time_filtered.append(b)
                elif "night" in p_time and (hour >= 21 or hour < 5):
                    time_filtered.append(b)
            except Exception:
                time_filtered.append(b)

        if time_filtered:
            filtered = time_filtered

    # Sort remaining buses by fare ascending
    filtered.sort(key=lambda b: b.fare)
    return filtered


def format_bus_results(result: BusSearchResult) -> str:
    """
    Format extracted bus search results into a clean, readable Markdown table.

    Args:
        result (BusSearchResult): Result object containing route and extracted buses.

    Returns:
        str: GitHub-flavored markdown representation.
    """
    if result.error_message:
        return f"\n\n---\n⚠️ **Bus Search Notice:** {result.error_message}"

    if not result.buses:
        return (
            f"\n\n---\n"
            f"🔍 **No buses found** for **{result.source} ➔ {result.destination}** on **{result.journey_date}**.\n"
            f"*Try modifying your route, journey date, or relaxing your budget/bus type filters.*"
        )

    lines = [
        f"\n\n---\n",
        f"🚌 **Available Buses ({result.source} ➔ {result.destination} • {result.journey_date}):**\n",
        f"| Operator | Bus Type | Departs | Arrives | Duration | Fare | Rating | Seats |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for b in result.buses:
        rating_str = f"⭐ {b.rating}" if b.rating else "—"
        seats_str = b.seats_available if b.seats_available else "Available"
        fare_str = f"₹{int(b.fare):,}" if b.fare.is_integer() else f"₹{b.fare:,.2f}"

        lines.append(
            f"| **{b.operator_name}** | {b.bus_type} | {b.departure_time} | {b.arrival_time} | {b.duration} | **{fare_str}** | {rating_str} | {seats_str} |"
        )

    lines.append(f"\n*Showing top {len(result.buses)} buses matching your travel criteria.*")
    return "\n".join(lines)
