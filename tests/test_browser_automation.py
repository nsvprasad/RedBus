"""
test_browser_automation.py - Unit, foundation, and search workflow tests for Module 3 Browser Automation.
"""

import os
import sys
import unittest
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# Ensure project root is discoverable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.browser_automation import (
    BusItem,
    BusSearchResult,
    launch_browser_session,
    init_browser,
    close_browser,
    check_website_accessibility,
    open_redbus,
    enter_source_city,
    enter_destination_city,
    select_journey_date,
    search_redbus,
    REDBUS_HOME_URL,
)


class TestBrowserAutomationModels(unittest.TestCase):
    """Tests for Pydantic models (BusItem, BusSearchResult)."""

    def test_01_valid_bus_item(self):
        """Test valid BusItem creation and fare normalization."""
        item = BusItem(
            operator_name="IntrCity SmartBus",
            bus_type="AC Sleeper (2+1)",
            departure_time="21:30",
            arrival_time="05:45",
            duration="08h 15m",
            fare="₹1,250",
            rating="4.4",
            seats_available="14 Seats"
        )
        self.assertEqual(item.operator_name, "IntrCity SmartBus")
        self.assertEqual(item.bus_type, "AC Sleeper (2+1)")
        self.assertEqual(item.fare, 1250.0)
        self.assertEqual(item.rating, "4.4")
        self.assertEqual(item.seats_available, "14 Seats")

    def test_02_bus_item_numeric_fare(self):
        """Test numeric float/int fare values."""
        item = BusItem(
            operator_name="SRS Travels",
            bus_type="Non-AC Seater",
            departure_time="06:00",
            arrival_time="11:30",
            duration="05h 30m",
            fare=750
        )
        self.assertEqual(item.fare, 750.0)
        self.assertIsNone(item.rating)
        self.assertIsNone(item.seats_available)

    def test_03_invalid_negative_fare(self):
        """Test that negative fares raise ValidationError."""
        with self.assertRaises(ValidationError):
            BusItem(
                operator_name="SRS Travels",
                bus_type="Non-AC Seater",
                departure_time="06:00",
                arrival_time="11:30",
                duration="05h 30m",
                fare=-50.0
            )

    def test_04_invalid_fare_string(self):
        """Test that non-numeric fare strings raise ValidationError."""
        with self.assertRaises(ValidationError):
            BusItem(
                operator_name="SRS Travels",
                bus_type="Non-AC Seater",
                departure_time="06:00",
                arrival_time="11:30",
                duration="05h 30m",
                fare="Free Bus"
            )

    def test_05_empty_string_fields(self):
        """Test that required core string fields cannot be blank."""
        with self.assertRaises(ValidationError):
            BusItem(
                operator_name="   ",
                bus_type="AC Sleeper",
                departure_time="21:00",
                arrival_time="06:00",
                duration="09h 00m",
                fare=1000.0
            )

    def test_06_bus_search_result_container(self):
        """Test BusSearchResult model with multiple bus items."""
        item1 = BusItem(
            operator_name="Kaveri Travels",
            bus_type="AC Sleeper",
            departure_time="22:00",
            arrival_time="06:30",
            duration="08h 30m",
            fare=1100.0
        )
        item2 = BusItem(
            operator_name="VRL Travels",
            bus_type="AC Semi-Sleeper",
            departure_time="23:00",
            arrival_time="07:00",
            duration="08h 00m",
            fare=950.0
        )
        result = BusSearchResult(
            source="chennai",
            destination="bangalore",
            journey_date="2026-11-20",
            total_found=2,
            buses=[item1, item2]
        )
        self.assertEqual(result.source, "Chennai")
        self.assertEqual(result.destination, "Bangalore")
        self.assertEqual(result.total_found, 2)
        self.assertEqual(len(result.buses), 2)
        self.assertIsNone(result.error_message)


class TestBrowserLifecycleAndAccessibility(unittest.TestCase):
    """Tests for browser lifecycle management and accessibility checks."""

    def test_07_access_denied_detection_logic(self):
        """Test detection of blocked or security challenge screens using a mock page."""
        mock_page = MagicMock()
        mock_page.goto.return_value = MagicMock(status=403)
        mock_page.url = "https://www.redbus.in"
        mock_page.title.return_value = "Access Denied"

        res = check_website_accessibility(mock_page, "https://www.redbus.in")
        self.assertFalse(res["is_accessible"])
        self.assertTrue(res["access_blocked"])
        self.assertIn("403", res["block_reason"])

    def test_08_cloudflare_challenge_detection(self):
        """Test detection of Cloudflare waiting room challenge in page title."""
        mock_page = MagicMock()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.url = "https://www.redbus.in"
        mock_page.title.return_value = "Just a moment... Attention Required!"
        mock_page.locator.return_value.inner_text.return_value = "Verify you are human"

        res = check_website_accessibility(mock_page, "https://www.redbus.in")
        self.assertFalse(res["is_accessible"])
        self.assertTrue(res["access_blocked"])

    def test_09_browser_launch_and_cleanup(self):
        """Test that Playwright Chromium initializes and closes cleanly via context manager."""
        try:
            with launch_browser_session(headless=True) as page:
                self.assertIsNotNone(page)
                page.goto("about:blank")
                self.assertEqual(page.url, "about:blank")
        except Exception as e:
            self.fail(f"Browser launch and cleanup failed: {e}")

    def test_10_redbus_accessibility_or_graceful_handling(self):
        """
        Test open_redbus function.
        Does NOT strictly require live network connection: if network is blocked/offline,
        it must gracefully record the status/error without raising an uncaught exception.
        """
        try:
            with launch_browser_session(headless=True) as page:
                res = open_redbus(page, url=REDBUS_HOME_URL, timeout_ms=10000)
                self.assertIsInstance(res, dict)
                self.assertIn("is_accessible", res)
                self.assertIn("access_blocked", res)
                self.assertIn("url", res)
        except Exception as e:
            self.fail(f"open_redbus raised unexpected exception: {e}")


class TestRedBusSearchAutomation(unittest.TestCase):
    """Tests for Module 3.2: RedBus search interaction flow."""

    def test_11_input_validation_empty_cities(self):
        """Test enter_source_city and enter_destination_city reject blank strings."""
        mock_page = MagicMock()
        self.assertFalse(enter_source_city(mock_page, ""))
        self.assertFalse(enter_source_city(mock_page, "   "))
        self.assertFalse(enter_destination_city(mock_page, ""))
        self.assertFalse(enter_destination_city(mock_page, "   "))

    def test_12_journey_date_validation(self):
        """Test select_journey_date rejects malformed date strings."""
        mock_page = MagicMock()
        self.assertFalse(select_journey_date(mock_page, "invalid-date"))
        self.assertFalse(select_journey_date(mock_page, "2026/11/15"))
        self.assertFalse(select_journey_date(mock_page, "15-11-2026"))
        self.assertFalse(select_journey_date(mock_page, "2026-02-31"))  # non-existent calendar date

    def test_13_search_redbus_input_validation(self):
        """Test search_redbus validates inputs before launching browser."""
        res1 = search_redbus("", "Bangalore", "2026-11-20")
        self.assertIn("Source city cannot be empty", res1.error_message)

        res2 = search_redbus("Chennai", "", "2026-11-20")
        self.assertIn("Destination city cannot be empty", res2.error_message)

        res3 = search_redbus("Chennai", "Bangalore", "20-11-2026")
        self.assertIn("Invalid journey date", res3.error_message)

    def test_14_mock_enter_source_city_success(self):
        """Test enter_source_city successful execution with mocked locator."""
        mock_page = MagicMock()
        mock_input = MagicMock()
        mock_input.is_visible.return_value = True
        mock_input.input_value.return_value = "Chennai"

        mock_sugg = MagicMock()
        mock_sugg.first.is_visible.return_value = True

        def locator_side_effect(selector):
            if "src" in selector:
                mock_input.first = mock_input
                return mock_input
            return mock_sugg

        mock_page.locator.side_effect = locator_side_effect
        success = enter_source_city(mock_page, "Chennai")
        self.assertTrue(success)
        mock_input.click.assert_called()
        mock_input.press_sequentially.assert_called_with("Chennai", delay=50)

    def test_15_mock_enter_destination_city_success(self):
        """Test enter_destination_city successful execution with mocked locator."""
        mock_page = MagicMock()
        mock_input = MagicMock()
        mock_input.is_visible.return_value = True
        mock_input.input_value.return_value = "Bangalore"

        mock_sugg = MagicMock()
        mock_sugg.first.is_visible.return_value = True

        def locator_side_effect(selector):
            if "dest" in selector:
                mock_input.first = mock_input
                return mock_input
            return mock_sugg

        mock_page.locator.side_effect = locator_side_effect
        success = enter_destination_city(mock_page, "Bangalore")
        self.assertTrue(success)
        mock_input.click.assert_called()

    def test_16_mock_select_journey_date_today_and_tomorrow(self):
        """Test select_journey_date fast-path buttons for today and tomorrow."""
        mock_page = MagicMock()
        mock_btn = MagicMock()
        mock_btn.first.is_visible.return_value = True
        mock_page.locator.return_value = mock_btn

        try:
            today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
            tomorrow_str = (datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            today_str = date.today().strftime("%Y-%m-%d")
            tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Today selection
        res_today = select_journey_date(mock_page, today_str)
        self.assertTrue(res_today)

        # Tomorrow selection
        res_tomorrow = select_journey_date(mock_page, tomorrow_str)
        self.assertTrue(res_tomorrow)

    def test_17_mock_select_journey_date_calendar_popup(self):
        """Test select_journey_date interaction with calendar popup for future dates."""
        mock_page = MagicMock()
        mock_el = MagicMock()
        mock_el.first.is_visible.return_value = True
        mock_el.first.inner_text.return_value = "Dec 2026"
        mock_page.locator.return_value = mock_el

        # Arbitrary future date (e.g. 2026-12-25)
        success = select_journey_date(mock_page, "2026-12-25")
        self.assertTrue(success)

    @patch("modules.browser_automation.launch_browser_session")
    @patch("modules.browser_automation.open_redbus")
    @patch("modules.browser_automation.enter_source_city")
    @patch("modules.browser_automation.enter_destination_city")
    @patch("modules.browser_automation.select_journey_date")
    def test_18_mock_search_redbus_full_workflow(
        self, mock_date, mock_dest, mock_src, mock_open, mock_launch
    ):
        """Test full search_redbus workflow with mocked browser session."""
        mock_page = MagicMock()
        mock_page.url = "https://www.redbus.in/bus-tickets/chennai-to-bangalore"
        mock_btn = MagicMock()
        mock_btn.first.is_visible.return_value = True
        mock_page.locator.return_value = mock_btn

        mock_launch.return_value.__enter__.return_value = mock_page
        mock_open.return_value = {"is_accessible": True, "access_blocked": False, "error": None}
        mock_src.return_value = True
        mock_dest.return_value = True
        mock_date.return_value = True

        result = search_redbus("Chennai", "Bangalore", "2026-11-25", headless=True)
        self.assertIsInstance(result, BusSearchResult)
        self.assertEqual(result.source, "Chennai")
        self.assertEqual(result.destination, "Bangalore")
        self.assertEqual(result.journey_date, "2026-11-25")
        self.assertIsNone(result.error_message)

    @patch("modules.browser_automation.launch_browser_session")
    @patch("modules.browser_automation.open_redbus")
    def test_19_search_redbus_blocked_handling(self, mock_open, mock_launch):
        """Test search_redbus gracefully handles anti-bot challenge without crashing."""
        mock_page = MagicMock()
        mock_launch.return_value.__enter__.return_value = mock_page
        mock_open.return_value = {
            "is_accessible": False,
            "access_blocked": True,
            "block_reason": "Detected security challenge matching 'challenge-platform'",
            "error": None
        }

        result = search_redbus("Chennai", "Bangalore", "2026-11-25", headless=True)
        self.assertIsNotNone(result.error_message)
        self.assertIn("restricted", result.error_message)

    @patch("modules.browser_automation.launch_browser_session")
    @patch("modules.browser_automation.open_redbus")
    def test_20_search_redbus_network_error_handling(self, mock_open, mock_launch):
        """Test search_redbus gracefully handles network failure."""
        mock_page = MagicMock()
        mock_launch.return_value.__enter__.return_value = mock_page
        mock_open.return_value = {
            "is_accessible": False,
            "access_blocked": False,
            "error": "net::ERR_INTERNET_DISCONNECTED"
        }

        result = search_redbus("Chennai", "Bangalore", "2026-11-25", headless=True)
        self.assertIsNotNone(result.error_message)
        self.assertIn("Could not connect to RedBus", result.error_message)


if __name__ == "__main__":
    unittest.main()
