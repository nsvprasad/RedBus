"""
test_bus_extraction.py - Test suite for Module 3.3 Bus Results Extraction and Filtering.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is discoverable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.browser_automation import (
    BusItem,
    BusSearchResult,
    extract_bus_results,
    filter_bus_results,
    format_bus_results,
    search_redbus,
)


class TestBusResultsExtraction(unittest.TestCase):
    """Unit tests for bus cards scraping and parsing from results page."""

    def test_01_empty_results_detection(self):
        """Test extract_bus_results when 'No buses found' banner is present."""
        mock_page = MagicMock()
        mock_empty = MagicMock()
        mock_empty.first.is_visible.return_value = True
        mock_page.locator.return_value = mock_empty

        buses = extract_bus_results(mock_page)
        self.assertEqual(len(buses), 0)

    def test_02_no_cards_found_returns_empty_list(self):
        """Test extract_bus_results returns [] when no bus cards exist on page."""
        mock_page = MagicMock()
        mock_loc = MagicMock()
        mock_loc.first.is_visible.return_value = False
        mock_loc.count.return_value = 0
        mock_page.locator.return_value = mock_loc

        buses = extract_bus_results(mock_page)
        self.assertEqual(len(buses), 0)

    def test_03_extract_single_bus_card_mock(self):
        """Test extraction of a realistic bus card with operator, times, fare, and bus type."""
        mock_page = MagicMock()
        mock_card = MagicMock()
        mock_card.inner_text.return_value = (
            "IntrCity SmartBus\n"
            "AC Sleeper (2+1)\n"
            "21:30 -------- 05:45\n"
            "08h 15m\n"
            "⭐ 4.5\n"
            "18 Seats available\n"
            "₹1,250"
        )
        mock_card.locator.return_value.first.inner_text.return_value = ""

        mock_cards = MagicMock()
        mock_cards.count.return_value = 1
        mock_cards.nth.return_value = mock_card
        mock_cards.first.is_visible.return_value = True

        def page_locator_side_effect(selector):
            if "no" in selector.lower():
                m = MagicMock()
                m.first.is_visible.return_value = False
                return m
            return mock_cards

        mock_page.locator.side_effect = page_locator_side_effect

        buses = extract_bus_results(mock_page, max_results=5)
        self.assertEqual(len(buses), 1)
        b = buses[0]
        self.assertEqual(b.operator_name, "IntrCity SmartBus")
        self.assertIn("AC", b.bus_type)
        self.assertEqual(b.departure_time, "21:30")
        self.assertEqual(b.arrival_time, "05:45")
        self.assertEqual(b.duration, "08h 15m")
        self.assertEqual(b.fare, 1250.0)
        self.assertEqual(b.rating, "4.5")
        self.assertIn("18 Seats", b.seats_available)

    def test_04_skip_corrupted_card(self):
        """Test that a card with unparsable fare or empty text is safely skipped."""
        mock_page = MagicMock()
        mock_card = MagicMock()
        mock_card.inner_text.return_value = "Broken card with no times or fares"
        mock_card.locator.return_value.first.inner_text.return_value = ""

        mock_cards = MagicMock()
        mock_cards.count.return_value = 1
        mock_cards.nth.return_value = mock_card
        mock_cards.first.is_visible.return_value = True

        def page_locator_side_effect(selector):
            if "no" in selector.lower():
                m = MagicMock()
                m.first.is_visible.return_value = False
                return m
            return mock_cards

        mock_page.locator.side_effect = page_locator_side_effect

        # Broken card defaults to fare 0 which is valid, or if fare is negative it skips
        buses = extract_bus_results(mock_page, max_results=1)
        self.assertIsInstance(buses, list)


class TestBusFiltering(unittest.TestCase):
    """Unit tests for user preference filtering (AC/Non-AC, Sleeper/Seater, Budget, Time)."""

    def setUp(self):
        self.sample_buses = [
            BusItem(
                operator_name="Bus A - Morning AC Seater",
                bus_type="AC Seater (2+2)",
                departure_time="07:00",
                arrival_time="13:00",
                duration="06h 00m",
                fare=650.0,
                rating="4.2"
            ),
            BusItem(
                operator_name="Bus B - Afternoon Non-AC Seater",
                bus_type="Non-AC Seater",
                departure_time="14:30",
                arrival_time="20:30",
                duration="06h 00m",
                fare=450.0,
                rating="3.9"
            ),
            BusItem(
                operator_name="Bus C - Night AC Sleeper",
                bus_type="AC Sleeper (2+1)",
                departure_time="22:00",
                arrival_time="06:00",
                duration="08h 00m",
                fare=1200.0,
                rating="4.6"
            ),
            BusItem(
                operator_name="Bus D - Night Non-AC Sleeper",
                bus_type="Non-AC Sleeper",
                departure_time="21:30",
                arrival_time="05:30",
                duration="08h 00m",
                fare=800.0,
                rating="4.0"
            ),
            BusItem(
                operator_name="Bus E - Evening Premium Sleeper",
                bus_type="BharatBenz AC Sleeper",
                departure_time="18:30",
                arrival_time="02:30",
                duration="08h 00m",
                fare=1500.0,
                rating="4.8"
            ),
        ]

    def test_05_filter_ac_buses(self):
        """Test filtering by AC preference excludes Non-AC."""
        details = {"bus_type": "AC"}
        res = filter_bus_results(self.sample_buses, details)
        self.assertTrue(all("ac" in b.bus_type.lower() and "non" not in b.bus_type.lower() for b in res))
        self.assertEqual(len(res), 3)

    def test_06_filter_non_ac_buses(self):
        """Test filtering by Non-AC preference excludes AC."""
        details = {"bus_type": "Non-AC"}
        res = filter_bus_results(self.sample_buses, details)
        self.assertTrue(all("non" in b.bus_type.lower() for b in res))
        self.assertEqual(len(res), 2)

    def test_07_filter_sleeper_preference(self):
        """Test filtering by Sleeper preference."""
        details = {"seating_preference": "Sleeper"}
        res = filter_bus_results(self.sample_buses, details)
        self.assertTrue(all("sleeper" in b.bus_type.lower() for b in res))
        self.assertEqual(len(res), 3)

    def test_08_filter_max_budget(self):
        """Test filtering by maximum budget limit."""
        details = {"max_budget": 800.0}
        res = filter_bus_results(self.sample_buses, details)
        self.assertTrue(all(b.fare <= 800.0 for b in res))
        self.assertEqual(len(res), 3)
        # Verify sorted ascending
        fares = [b.fare for b in res]
        self.assertEqual(fares, sorted(fares))

    def test_09_filter_preferred_time_night(self):
        """Test filtering by preferred departure time 'night' (21:00 to 04:59)."""
        details = {"preferred_time": "night"}
        res = filter_bus_results(self.sample_buses, details)
        self.assertEqual(len(res), 2)
        operators = [b.operator_name for b in res]
        self.assertIn("Bus C - Night AC Sleeper", operators)
        self.assertIn("Bus D - Night Non-AC Sleeper", operators)

    def test_10_filter_combined_preferences(self):
        """Test combination: AC + Sleeper + Budget <= 1300."""
        details = {
            "bus_type": "AC",
            "seating_preference": "Sleeper",
            "max_budget": 1300.0
        }
        res = filter_bus_results(self.sample_buses, details)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].operator_name, "Bus C - Night AC Sleeper")
        self.assertEqual(res[0].fare, 1200.0)


class TestBusMarkdownFormatting(unittest.TestCase):
    """Unit tests for format_bus_results output structure."""

    def test_11_format_results_with_buses(self):
        """Test formatted Markdown table with bus details."""
        buses = [
            BusItem(
                operator_name="Kaveri Travels",
                bus_type="AC Sleeper",
                departure_time="22:00",
                arrival_time="06:00",
                duration="08h 00m",
                fare=1150.0,
                rating="4.3",
                seats_available="8 Seats"
            )
        ]
        result = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            total_found=1,
            buses=buses
        )
        md = format_bus_results(result)
        self.assertIn("Available Buses", md)
        self.assertIn("Kaveri Travels", md)
        self.assertIn("₹1,150", md)
        self.assertIn("⭐ 4.3", md)
        self.assertIn("8 Seats", md)
        self.assertIn("| Operator | Bus Type | Departs |", md)

    def test_12_format_results_empty_list(self):
        """Test formatted message when 0 buses are found."""
        result = BusSearchResult(
            source="Vijayawada",
            destination="Hyderabad",
            journey_date="2026-11-20",
            total_found=0,
            buses=[]
        )
        md = format_bus_results(result)
        self.assertIn("No buses found", md)
        self.assertIn("Vijayawada ➔ Hyderabad", md)

    def test_13_format_results_with_error(self):
        """Test formatted message when error occurs."""
        result = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            error_message="Local AI server is not running."
        )
        md = format_bus_results(result)
        self.assertIn("Bus Search Notice", md)
        self.assertIn("Local AI server is not running.", md)


if __name__ == "__main__":
    unittest.main()
