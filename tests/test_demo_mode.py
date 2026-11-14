"""
test_demo_mode.py - Test suite for Demonstration Mode (Module 3.5).

Covers:
- Demo data loading and Pydantic validation (all 8 fields).
- Demo data preference filtering (AC/Non-AC, Sleeper/Seater, budget, timing).
- Source separation (live vs demo isolation; no mock leak into live search).
- Prominent disclaimer labels ("DEMO DATA – NOT LIVE REDBUS AVAILABILITY").
- Chat logic fallback, activation ("show demo"), and state preservation.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.browser_automation import BusItem, BusSearchResult
from modules.demo_data import (
    load_raw_demo_buses,
    get_demo_buses,
    format_demo_bus_results,
    DEMO_DATA_LABEL,
)
from modules.chat_logic import (
    get_bot_response,
    get_active_booking_details,
    reset_conversation_state,
    is_demo_mode_active,
    set_demo_mode,
)
from modules.ai_agent import BookingDetails


class TestDemoDataLoading(unittest.TestCase):
    """Tests for loading and validating sample bus records."""

    def test_01_load_raw_demo_buses_chennai_bangalore(self):
        """Verify each demo bus record has all 8 fields properly populated and validated."""
        buses = load_raw_demo_buses("Chennai", "Bangalore")
        self.assertGreater(len(buses), 0)

        for bus in buses:
            self.assertIsInstance(bus, BusItem)
            self.assertTrue(bool(bus.operator_name.strip()))
            self.assertTrue(bool(bus.bus_type.strip()))
            self.assertRegex(bus.departure_time, r"^\d{2}:\d{2}$")
            self.assertRegex(bus.arrival_time, r"^\d{2}:\d{2}$")
            self.assertTrue(bool(bus.duration.strip()))
            self.assertGreater(bus.fare, 0.0)
            self.assertIsNotNone(bus.rating)
            self.assertIsNotNone(bus.seats_available)

    def test_02_load_raw_demo_buses_generic_fallback(self):
        """Verify unrecognized routes receive generic representative demo records."""
        buses = load_raw_demo_buses("Jaipur", "Udaipur")
        self.assertGreater(len(buses), 0)
        for bus in buses:
            self.assertIsInstance(bus, BusItem)
            self.assertGreater(bus.fare, 0.0)

    def test_03_get_demo_buses_result_structure(self):
        """Verify get_demo_buses returns a structured BusSearchResult."""
        result = get_demo_buses("Chennai", "Bangalore", "2026-11-20")
        self.assertIsInstance(result, BusSearchResult)
        self.assertEqual(result.source, "Chennai")
        self.assertEqual(result.destination, "Bangalore")
        self.assertEqual(result.journey_date, "2026-11-20")
        self.assertGreater(result.total_found, 0)
        self.assertEqual(len(result.buses), result.total_found)
        self.assertIsNone(result.error_message)


class TestDemoDataFiltering(unittest.TestCase):
    """Tests for user preference filtering on demo records."""

    def test_04_filter_ac_vs_non_ac(self):
        """Verify AC and Non-AC filtering on demo data."""
        # AC only
        details_ac = BookingDetails(bus_type="AC", status="incomplete")
        res_ac = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_ac)
        self.assertGreater(len(res_ac.buses), 0)
        for b in res_ac.buses:
            self.assertIn("ac", b.bus_type.lower())
            self.assertNotIn("non", b.bus_type.lower())

        # Non-AC only
        details_non_ac = BookingDetails(bus_type="Non-AC", status="incomplete")
        res_non_ac = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_non_ac)
        self.assertGreater(len(res_non_ac.buses), 0)
        for b in res_non_ac.buses:
            self.assertIn("non", b.bus_type.lower())

    def test_05_filter_seating_preference(self):
        """Verify Sleeper and Seater preference filtering on demo data."""
        details_sleeper = BookingDetails(seating_preference="Sleeper", status="incomplete")
        res_sleeper = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_sleeper)
        self.assertGreater(len(res_sleeper.buses), 0)
        for b in res_sleeper.buses:
            self.assertIn("sleeper", b.bus_type.lower())

    def test_06_filter_max_budget(self):
        """Verify maximum budget filtering on demo data."""
        details_budget = BookingDetails(max_budget=900.0, status="incomplete")
        res_budget = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_budget)
        self.assertGreater(len(res_budget.buses), 0)
        for b in res_budget.buses:
            self.assertLessEqual(b.fare, 900.0)

    def test_07_filter_preferred_time(self):
        """Verify departure time filtering on demo data."""
        details_morning = BookingDetails(preferred_time="morning", status="incomplete")
        res_morning = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_morning)
        for b in res_morning.buses:
            hour = int(b.departure_time.split(":")[0])
            self.assertTrue(5 <= hour < 12)

    def test_08_filter_no_matching_buses(self):
        """Verify empty result formatting when no buses meet strict filters."""
        details_strict = BookingDetails(max_budget=100.0, status="incomplete")
        res_strict = get_demo_buses("Chennai", "Bangalore", "2026-11-20", details=details_strict)
        self.assertEqual(len(res_strict.buses), 0)
        formatted = format_demo_bus_results(res_strict)
        self.assertIn(DEMO_DATA_LABEL, formatted)
        self.assertIn("No demonstration buses match your specific filters", formatted)


class TestDemoSourceSeparation(unittest.TestCase):
    """Tests ensuring complete isolation between live search and demo mode."""

    def test_09_search_redbus_does_not_return_demo_data(self):
        """Verify live search_redbus never injects fake or demo buses on failure."""
        from modules.browser_automation import search_redbus
        # Trigger an invalid/failing live search
        res = search_redbus("", "Bangalore", "2026-11-20")
        self.assertEqual(res.total_found, 0)
        self.assertEqual(len(res.buses), 0)
        self.assertIsNotNone(res.error_message)

    def test_10_demo_results_prominently_labeled(self):
        """Verify demo results always carry the mandatory DEMO DATA label and disclaimer."""
        res = get_demo_buses("Chennai", "Bangalore", "2026-11-20")
        formatted = format_demo_bus_results(res)
        self.assertIn(DEMO_DATA_LABEL, formatted)
        self.assertIn("DEMO DATA – NOT LIVE REDBUS AVAILABILITY", formatted)
        self.assertIn("simulated demonstration", formatted)
        self.assertIn("do not reflect real-time live availability", formatted)


class TestDemoChatLogicIntegration(unittest.TestCase):
    """Tests for conversational demo mode triggering, fallback, and state preservation."""

    def setUp(self):
        reset_conversation_state()

    def tearDown(self):
        reset_conversation_state()

    def test_11_live_failure_offers_demo_mode(self):
        """Verify that when live search fails, the response explicitly offers demo mode."""
        mock_blocked = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            error_message="Could not connect to RedBus: net::ERR_HTTP2_PROTOCOL_ERROR"
        )

        with patch("modules.chat_logic.search_redbus", return_value=mock_blocked):
            resp = get_bot_response("Chennai to Bangalore on 2026-11-20")
            self.assertIn("Bus Search Notice", resp)
            self.assertIn("Demonstration Mode Available", resp)
            self.assertIn("show demo", resp)

    def test_12_show_demo_activates_demo_results(self):
        """Verify that replying 'show demo' after live failure immediately shows demo buses for the route."""
        mock_blocked = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            error_message="Could not connect to RedBus: net::ERR_HTTP2_PROTOCOL_ERROR"
        )

        with patch("modules.chat_logic.search_redbus", return_value=mock_blocked):
            # Turn 1: User requests route -> live search fails -> offers demo
            resp1 = get_bot_response("Chennai to Bangalore on 2026-11-20")
            self.assertIn("Demonstration Mode Available", resp1)

            # Turn 2: User responds "show demo" -> returns demo listings without re-asking route
            resp2 = get_bot_response("show demo")
            self.assertIn("DEMO DATA – NOT LIVE REDBUS AVAILABILITY", resp2)
            self.assertIn("Demonstration Buses", resp2)
            self.assertIn("Chennai", resp2)
            self.assertIn("Bangalore", resp2)
            self.assertTrue(is_demo_mode_active())

    def test_13_demo_mode_preference_filtering_across_turns(self):
        """Verify multi-turn preference adjustments on active demo results."""
        # Enable demo mode and submit route
        set_demo_mode(True)
        resp1 = get_bot_response("Chennai to Bangalore on 2026-11-20")
        self.assertIn("DEMO DATA – NOT LIVE REDBUS AVAILABILITY", resp1)

        # Refine preference: Non-AC under 600
        resp2 = get_bot_response("Only Non-AC under 600")
        self.assertIn("DEMO DATA – NOT LIVE REDBUS AVAILABILITY", resp2)
        self.assertIn("SRS Travels", resp2)
        self.assertIn("₹550", resp2)

    def test_14_exit_demo_mode_command(self):
        """Verify 'exit demo' and 'live mode' commands switch back to live search."""
        set_demo_mode(True)
        self.assertTrue(is_demo_mode_active())

        resp = get_bot_response("exit demo")
        self.assertIn("Switched back to Live Search mode", resp)
        self.assertFalse(is_demo_mode_active())

    def test_15_reset_clears_demo_mode_and_booking(self):
        """Verify reset clears both booking state and demo mode state."""
        set_demo_mode(True)
        get_bot_response("Chennai to Bangalore on 2026-11-20")
        self.assertIsNotNone(get_active_booking_details())

        resp = get_bot_response("reset")
        self.assertIn("Chat reset!", resp)
        self.assertIsNone(get_active_booking_details())
        self.assertFalse(is_demo_mode_active())


if __name__ == "__main__":
    unittest.main()
