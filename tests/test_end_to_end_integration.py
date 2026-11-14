"""
test_end_to_end_integration.py - Integration tests for Module 3.4 (Local GenAI + Playwright + Gradio).
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is discoverable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.chat_logic import (
    get_bot_response,
    get_active_booking_details,
    reset_conversation_state,
    clear_conversation,
    _merge_booking_details,
)
from modules.ai_agent import BookingDetails
from modules.browser_automation import BusItem, BusSearchResult


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-End integration test suite for chat_logic, GenAI agent, and browser search."""

    def setUp(self):
        # Reset conversation state before each test
        reset_conversation_state()

    def tearDown(self):
        reset_conversation_state()

    def test_01_greeting_fast_path(self):
        """Test that greetings return welcome message without calling Qwen LLM extraction or browser search."""
        test_greetings = [
            "Hi",
            "Hi there!",
            "Hello",
            "Hello there!",
            "Hey!",
            "Good morning",
            "Good afternoon",
            "Good evening",
        ]

        with patch("modules.chat_logic.extract_travel_info") as mock_extract:
            for greeting in test_greetings:
                reset_conversation_state()
                response = get_bot_response(greeting)
                self.assertIn("Welcome to RedBus AI Assistant", response)
                self.assertIsNone(get_active_booking_details())
            mock_extract.assert_not_called()

    def test_02_missing_information_multi_turn(self):
        """Test multi-turn conversational context merging across sequential messages."""
        # Turn 1: User mentions destination only
        resp_turn1 = get_bot_response("I want to travel to Bangalore")
        self.assertIn("Extracted Booking Details", resp_turn1)
        self.assertIn("Bangalore", resp_turn1)

        active = get_active_booking_details()
        self.assertIsNotNone(active)
        self.assertEqual(active.destination, "Bangalore")
        self.assertIsNone(active.source)
        self.assertEqual(active.status, "incomplete")

        # Turn 2: User provides origin and date
        mock_buses = [
            BusItem(
                operator_name="Kaveri Travels",
                bus_type="AC Sleeper",
                departure_time="22:00",
                arrival_time="06:00",
                duration="08h 00m",
                fare=950.0,
                rating="4.3",
                seats_available="12 Seats"
            )
        ]
        mock_result = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            total_found=1,
            buses=mock_buses
        )

        with patch("modules.chat_logic.search_redbus", return_value=mock_result) as mock_search:
            resp_turn2 = get_bot_response("From Chennai on 2026-11-20")

            # Check that search was called with merged parameters
            mock_search.assert_called_once()
            called_kwargs = mock_search.call_args.kwargs
            self.assertEqual(called_kwargs["source"], "Chennai")
            self.assertEqual(called_kwargs["destination"], "Bangalore")
            self.assertEqual(called_kwargs["journey_date"], "2026-11-20")

            # Verify response contains summary and bus table
            self.assertIn("Extracted Booking Details", resp_turn2)
            self.assertIn("Available Buses", resp_turn2)
            self.assertIn("Kaveri Travels", resp_turn2)
            self.assertIn("₹950", resp_turn2)
            self.assertIn("Search & Planning mode", resp_turn2)

    def test_03_preference_retention_across_turns(self):
        """Test updating preferences while retaining existing route and journey date."""
        # Initialize active booking state
        mock_buses = [
            BusItem(
                operator_name="VRL Travels",
                bus_type="AC Sleeper",
                departure_time="21:30",
                arrival_time="05:30",
                duration="08h 00m",
                fare=1100.0,
                rating="4.5"
            )
        ]
        mock_result = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            total_found=1,
            buses=mock_buses
        )

        with patch("modules.chat_logic.search_redbus", return_value=mock_result):
            # Turn 1: Complete route
            get_bot_response("Chennai to Bangalore on 2026-11-20")

            active = get_active_booking_details()
            self.assertEqual(active.source, "Chennai")
            self.assertEqual(active.destination, "Bangalore")
            self.assertEqual(active.journey_date, "2026-11-20")

            # Turn 2: User specifies AC Sleeper preference
            resp_turn2 = get_bot_response("Only AC Sleeper under 1500")

            active_updated = get_active_booking_details()
            self.assertEqual(active_updated.source, "Chennai")
            self.assertEqual(active_updated.destination, "Bangalore")
            self.assertEqual(active_updated.journey_date, "2026-11-20")
            self.assertEqual(active_updated.bus_type, "AC")
            self.assertEqual(active_updated.seating_preference, "Sleeper")
            self.assertEqual(active_updated.max_budget, 1500.0)

            self.assertIn("AC Sleeper", resp_turn2)
            self.assertIn("VRL Travels", resp_turn2)

    def test_04_reset_command(self):
        """Test 'reset' and 'clear' commands clear active conversational memory."""
        get_bot_response("I want to travel to Hyderabad")
        self.assertIsNotNone(get_active_booking_details())

        resp = get_bot_response("reset")
        self.assertIn("Chat reset!", resp)
        self.assertIsNone(get_active_booking_details())

    def test_05_browser_access_restriction_handled(self):
        """Test that browser anti-bot or access restrictions are presented cleanly to the user."""
        mock_blocked = BusSearchResult(
            source="Chennai",
            destination="Bangalore",
            journey_date="2026-11-20",
            error_message="RedBus access restricted: Cloudflare challenge detected."
        )

        with patch("modules.chat_logic.search_redbus", return_value=mock_blocked):
            resp = get_bot_response("Chennai to Bangalore on 2026-11-20")
            self.assertIn("Bus Search Notice", resp)
            self.assertIn("Cloudflare challenge detected", resp)

    def test_06_local_llm_unavailable_handled(self):
        """Test graceful response when local llama.cpp server is offline."""
        mock_unavailable = BookingDetails(
            status="incomplete",
            assistant_reply="Local AI server is not running. Please start llama.cpp."
        )

        with patch("modules.chat_logic.extract_travel_info", return_value=(mock_unavailable, "Connection refused")):
            resp = get_bot_response("Chennai to Bangalore tomorrow")
            self.assertIn("Local AI server is not running", resp)

    def test_07_clear_conversation_and_reset(self):
        """Test clearing conversation resets all booking details and none are retained."""
        from app import clear_conversation as app_clear_conversation

        # Populate booking state
        _merge_booking_details(
            BookingDetails(
                source="Chennai",
                destination="Bangalore",
                journey_date="2026-11-20",
                bus_type="AC",
                seating_preference="Sleeper",
                max_budget=1500.0,
                status="complete",
            )
        )
        self.assertIsNotNone(get_active_booking_details())
        self.assertEqual(get_active_booking_details().source, "Chennai")

        # Test modules.chat_logic clear_conversation
        clear_conversation()
        self.assertIsNone(get_active_booking_details())

        # Repopulate and test app.py clear_conversation
        _merge_booking_details(
            BookingDetails(
                source="Hyderabad",
                destination="Vijayawada",
                journey_date="2026-11-22",
                status="complete",
            )
        )
        self.assertIsNotNone(get_active_booking_details())
        cleared_text, history = app_clear_conversation()
        self.assertEqual(cleared_text, "")
        self.assertEqual(len(history), 1)
        self.assertIsNone(get_active_booking_details())


if __name__ == "__main__":
    unittest.main()
