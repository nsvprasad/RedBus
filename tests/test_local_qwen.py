"""
test_local_qwen.py - Test suite for local Qwen2.5 llama.cpp integration in RedBus AI Assistant.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.ai_agent import (
    extract_travel_info,
    format_booking_summary,
    BookingDetails,
    build_date_context,
    get_kolkata_now,
)
from modules.chat_logic import get_bot_response


class TestLocalQwenIntegration(unittest.TestCase):
    """Test suite for local Qwen model inference and extraction logic."""

    @classmethod
    def setUpClass(cls):
        # Force local LLM backend for tests
        os.environ["LLM_BACKEND"] = "local"
        os.environ["LOCAL_LLM_URL"] = "http://127.0.0.1:9931/v1/chat/completions"
        os.environ["LOCAL_MODEL_NAME"] = "Qwen2.5-1.5B-Instruct-GGUF"

    def test_01_full_parameter_extraction(self):
        """Test extraction of all fields: source, destination, date, bus_type, seating, budget, preferred_time."""
        query = (
            "Book an AC Sleeper bus from Chennai to Bangalore for tomorrow night with a budget of 1500"
        )
        details, err = extract_travel_info(query)
        self.assertIsNone(err, f"Extraction failed with error: {err}")
        self.assertIsNotNone(details)

        # Source & Destination
        self.assertEqual(details.source, "Chennai")
        self.assertEqual(details.destination, "Bangalore")

        # Journey Date (must be YYYY-MM-DD format)
        self.assertIsNotNone(details.journey_date)
        self.assertRegex(details.journey_date, r"^\d{4}-\d{2}-\d{2}$")

        # Bus Type & Seating
        self.assertEqual(details.bus_type, "AC")
        self.assertEqual(details.seating_preference, "Sleeper")

        # Budget & Time
        self.assertEqual(details.max_budget, 1500.0)
        self.assertIsNotNone(details.preferred_time)

        # Status
        self.assertEqual(details.status, "complete")
        print(f"\n[PASS] Full Extraction: {details.source} -> {details.destination} on {details.journey_date} ({details.bus_type}, {details.seating_preference}, budget: {details.max_budget}, time: {details.preferred_time})")

    def test_02_partial_missing_info_extraction(self):
        """Test handling when source or date is missing."""
        query = "I want to travel to Hyderabad"
        details, err = extract_travel_info(query)
        self.assertIsNone(err)
        self.assertIsNotNone(details)

        # Destination identified, but source/date missing
        self.assertEqual(details.destination, "Hyderabad")
        self.assertIn(details.status, ["incomplete", "clarification_needed"])
        self.assertTrue(bool(details.assistant_reply))
        print(f"\n[PASS] Partial Extraction Status: {details.status}, Assistant: {details.assistant_reply}")

    def test_03_greeting_and_chat_logic(self):
        """Test chat_logic fast path for greeting and end-to-end get_bot_response."""
        greeting_resp = get_bot_response("Hello")
        self.assertIn("Welcome to RedBus AI Assistant", greeting_resp)

        travel_resp = get_bot_response("Vijayawada to Hyderabad tomorrow morning")
        self.assertIn("Vijayawada", travel_resp)
        self.assertIn("Hyderabad", travel_resp)
        self.assertIn("Extracted Booking Details", travel_resp)
        print(f"\n[PASS] Chat Logic Response:\n{travel_resp[:200]}...")

    def test_04_server_unavailable_error_handling(self):
        """Test graceful error message when local llama.cpp server is unreachable."""
        with patch.dict(os.environ, {"LOCAL_LLM_URL": "http://127.0.0.1:9998/v1/chat/completions"}):
            details, err = extract_travel_info("Chennai to Bangalore tomorrow")
            self.assertIsNotNone(err)
            self.assertEqual(details.status, "incomplete")
            self.assertEqual(details.assistant_reply, "Local AI server is not running. Please start llama.cpp.")
            
            # Verify format_booking_summary outputs the exact error message cleanly
            summary = format_booking_summary(details)
            self.assertEqual(summary, "Local AI server is not running. Please start llama.cpp.")
            print(f"\n[PASS] Server Unavailable Handled: '{summary}'")

    def test_05_invalid_json_handling(self):
        """Test graceful recovery when model produces corrupted or unparsable output."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            class MockResponse:
                def read(self):
                    return b'{"choices": [{"message": {"content": "Sorry, cannot parse this input into json"}}]}'
                def getcode(self):
                    return 200
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass

            mock_urlopen.return_value = MockResponse()
            details, err = extract_travel_info("random ambiguous gibberish")
            self.assertIsNotNone(err)
            self.assertEqual(details.status, "clarification_needed")
            self.assertIn("couldn't clearly parse", details.assistant_reply.lower())
            print(f"\n[PASS] Invalid JSON Handled: {details.assistant_reply}")


if __name__ == "__main__":
    unittest.main()
