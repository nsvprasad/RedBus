"""
chat_logic.py - End-to-End Conversational & Automation Controller for RedBus AI Assistant.

Integrates:
- Module 1: Gradio Frontend UI (app.py)
- Module 2: Local Qwen2.5 GenAI Travel Extraction Engine (modules/ai_agent.py)
- Module 3: Playwright Browser Search & Results Filtering (modules/browser_automation.py)

Preserves conversation context across multi-turn interactions, remembers previously
provided travel parameters (source, destination, date, preferences), detects missing fields,
and coordinates browser search execution when all mandatory fields are present.
"""

import re
import logging
from typing import Optional

from modules.ai_agent import (
    BookingDetails,
    extract_travel_info,
    format_booking_summary,
)
from modules.browser_automation import (
    search_redbus,
    format_bus_results,
    BusSearchResult,
)
from modules.demo_data import (
    get_demo_buses,
    format_demo_bus_results,
    DEMO_DATA_LABEL,
)

logger = logging.getLogger(__name__)

# Initial welcome message displayed when the chat starts or resets
WELCOME_MESSAGE = (
    "Hello! Welcome to RedBus AI Assistant. Where would you like to travel?"
)

# Standalone greeting regex recognizing Hi, Hello, Hey, Good morning/afternoon/evening,
# with optional "there", punctuation, and whitespace.
GREETING_REGEX = re.compile(
    r"^(?:(?:hi|hello|hey)(?:[,\s]+there)?|good\s+(?:morning|afternoon|evening)|greetings)[!.,?\s]*$",
    re.IGNORECASE,
)

# Regex recognizing user intent to switch to / view demonstration mode
DEMO_COMMAND_REGEX = re.compile(
    r"^(?:(?:show|view|try|enable|switch\s+to|use)\s+)?demo(?:\s+mode|\s+data|\s+buses)?(?:[!\.\?]*)$",
    re.IGNORECASE,
)

# Regex recognizing user intent to exit demonstration mode and return to live search
LIVE_MODE_COMMAND_REGEX = re.compile(
    r"^(?:(?:exit|leave|disable)\s+demo|live(?:\s+mode)?)[!\.\?]*$",
    re.IGNORECASE,
)

# Active conversation state (session memory)
_active_booking: Optional[BookingDetails] = None
_demo_mode_active: bool = False


def is_demo_mode_active() -> bool:
    """Return whether demonstration mode is currently enabled."""
    global _demo_mode_active
    return _demo_mode_active


def set_demo_mode(enabled: bool) -> None:
    """Enable or disable demonstration mode."""
    global _demo_mode_active
    _demo_mode_active = enabled
    logger.info("Demonstration mode set to %s.", enabled)


def get_active_booking_details() -> Optional[BookingDetails]:
    """Return the currently stored conversation booking details."""
    global _active_booking
    return _active_booking


def reset_conversation_state() -> None:
    """Reset the multi-turn conversational booking memory and demo mode state."""
    global _active_booking, _demo_mode_active
    _active_booking = None
    _demo_mode_active = False
    logger.info("Conversational booking state and demo mode reset.")


def clear_conversation() -> None:
    """Alias for reset_conversation_state to clear active booking memory."""
    reset_conversation_state()


def _merge_booking_details(new_details: BookingDetails) -> None:
    """
    Merge newly extracted travel details into the active conversation state.
    Preserves previously supplied information unless explicitly updated.
    """
    global _active_booking

    if _active_booking is None:
        _active_booking = BookingDetails(status="incomplete")

    # Merge non-null fields
    for field in (
        "source",
        "destination",
        "journey_date",
        "bus_type",
        "seating_preference",
        "max_budget",
        "preferred_time",
    ):
        new_val = getattr(new_details, field, None)
        if new_val is not None:
            setattr(_active_booking, field, new_val)

    # Re-evaluate completeness
    if _active_booking.source and _active_booking.destination and _active_booking.journey_date:
        _active_booking.status = "complete"
        _active_booking.assistant_reply = (
            f"Great! Finding buses from {_active_booking.source} to "
            f"{_active_booking.destination} on {_active_booking.journey_date}."
        )
    else:
        _active_booking.status = "incomplete"
        # Formulate targeted follow-up question for missing parameters
        missing = []
        if not _active_booking.source:
            missing.append("departure city (origin)")
        if not _active_booking.destination:
            missing.append("destination city")
        if not _active_booking.journey_date:
            missing.append("journey date")

        if len(missing) == 3:
            _active_booking.assistant_reply = (
                "Where would you like to travel? Please share your departure city, destination, and travel date."
            )
        elif len(missing) == 2:
            _active_booking.assistant_reply = (
                f"Sure! Please specify your {' and '.join(missing)}."
            )
        else:
            _active_booking.assistant_reply = (
                f"Got it! Please provide your {missing[0]} so I can search available buses."
            )


def get_bot_response(user_message: str) -> str:
    """
    Generate an intelligent, context-aware response integrating GenAI extraction and browser search.

    1. Handles standalone greetings and reset commands.
    2. Sends natural-language travel requests to the GenAI Agent (Qwen2.5 local LLM).
    3. Merges extracted parameters into active conversation memory.
    4. If required fields are missing, prompts user for remaining details.
    5. When booking details are complete (source, destination, date), triggers RedBus search
       via Playwright, filters results by preferences, and returns formatted Markdown.

    Args:
        user_message (str): Text message entered by user.

    Returns:
        str: Assistant response message formatted with booking summary and bus results.
    """
    global _active_booking

    if not user_message or not user_message.strip():
        return "Please enter your travel details or choose one of the quick suggestions below."

    msg = user_message.strip()

    # 1. Reset / Start Over Command
    if msg.lower() in ("reset", "clear", "start over", "restart", "new search"):
        reset_conversation_state()
        return "Chat reset! Where would you like to travel today?"

    # 2. Fast-path for simple standalone greetings
    if GREETING_REGEX.match(msg):
        return (
            "Hello! Welcome to RedBus AI Assistant. Where would you like to travel today? "
            "You can specify your route (e.g., *'Chennai to Bangalore tomorrow'*)."
        )

    # 2b. Exit demo mode command
    if LIVE_MODE_COMMAND_REGEX.match(msg):
        set_demo_mode(False)
        return "Switched back to Live Search mode! Where would you like to travel today?"

    # 2c. Switch to / trigger Demo Mode command
    if DEMO_COMMAND_REGEX.match(msg):
        set_demo_mode(True)
        if _active_booking and _active_booking.source and _active_booking.destination and _active_booking.journey_date:
            summary_md = format_booking_summary(_active_booking)
            demo_result = get_demo_buses(
                source=_active_booking.source,
                destination=_active_booking.destination,
                journey_date=_active_booking.journey_date,
                details=_active_booking,
            )
            demo_md = format_demo_bus_results(demo_result)
            safety_notice = (
                "\n\n*🔒 RedBus AI Assistant operates in Demonstration & Planning mode. "
                "Review options above and complete actual bookings directly on redBus.in.*"
            )
            return summary_md + demo_md + safety_notice
        else:
            return (
                "🎭 **Demonstration Mode Enabled!**\n\n"
                "Live search on redBus.in is currently restricted by automated access protections. "
                "In Demo Mode, I will provide realistic simulated bus schedules and pricing.\n\n"
                "Where would you like to travel? Please share your departure city, destination, and travel date "
                "(e.g., *'Chennai to Bangalore tomorrow'*)."
            )

    # Check if user message explicitly requests demo mode
    if re.search(r"\bdemo\b", msg, re.IGNORECASE):
        set_demo_mode(True)

    # 3. Process natural language query via GenAI Agent
    new_details, error = extract_travel_info(msg)

    # 4. Handle LLM service errors (e.g., local llama.cpp server unavailable)
    if error and not new_details.source and not new_details.destination and not new_details.journey_date:
        return format_booking_summary(new_details)

    # 5. Merge parameters into active conversation memory
    _merge_booking_details(new_details)

    # 6. If information is still incomplete, prompt user for missing fields
    if _active_booking.status != "complete":
        return format_booking_summary(_active_booking)

    # 7. Information is COMPLETE -> Execute Search (Demo or Live)
    summary_md = format_booking_summary(_active_booking)

    if _demo_mode_active:
        logger.info(
            "Demo mode active: generating demonstration buses for %s -> %s on %s.",
            _active_booking.source,
            _active_booking.destination,
            _active_booking.journey_date,
        )
        demo_result = get_demo_buses(
            source=_active_booking.source,
            destination=_active_booking.destination,
            journey_date=_active_booking.journey_date,
            details=_active_booking,
        )
        demo_md = format_demo_bus_results(demo_result)
        safety_notice = (
            "\n\n*🔒 RedBus AI Assistant operates in Demonstration & Planning mode. "
            "Review options above and complete actual bookings directly on redBus.in.*"
        )
        return summary_md + demo_md + safety_notice

    logger.info(
        "Complete booking request detected: %s -> %s on %s. Executing browser search.",
        _active_booking.source,
        _active_booking.destination,
        _active_booking.journey_date,
    )

    # Execute Playwright browser automation search
    search_result = search_redbus(
        source=_active_booking.source,
        destination=_active_booking.destination,
        journey_date=_active_booking.journey_date,
        headless=True,
        details=_active_booking,
    )

    # Format extracted bus listings
    bus_results_md = format_bus_results(search_result)

    # If live search encountered an error, offer user an option to switch to demo mode
    demo_fallback_offer = ""
    if search_result.error_message:
        demo_fallback_offer = (
            "\n\n💡 **Demonstration Mode Available:**\n"
            "Live RedBus search is currently restricted by website security protections. "
            "Would you like to view realistic simulated bus options for this route? "
            "Reply with **'show demo'** or **'demo mode'**."
        )

    # Safety disclaimer (Search-only milestone)
    safety_notice = (
        "\n\n*🔒 RedBus AI Assistant operates in Search & Planning mode. "
        "Review bus options above and complete booking directly on redBus.in.*"
    )

    return summary_md + bus_results_md + demo_fallback_offer + safety_notice
