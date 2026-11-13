"""
chat_logic.py - Rule-based response handler for RedBus AI Assistant (Module 1).

This module provides simple, beginner-friendly rule-based responses to handle
user queries about bus travel, routes, budgets, and bus types.
These responses will be replaced with Hugging Face AI in Module 2.
"""

import re

# Initial welcome message displayed when the chat starts or resets
WELCOME_MESSAGE = (
    "Hello! Welcome to RedBus AI Assistant. Where would you like to travel?"
)

def get_bot_response(user_message: str) -> str:
    """
    Generate a rule-based response for a given user message.
    
    Args:
        user_message (str): The text message entered by the user.
        
    Returns:
        str: Assistant response message.
    """
    if not user_message or not user_message.strip():
        return "Please enter a message or choose one of the quick suggestions below."

    msg = user_message.strip().lower()

    # 1. Greetings
    if re.search(r"\b(hi|hello|hey|greetings|good (morning|afternoon|evening))\b", msg):
        return "Hello! Welcome to RedBus AI Assistant. Where would you like to travel?"

    # 2. Specific Quick Suggestions & Common Routes
    if "chennai to bangalore" in msg:
        return "Great! I can help you find buses from Chennai to Bangalore. Please provide your journey date."
    
    if "vijayawada to hyderabad" in msg:
        return "Great! I can help you find buses from Vijayawada to Hyderabad. Please provide your journey date."

    # Generic Route Matching: "from X to Y" or "X to Y"
    route_match = re.search(r"(?:from\s+)?([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)", msg)
    if route_match:
        origin = route_match.group(1).strip().title()
        destination = route_match.group(2).strip().title()
        # Ensure it's not matching arbitrary phrases
        if len(origin) >= 3 and len(destination) >= 3:
            return f"Great! I can help you find buses from {origin} to {destination}. Please provide your journey date."

    # 3. Budget / Fare Queries (e.g. "under 1000", "under 1000 rupees", "₹1000", "budget")
    if re.search(r"\b(under|below|less than|within)\s*(?:rs\.?|inr|₹)?\s*\d+", msg) or "budget" in msg or "under 1000" in msg or "₹1000" in msg:
        return "Sure! I'll help you find buses within your budget."

    # 4. Bus Type Queries (e.g. "AC Sleeper Buses", "volvo", "seater")
    if "sleeper" in msg or "ac" in msg or "volvo" in msg or "seater" in msg:
        return (
            "We have a wide selection of AC Sleeper and luxury buses with "
            "comfortable berths, charging points, and live tracking. "
            "Where would you like to travel?"
        )

    # 5. Journey Dates / Days
    if re.search(r"\b(today|tomorrow|day after tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", msg):
        return (
            "Got your travel date! I am checking for top-rated buses available on that day. "
            "Do you prefer morning, afternoon, or night departures?"
        )

    # 6. Gratitude / Closure
    if re.search(r"\b(thanks|thank you|bye|goodbye|cya)\b", msg):
        return "You're welcome! Have a pleasant and safe journey with RedBus. Feel free to ask anytime!"

    # 7. Fallback Response
    return (
        "I'm here to assist your bus journey! You can ask me for routes like "
        "'Chennai to Bangalore', 'Vijayawada to Hyderabad', or search by budget like "
        "'Find buses under ₹1000'."
    )
