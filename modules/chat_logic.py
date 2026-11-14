"""
chat_logic.py - Conversational logic for RedBus AI Assistant.
Integrates with Module 2 Hugging Face AI Agent (modules/ai_agent.py).
"""

import re
from modules.ai_agent import extract_travel_info, format_booking_summary

# Initial welcome message displayed when the chat starts or resets
WELCOME_MESSAGE = (
    "Hello! Welcome to RedBus AI Assistant. Where would you like to travel?"
)


def get_bot_response(user_message: str) -> str:
    """
    Generate an intelligent response using Hugging Face GenAI.
    
    1. Handles quick greetings cleanly.
    2. Sends natural-language travel requests to the Hugging Face AI Agent.
    3. Formats extracted parameters (route, date, bus type, budget, status).
    4. Displays informative guidance if the Hugging Face API is unconfigured/unavailable.
    
    Args:
        user_message (str): Text message entered by the user.
        
    Returns:
        str: Assistant response message formatted with booking summary card.
    """
    if not user_message or not user_message.strip():
        return "Please enter your travel details or choose one of the quick suggestions below."

    msg = user_message.strip()

    # Fast-path for simple standalone greetings
    if re.fullmatch(r"(?i)(hi|hello|hey|greetings|good (morning|afternoon|evening))[!. ]*", msg):
        return (
            "Hello! Welcome to RedBus AI Assistant. Where would you like to travel today? "
            "You can specify your route (e.g., *'Chennai to Bangalore tomorrow'*)."
        )

    # Process travel query via Hugging Face GenAI Agent
    details, error = extract_travel_info(msg)
    
    # Return formatted response including natural language reply and extracted details card
    return format_booking_summary(details)
