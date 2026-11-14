"""
ai_agent.py - GenAI Travel Intelligence Module for RedBus AI Assistant (Module 2).

This module connects to Hugging Face language models using Hugging Face InferenceClient
to understand natural language travel requests, extract structured booking parameters,
calculate dynamic dates in the Asia/Kolkata timezone, and validate output using Pydantic.
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Literal, Tuple, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

# Load environment variables from .env if present
load_dotenv()

# Configure module logger
logger = logging.getLogger(__name__)

# Default LLM configuration
DEFAULT_LLM_BACKEND = "local"
DEFAULT_LOCAL_LLM_URL = "http://127.0.0.1:9931/v1/chat/completions"
DEFAULT_LOCAL_MODEL_NAME = "Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_LOCAL_TIMEOUT = 60
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def get_kolkata_now() -> datetime:
    """
    Get current datetime in the Asia/Kolkata timezone.
    
    Returns:
        datetime: Current datetime localized to Asia/Kolkata (IST).
    """
    try:
        return datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        # Fallback to local time if zoneinfo database lookup fails
        return datetime.now()


def build_date_context() -> str:
    """
    Generate dynamic relative date context based on current Asia/Kolkata time.
    Provides reference dates for today, tomorrow, and upcoming weekdays so the
    LLM accurately translates natural-language relative dates into YYYY-MM-DD.
    
    Returns:
        str: Date reference guide formatted for prompt injection.
    """
    now = get_kolkata_now()
    today_str = now.strftime("%Y-%m-%d")
    today_name = now.strftime("%A")
    tomorrow = now + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    tomorrow_name = tomorrow.strftime("%A")
    day_after = now + timedelta(days=2)
    day_after_str = day_after.strftime("%Y-%m-%d")

    # Calculate upcoming weekdays
    weekdays = {}
    for i in range(1, 8):
        future_day = now + timedelta(days=i)
        day_label = future_day.strftime("%A")
        if day_label not in weekdays:
            weekdays[day_label] = future_day.strftime("%Y-%m-%d")

    context = (
        f"Current Timezone: Asia/Kolkata (IST)\n"
        f"Today's Date: {today_str} ({today_name})\n"
        f"Tomorrow: {tomorrow_str} ({tomorrow_name})\n"
        f"Day after tomorrow: {day_after_str}\n"
        f"Upcoming days reference:\n"
    )
    for day_label, d_str in weekdays.items():
        context += f" - Next/This {day_label}: {d_str}\n"
    return context


# ============================================================================
# Pydantic Output Validation Schema
# ============================================================================

class BookingDetails(BaseModel):
    """
    Pydantic schema representing structured bus booking parameters.
    Validates field types, date format, and ensures missing fields are null.
    """
    source: Optional[str] = Field(
        default=None,
        description="Departure or origin city/town name, or null if not provided."
    )
    destination: Optional[str] = Field(
        default=None,
        description="Arrival or destination city/town name, or null if not provided."
    )
    journey_date: Optional[str] = Field(
        default=None,
        description="Journey date strictly in YYYY-MM-DD format, or null if not specified."
    )
    bus_type: Optional[str] = Field(
        default=None,
        description="Bus type preference: 'AC', 'Non-AC', or null if unspecified."
    )
    seating_preference: Optional[str] = Field(
        default=None,
        description="Seating preference: 'Sleeper', 'Seater', or null if unspecified."
    )
    max_budget: Optional[float] = Field(
        default=None,
        description="Maximum budget amount as a numeric value (e.g. 1000.0), or null."
    )
    preferred_time: Optional[str] = Field(
        default=None,
        description="Preferred departure time window (e.g., 'morning', 'night', '18:00'), or null."
    )
    status: Literal["complete", "incomplete", "clarification_needed"] = Field(
        default="incomplete",
        description="Status: 'complete' if source, destination, and journey_date are present; 'incomplete' if any are missing; 'clarification_needed' if input is ambiguous."
    )
    assistant_reply: Optional[str] = Field(
        default=None,
        description="Natural language conversational response to show the user."
    )

    @field_validator("journey_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Ensures journey_date conforms to YYYY-MM-DD or is None."""
        if v is None:
            return None
        v = v.strip()
        if not v or v.lower() in ("null", "none"):
            return None
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
            # Attempt to resolve common non-standard formats if possible
            try:
                dt = datetime.strptime(v, "%Y/%m/%d")
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
            return None
        return v

    @field_validator("source", "destination", mode="before")
    @classmethod
    def clean_city_name(cls, v: Any) -> Optional[str]:
        """Cleans city names and converts null-like strings to None."""
        if v is None:
            return None
        s = str(v).strip()
        if not s or s.lower() in ("null", "none", "unknown", "unspecified", "any"):
            return None
        return s.title()

    @field_validator("max_budget", mode="before")
    @classmethod
    def clean_budget(cls, v: Any) -> Optional[float]:
        """Extracts numeric budget or returns None."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s or s.lower() in ("null", "none"):
            return None
        # Extract digits
        match = re.search(r"(\d+(?:\.\d+)?)", s)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    @field_validator("bus_type", mode="before")
    @classmethod
    def normalize_bus_type(cls, v: Any) -> Optional[str]:
        """Normalizes AC / Non-AC values."""
        if v is None:
            return None
        s = str(v).strip().lower()
        if "non-ac" in s or "non ac" in s:
            return "Non-AC"
        if "ac" in s:
            return "AC"
        return None

    @field_validator("seating_preference", mode="before")
    @classmethod
    def normalize_seating(cls, v: Any) -> Optional[str]:
        """Normalizes Sleeper / Seater values."""
        if v is None:
            return None
        s = str(v).strip().lower()
        if "sleeper" in s:
            return "Sleeper"
        if "seater" in s or "sitting" in s:
            return "Seater"
        return None

    def enforce_status(self) -> None:
        """
        Enforce strict status rules:
        If source, destination, or journey_date is missing, status CANNOT be 'complete'.
        """
        if not self.source or not self.destination or not self.journey_date:
            if self.status == "complete":
                self.status = "incomplete"
        else:
            self.status = "complete"


# ============================================================================
# Hugging Face AI Extraction Logic
# ============================================================================

def create_system_prompt() -> str:
    """
    Build the instruction prompt for the Hugging Face model,
    enforcing dynamic IST dates and valid JSON output.
    """
    date_context = build_date_context()
    
    prompt = f"""You are the intelligent travel booking engine for RedBus AI Assistant.
Your task is to understand the user's travel request, extract structured booking parameters, identify any missing information, and formulate a helpful conversational reply.

### Dynamic Date Reference (Asia/Kolkata IST):
{date_context}

### Extraction Rules:
1. 'source': Origin/departure city name (e.g., "Chennai", "Vijayawada"). If not mentioned, set to null. DO NOT guess or invent.
2. 'destination': Destination city name (e.g., "Bangalore", "Hyderabad"). If not mentioned, set to null. DO NOT guess or invent.
3. 'journey_date': Calculate exact YYYY-MM-DD date based on the user's input and the date reference above. If no date is mentioned, set to null.
4. 'bus_type': "AC" or "Non-AC", or null if not specified.
5. 'seating_preference': "Sleeper" or "Seater", or null if not specified.
6. 'max_budget': Maximum budget as a number (e.g. 1000) or null.
7. 'preferred_time': Departure time preference (e.g. "morning", "night", "afternoon") or null.
8. 'status':
   - "complete": ONLY if 'source', 'destination', AND 'journey_date' are ALL provided.
   - "incomplete": If any of 'source', 'destination', or 'journey_date' is null.
   - "clarification_needed": If the query is vague, greeting only, or ambiguous.
9. 'assistant_reply':
   - If incomplete: Politely ask for the specific missing information (e.g. "Sure! Which city are you travelling from and where would you like to go?").
   - If complete: Confirm the route, date, and any preferences enthusiastically.
   - Never invent information not provided by the user.

### Strict Output Format:
Return ONLY a valid JSON object matching this schema. No markdown code blocks, no other text:
{{
    "source": string or null,
    "destination": string or null,
    "journey_date": "YYYY-MM-DD" or null,
    "bus_type": "AC" or "Non-AC" or null,
    "seating_preference": "Sleeper" or "Seater" or null,
    "max_budget": number or null,
    "preferred_time": string or null,
    "status": "complete" | "incomplete" | "clarification_needed",
    "assistant_reply": "Conversational message to user"
}}
"""
    return prompt


def extract_travel_info(user_message: str) -> Tuple[BookingDetails, Optional[str]]:
    """
    Analyze user travel message using configured language model (Local llama.cpp or Hugging Face)
    and extract validated booking information.

    Args:
        user_message (str): Natural language message from user.

    Returns:
        Tuple[BookingDetails, Optional[str]]: (Extracted validated details, error message if any)
    """
    if not user_message or not user_message.strip():
        details = BookingDetails(
            status="clarification_needed",
            assistant_reply="Hello! Welcome to RedBus AI Assistant. Where would you like to travel?"
        )
        return details, None

    backend = os.getenv("LLM_BACKEND", DEFAULT_LLM_BACKEND).strip().lower()
    system_prompt = create_system_prompt()
    raw_response = None

    if backend == "local":
        local_url = os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_LLM_URL).strip()
        model_name = os.getenv("LOCAL_MODEL_NAME", DEFAULT_LOCAL_MODEL_NAME).strip()
        try:
            local_timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", str(DEFAULT_LOCAL_TIMEOUT)))
        except ValueError:
            local_timeout = DEFAULT_LOCAL_TIMEOUT

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message.strip()}
            ],
            "temperature": 0.1,
            "max_tokens": 600,
            "response_format": {"type": "json_object"}
        }

        try:
            req_body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                local_url,
                data=req_body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=local_timeout) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                raw_response = res_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as http_err:
            # Fallback without response_format if local provider rejected json_object mode
            try:
                payload.pop("response_format", None)
                req_body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    local_url,
                    data=req_body,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=local_timeout) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    raw_response = res_data["choices"][0]["message"]["content"]
            except Exception as fb_err:
                err_str = f"Local AI server error: {fb_err}"
                logger.error(err_str)
                details = BookingDetails(
                    status="incomplete",
                    assistant_reply="Local AI server is not running. Please start llama.cpp."
                )
                return details, err_str
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            err_str = f"Local AI server unavailable: {e}"
            logger.error(err_str)
            details = BookingDetails(
                status="incomplete",
                assistant_reply="Local AI server is not running. Please start llama.cpp."
            )
            return details, err_str
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            err_str = f"Unexpected response from local AI server: {e}"
            logger.error(err_str)
            details = BookingDetails(
                status="incomplete",
                assistant_reply="Local AI server is not running. Please start llama.cpp."
            )
            return details, err_str

    else:
        # Hugging Face Inference Backend
        token = os.getenv("HF_TOKEN")
        model_id = os.getenv("HF_MODEL", DEFAULT_HF_MODEL)

        # Check for missing or placeholder token
        if not token or not token.strip() or token.strip().lower() in ("your_huggingface_token", "your_huggingface_token_here", "your_token_here"):
            err_msg = (
                "Hugging Face API token is not configured. "
                "Please configure your HF_TOKEN in the .env file or system environment to enable AI travel extraction."
            )
            logger.warning(err_msg)
            details = BookingDetails(
                status="incomplete",
                assistant_reply=(
                    "⚠️ **Hugging Face API Token Required**\n\n"
                    "To enable GenAI intelligence, please set your Hugging Face API token (`HF_TOKEN`) "
                    "in a `.env` file or environment variable.\n\n"
                    "*Example:* `HF_TOKEN=hf_your_actual_token_here`"
                )
            )
            return details, err_msg

        if InferenceClient is None:
            err_msg = "huggingface-hub is not installed. Please install huggingface-hub or set LLM_BACKEND=local."
            logger.error(err_msg)
            details = BookingDetails(status="incomplete", assistant_reply=err_msg)
            return details, err_msg

        client = InferenceClient(token=token.strip())

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message.strip()}
            ]

            try:
                chat_completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                raw_response = chat_completion.choices[0].message.content
            except Exception:
                chat_completion = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.1
                )
                raw_response = chat_completion.choices[0].message.content

        except Exception as e:
            err_str = f"Hugging Face API Error: {str(e)}"
            logger.error(err_str)
            details = BookingDetails(
                status="incomplete",
                assistant_reply=(
                    f"⚠️ **Hugging Face Service Unavailable**\n\n"
                    f"Could not reach model `{model_id}`. Please check your network connection and `HF_TOKEN`.\n\n"
                    f"*Details:* `{str(e)}`"
                )
            )
            return details, err_str

    # 3. Parse JSON from model response
    parsed_json: Dict[str, Any] = {}
    try:
        if raw_response:
            # Extract JSON block using regex if model included extra text or markdown fences
            json_match = re.search(r"\{[\s\S]*\}", raw_response)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
            else:
                parsed_json = json.loads(raw_response)
    except Exception as parse_err:
        err_str = f"Model JSON Parsing Failed: {str(parse_err)}"
        logger.error(f"{err_str}. Raw content: {raw_response}")
        details = BookingDetails(
            status="clarification_needed",
            assistant_reply="I couldn't clearly parse your travel details. Could you please specify your departure city, destination, and travel date?"
        )
        return details, err_str

    # 4. Validate with Pydantic
    try:
        details = BookingDetails.model_validate(parsed_json)
        details.enforce_status()
        
        # If assistant_reply is missing, construct an appropriate one
        if not details.assistant_reply:
            if details.status == "complete":
                details.assistant_reply = (
                    f"Great! I found buses from {details.source} to {details.destination} on {details.journey_date}."
                )
            elif not details.source or not details.destination:
                details.assistant_reply = "Sure! Which city are you travelling from and where would you like to go?"
            elif not details.journey_date:
                details.assistant_reply = (
                    f"Great! I can help you find buses from {details.source} to {details.destination}. Please provide your journey date."
                )
            else:
                details.assistant_reply = "Please share your travel route and date so I can find the best buses for you."

        return details, None

    except ValidationError as val_err:
        err_str = f"Pydantic Validation Error: {str(val_err)}"
        logger.error(err_str)
        details = BookingDetails(
            status="clarification_needed",
            assistant_reply="I received your request, but some details could not be validated. Please specify your travel cities and journey date."
        )
        return details, err_str


# ============================================================================
# Markdown Summary Presentation Helper
# ============================================================================

def format_booking_summary(details: BookingDetails) -> str:
    """
    Format extracted booking details into a professional, readable card
    for display in the Gradio chat window.
    
    Args:
        details (BookingDetails): Validated booking parameters.
        
    Returns:
        str: GitHub-flavored markdown representation.
    """
    reply = details.assistant_reply or ""
    
    # Check if any travel field was extracted
    has_any_info = any([
        details.source,
        details.destination,
        details.journey_date,
        details.bus_type,
        details.seating_preference,
        details.max_budget is not None
    ])

    if not has_any_info and details.status != "complete":
        # Just return assistant conversational reply without an empty card
        return reply

    status_badge = "Complete ✅" if details.status == "complete" else "Incomplete ⏳"
    
    # Format budget display
    if details.max_budget is not None:
        budget_str = f"₹{int(details.max_budget):,}" if details.max_budget.is_integer() else f"₹{details.max_budget:,.2f}"
    else:
        budget_str = "No budget limit specified"

    # Construct clean summary card
    summary_card = (
        f"\n\n---\n"
        f"📋 **Extracted Booking Details:**\n"
        f"- 🏙️ **From (Source):** {details.source or '*(Not specified)* ❓'}\n"
        f"- 🎯 **To (Destination):** {details.destination or '*(Not specified)* ❓'}\n"
        f"- 📅 **Journey Date:** {details.journey_date or '*(Not specified)* ❓'}\n"
        f"- 🚌 **Bus Type:** {details.bus_type or 'Any'}\n"
        f"- 💺 **Seating:** {details.seating_preference or 'Any'}\n"
        f"- 💰 **Max Budget:** {budget_str}\n"
        f"- ⏰ **Preferred Time:** {details.preferred_time or 'Any time'}\n"
        f"- 📌 **Status:** {status_badge}"
    )

    return reply + summary_card
