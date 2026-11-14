"""
demo_data.py - Clearly Separated Demonstration Data Source for RedBus AI Assistant.

Provides realistic sample bus records for testing, evaluation, and fallback demonstration
when live RedBus website access is restricted (e.g., net::ERR_HTTP2_PROTOCOL_ERROR or Akamai challenge).

IMPORTANT:
- Demo data is strictly separated from live web scraping.
- Never presented as real-time availability.
- Prominently marked with "DEMO DATA – NOT LIVE REDBUS AVAILABILITY".
- Reuses the existing BusItem model and filter_bus_results() function.
"""

import logging
from typing import List, Dict, Any, Optional

from modules.browser_automation import (
    BusItem,
    BusSearchResult,
    filter_bus_results,
)

logger = logging.getLogger(__name__)

# Prominent disclaimer label required by specification
DEMO_DATA_LABEL = "DEMO DATA – NOT LIVE REDBUS AVAILABILITY"

# ============================================================================
# Realistic Sample Bus Data Source (Separated from live extraction)
# ============================================================================

RAW_DEMO_ROUTES: Dict[str, List[Dict[str, Any]]] = {
    "chennai-bangalore": [
        {
            "operator_name": "IntrCity SmartBus",
            "bus_type": "AC Sleeper (2+1)",
            "departure_time": "21:30",
            "arrival_time": "05:45",
            "duration": "08h 15m",
            "fare": 1250.0,
            "rating": "4.5",
            "seats_available": "18 Seats",
        },
        {
            "operator_name": "Kaveri Travels",
            "bus_type": "AC Sleeper",
            "departure_time": "22:00",
            "arrival_time": "06:00",
            "duration": "08h 00m",
            "fare": 950.0,
            "rating": "4.3",
            "seats_available": "12 Seats",
        },
        {
            "operator_name": "VRL Travels",
            "bus_type": "AC Semi-Sleeper",
            "departure_time": "23:00",
            "arrival_time": "07:00",
            "duration": "08h 00m",
            "fare": 850.0,
            "rating": "4.4",
            "seats_available": "24 Seats",
        },
        {
            "operator_name": "SRS Travels",
            "bus_type": "Non-AC Seater (2+2)",
            "departure_time": "06:00",
            "arrival_time": "12:30",
            "duration": "06h 30m",
            "fare": 550.0,
            "rating": "3.9",
            "seats_available": "31 Seats",
        },
        {
            "operator_name": "Greenline Travels",
            "bus_type": "AC Sleeper",
            "departure_time": "21:00",
            "arrival_time": "05:00",
            "duration": "08h 00m",
            "fare": 1450.0,
            "rating": "4.7",
            "seats_available": "7 Seats",
        },
        {
            "operator_name": "Jabbar Travels",
            "bus_type": "Non-AC Sleeper",
            "departure_time": "22:30",
            "arrival_time": "06:30",
            "duration": "08h 00m",
            "fare": 750.0,
            "rating": "4.1",
            "seats_available": "15 Seats",
        },
    ],
    "bangalore-chennai": [
        {
            "operator_name": "IntrCity SmartBus",
            "bus_type": "AC Sleeper (2+1)",
            "departure_time": "22:00",
            "arrival_time": "06:15",
            "duration": "08h 15m",
            "fare": 1250.0,
            "rating": "4.5",
            "seats_available": "14 Seats",
        },
        {
            "operator_name": "Kaveri Travels",
            "bus_type": "AC Sleeper",
            "departure_time": "21:30",
            "arrival_time": "05:30",
            "duration": "08h 00m",
            "fare": 950.0,
            "rating": "4.3",
            "seats_available": "16 Seats",
        },
        {
            "operator_name": "VRL Travels",
            "bus_type": "AC Semi-Sleeper",
            "departure_time": "23:15",
            "arrival_time": "07:15",
            "duration": "08h 00m",
            "fare": 850.0,
            "rating": "4.4",
            "seats_available": "20 Seats",
        },
        {
            "operator_name": "Asian Xpress",
            "bus_type": "AC Sleeper",
            "departure_time": "23:45",
            "arrival_time": "07:00",
            "duration": "07h 15m",
            "fare": 1100.0,
            "rating": "4.2",
            "seats_available": "9 Seats",
        },
    ],
    "vijayawada-hyderabad": [
        {
            "operator_name": "Orange Tours and Travels",
            "bus_type": "AC Sleeper",
            "departure_time": "23:00",
            "arrival_time": "05:30",
            "duration": "06h 30m",
            "fare": 1100.0,
            "rating": "4.5",
            "seats_available": "11 Seats",
        },
        {
            "operator_name": "Morning Star Travels",
            "bus_type": "AC Semi-Sleeper",
            "departure_time": "22:30",
            "arrival_time": "05:00",
            "duration": "06h 30m",
            "fare": 800.0,
            "rating": "4.3",
            "seats_available": "19 Seats",
        },
        {
            "operator_name": "APSRTC (Amaravathi)",
            "bus_type": "AC Seater",
            "departure_time": "07:00",
            "arrival_time": "13:00",
            "duration": "06h 00m",
            "fare": 650.0,
            "rating": "4.2",
            "seats_available": "28 Seats",
        },
        {
            "operator_name": "Kaveri Travels",
            "bus_type": "Non-AC Sleeper",
            "departure_time": "23:45",
            "arrival_time": "06:30",
            "duration": "06h 45m",
            "fare": 700.0,
            "rating": "4.0",
            "seats_available": "15 Seats",
        },
    ],
    "hyderabad-vijayawada": [
        {
            "operator_name": "Orange Tours and Travels",
            "bus_type": "AC Sleeper",
            "departure_time": "22:30",
            "arrival_time": "05:00",
            "duration": "06h 30m",
            "fare": 1100.0,
            "rating": "4.5",
            "seats_available": "10 Seats",
        },
        {
            "operator_name": "Morning Star Travels",
            "bus_type": "AC Semi-Sleeper",
            "departure_time": "23:00",
            "arrival_time": "05:30",
            "duration": "06h 30m",
            "fare": 800.0,
            "rating": "4.3",
            "seats_available": "18 Seats",
        },
        {
            "operator_name": "APSRTC (Garuda Plus)",
            "bus_type": "AC Sleeper",
            "departure_time": "21:00",
            "arrival_time": "03:30",
            "duration": "06h 30m",
            "fare": 950.0,
            "rating": "4.4",
            "seats_available": "14 Seats",
        },
    ],
}

GENERIC_DEFAULT_BUSES: List[Dict[str, Any]] = [
    {
        "operator_name": "Express Line Travels",
        "bus_type": "AC Sleeper (2+1)",
        "departure_time": "21:30",
        "arrival_time": "05:30",
        "duration": "08h 00m",
        "fare": 1150.0,
        "rating": "4.4",
        "seats_available": "15 Seats",
    },
    {
        "operator_name": "National Roadways",
        "bus_type": "AC Semi-Sleeper (2+2)",
        "departure_time": "22:15",
        "arrival_time": "06:15",
        "duration": "08h 00m",
        "fare": 850.0,
        "rating": "4.2",
        "seats_available": "22 Seats",
    },
    {
        "operator_name": "Royal Tourist",
        "bus_type": "Non-AC Sleeper (2+1)",
        "departure_time": "22:45",
        "arrival_time": "06:45",
        "duration": "08h 00m",
        "fare": 700.0,
        "rating": "4.0",
        "seats_available": "18 Seats",
    },
    {
        "operator_name": "City Express Link",
        "bus_type": "Non-AC Seater (2+2)",
        "departure_time": "06:30",
        "arrival_time": "13:30",
        "duration": "07h 00m",
        "fare": 500.0,
        "rating": "3.8",
        "seats_available": "30 Seats",
    },
    {
        "operator_name": "Luxury Star Coach",
        "bus_type": "AC Sleeper (2+1)",
        "departure_time": "23:00",
        "arrival_time": "06:30",
        "duration": "07h 30m",
        "fare": 1350.0,
        "rating": "4.6",
        "seats_available": "8 Seats",
    },
]


def load_raw_demo_buses(source: str, destination: str) -> List[BusItem]:
    """
    Load raw simulated bus records for a specific route.
    Validates each record through Pydantic BusItem model.
    Falls back to generic representative buses if route has no specific mock dataset.

    Args:
        source (str): Origin city name.
        destination (str): Destination city name.

    Returns:
        List[BusItem]: List of validated BusItem instances.
    """
    route_key = f"{source.strip().lower()}-{destination.strip().lower()}"
    raw_list = RAW_DEMO_ROUTES.get(route_key, GENERIC_DEFAULT_BUSES)

    validated_buses: List[BusItem] = []
    for item in raw_list:
        try:
            bus = BusItem(
                operator_name=item["operator_name"],
                bus_type=item["bus_type"],
                departure_time=item["departure_time"],
                arrival_time=item["arrival_time"],
                duration=item["duration"],
                fare=item["fare"],
                rating=item.get("rating"),
                seats_available=item.get("seats_available"),
            )
            validated_buses.append(bus)
        except Exception as e:
            logger.warning("Error instantiating demo BusItem: %s", e)

    return validated_buses


def get_demo_buses(
    source: str,
    destination: str,
    journey_date: str,
    details: Optional[Any] = None,
) -> BusSearchResult:
    """
    Generate demonstration search results with optional preference filtering.
    Reuses filter_bus_results() from browser_automation module.

    Args:
        source (str): Departure origin city.
        destination (str): Arrival destination city.
        journey_date (str): Travel date (YYYY-MM-DD).
        details: Optional booking parameters (bus_type, seating, budget, preferred_time).

    Returns:
        BusSearchResult: Populated result container with demonstration buses.
    """
    raw_buses = load_raw_demo_buses(source, destination)
    filtered_buses = filter_bus_results(raw_buses, details=details)

    return BusSearchResult(
        source=source.strip(),
        destination=destination.strip(),
        journey_date=journey_date.strip(),
        total_found=len(filtered_buses),
        buses=filtered_buses,
    )


def format_demo_bus_results(result: BusSearchResult) -> str:
    """
    Format demo bus search results into Markdown with prominent disclaimer labels.
    Ensures demo data is never misrepresented as real-time availability.

    Args:
        result (BusSearchResult): Search result container with demo buses.

    Returns:
        str: Formatted Markdown table with mandatory demonstration disclaimers.
    """
    header_lines = [
        "\n\n---",
        f"### ⚠️ {DEMO_DATA_LABEL}",
        "> **Notice:** Live search on redBus.in is currently unavailable due to automated website access restrictions. "
        "Showing simulated demonstration schedules and pricing for evaluation.\n",
    ]

    if not result.buses:
        empty_lines = [
            f"🔍 **No demonstration buses match your specific filters** for **{result.source} ➔ {result.destination}** on **{result.journey_date}**.",
            "*Try relaxing your budget, bus type, or timing preferences.*",
        ]
        return "\n".join(header_lines + empty_lines)

    table_lines = [
        f"🚌 **Demonstration Buses ({result.source} ➔ {result.destination} • {result.journey_date}):**\n",
        "| Operator | Bus Type | Departs | Arrives | Duration | Fare | Rating | Seats |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for b in result.buses:
        rating_str = f"⭐ {b.rating}" if b.rating else "—"
        seats_str = b.seats_available if b.seats_available else "Available"
        fare_str = f"₹{int(b.fare):,}" if b.fare.is_integer() else f"₹{b.fare:,.2f}"

        table_lines.append(
            f"| **{b.operator_name}** | {b.bus_type} | {b.departure_time} | {b.arrival_time} | {b.duration} | **{fare_str}** | {rating_str} | {seats_str} |"
        )

    footer_lines = [
        f"\n*Showing {len(result.buses)} demonstration bus options matching your travel criteria.*",
        "*⚠️ These listings are simulated for testing and demonstration purposes only and do not reflect real-time live availability.*",
    ]

    return "\n".join(header_lines + table_lines + footer_lines)
