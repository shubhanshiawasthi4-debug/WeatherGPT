"""
WeatherGPT backend — SIH26068
FastAPI service that:
  1. Takes a natural-language weather question (any supported language)
  2. Extracts location / date / weather-parameter (rule-based, with an
     optional LLM upgrade if you plug in an API key)
  3. Fetches live data from Open-Meteo (free, no API key needed)
  4. Checks the forecast against alert thresholds
  5. Replies in the user's language

Run:
    pip install -r requirements.txt --break-system-packages
    uvicorn main:app --reload --port 8000
"""

import os
import re
import json
from datetime import date, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="WeatherGPT API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Optional: set OPENAI_API_KEY (or swap in Gemini/Llama) to replace the
# rule-based parser below with a real LLM for messier / multilingual input.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY")


# --------------------------------------------------------------------------
# 1. Language templates  (add more Indian languages by adding a dict entry)
# --------------------------------------------------------------------------
TEMPLATES = {
    "en": {
        "forecast": "In {place} on {when}: {desc}, temperature around {temp}°C, "
                    "{rain}% chance of rain, wind {wind} km/h.",
        "alert": "⚠️ Weather alert for {place}: {alert_text}",
        "no_alert": "No extreme weather alerts for {place} right now.",
        "not_found": "I couldn't find the location '{place}'. Try a nearby city name.",
        "greeting": "Ask me about weather, forecasts, or alerts for any place — "
                    "e.g. 'Will it rain in Lucknow tomorrow?'",
    },
    "hi": {
        "forecast": "{place} mein {when} ka mausam: {desc}, taapmaan lagbhag {temp}°C, "
                    "baarish ki sambhaavna {rain}%, hawa ki raftaar {wind} km/h.",
        "alert": "⚠️ {place} ke liye mausam chetavani: {alert_text}",
        "no_alert": "Abhi {place} ke liye koi khaas mausam chetavani nahi hai.",
        "not_found": "'{place}' jagah nahi mili. Kisi nazdeeki shahar ka naam try karo.",
        "greeting": "Mujhse kisi bhi jagah ka mausam, forecast ya alert poochho — "
                    "jaise 'kal Lucknow mein baarish hogi kya?'",
    },
}

WEATHER_DESC = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


# --------------------------------------------------------------------------
# 2. Request / response models
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    lang: str = "en"          # "en" or "hi" (extend as you add templates)
    default_location: Optional[str] = None  # used if none found in the message


class ChatResponse(BaseModel):
    reply: str
    place: Optional[str] = None
    date: Optional[str] = None
    data: Optional[dict] = None
    alerts: list[str] = []


# --------------------------------------------------------------------------
# 3. Rule-based NLU  (swap for an LLM call if LLM_API_KEY is set — see
#    parse_with_llm() below for the extension point)
# --------------------------------------------------------------------------
DAY_WORDS = {
    "today": 0, "aaj": 0,
    "tomorrow": 1, "kal": 1, "tommorow": 1,
    "day after tomorrow": 2, "parso": 2,
}

RAIN_WORDS = ["rain", "baarish", "barish", "shower", "precipitation"]
WIND_WORDS = ["wind", "hawa", "storm", "aandhi"]
TEMP_WORDS = ["temperature", "taapmaan", "tapman", "hot", "cold", "garmi", "thand"]
ALERT_WORDS = ["alert", "chetavani", "warning", "khatra", "cyclone", "flood", "baadh"]

LOCATION_STOPWORDS = {
    "the", "a", "an", "of", "for", "is", "in", "at", "on", "will", "it",
    "rain", "tomorrow", "today", "weather", "forecast", "hogi", "kya",
    "mein", "ka", "ki", "ke", "kal", "aaj", "mausam",
}


def extract_day_offset(text: str) -> int:
    text_l = text.lower()
    for word, offset in DAY_WORDS.items():
        if word in text_l:
            return offset
    return 0  # default: today


def extract_location(text: str, fallback: Optional[str]) -> Optional[str]:
    """Very light heuristic: look for 'in <Place>' / '<Place> mein', else
    take the longest capitalized word run. Replace with a proper NER model
    or an LLM call for production-grade extraction."""
    m = re.search(r"\bin\s+([A-Za-z][A-Za-z\s]{2,25})", text)
    if m:
        return m.group(1).strip().rstrip("?.!,")

    m = re.search(r"([A-Za-z][A-Za-z]{2,25})\s+mein\b", text)
    if m:
        return m.group(1).strip()

    caps = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    caps = [c for c in caps if c.lower() not in LOCATION_STOPWORDS]
    if caps:
        return caps[0]

    return fallback


def extract_param(text: str) -> str:
    text_l = text.lower()
    if any(w in text_l for w in ALERT_WORDS):
        return "alert"
    if any(w in text_l for w in RAIN_WORDS):
        return "rain"
    if any(w in text_l for w in WIND_WORDS):
        return "wind"
    if any(w in text_l for w in TEMP_WORDS):
        return "temperature"
    return "general"


def parse_with_llm(message: str) -> Optional[dict]:
    """Extension point: if LLM_API_KEY is set, call your LLM of choice here
    to return {"location": ..., "day_offset": ..., "param": ...} for far more
    robust / multilingual understanding than the regex rules above. Left as
    a stub so the app works fully offline-of-LLM out of the box."""
    if not LLM_API_KEY:
        return None
    return None


# --------------------------------------------------------------------------
# 4. Weather data (Open-Meteo — no API key required)
# --------------------------------------------------------------------------
async def geocode(place: str) -> Optional[dict]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(GEOCODE_URL, params={"name": place, "count": 1})
        r.raise_for_status()
        results = r.json().get("results")
        if not results:
            return None
        top = results[0]
        return {"lat": top["latitude"], "lon": top["longitude"],
                "name": top["name"], "country": top.get("country", "")}


async def get_forecast(lat: float, lon: float, day_offset: int) -> dict:
    target_date = date.today() + timedelta(days=day_offset)
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(FORECAST_URL, params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,windspeed_10m_max",
            "timezone": "auto",
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
        })
        r.raise_for_status()
        daily = r.json()["daily"]
        return {
            "date": target_date.isoformat(),
            "weathercode": daily["weathercode"][0],
            "temp_max": daily["temperature_2m_max"][0],
            "temp_min": daily["temperature_2m_min"][0],
            "rain_chance": daily["precipitation_probability_max"][0],
            "wind_max": daily["windspeed_10m_max"][0],
        }


def check_alerts(forecast: dict, lang: str) -> list[str]:
    alerts = []
    if forecast["rain_chance"] is not None and forecast["rain_chance"] >= 70:
        alerts.append(
            "Heavy rain likely — flood risk in low-lying areas." if lang == "en"
            else "Tez baarish ki sambhaavna — nichle ilaakon mein jalbharaav ka khatra."
        )
    if forecast["wind_max"] is not None and forecast["wind_max"] >= 40:
        alerts.append(
            "Strong winds expected — secure loose structures, avoid open areas." if lang == "en"
            else "Tez hawa ki chetavani — khuli jagah avoid karo, dhili cheezein sambhalo."
        )
    if forecast["temp_max"] is not None and forecast["temp_max"] >= 42:
        alerts.append(
            "Heatwave conditions — avoid outdoor exposure during midday." if lang == "en"
            else "Loo/heatwave ka asar — dopahar mein bahar nikalne se bacho."
        )
    if forecast["temp_min"] is not None and forecast["temp_min"] <= 4:
        alerts.append(
            "Cold wave conditions — protect crops and livestock at night." if lang == "en"
            else "Sheetlehar ki chetavani — raat mein fasal aur pashuon ka dhyaan rakho."
        )
    return alerts


# --------------------------------------------------------------------------
# 5. Chat endpoint
# --------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    lang = req.lang if req.lang in TEMPLATES else "en"
    tpl = TEMPLATES[lang]

    if not req.message.strip():
        return ChatResponse(reply=tpl["greeting"])

    parsed = parse_with_llm(req.message)
    if parsed:
        place = parsed.get("location") or req.default_location
        day_offset = parsed.get("day_offset", 0)
        param = parsed.get("param", "general")
    else:
        place = extract_location(req.message, req.default_location)
        day_offset = extract_day_offset(req.message)
        param = extract_param(req.message)

    if not place:
        return ChatResponse(reply=tpl["greeting"])

    geo = await geocode(place)
    if not geo:
        return ChatResponse(reply=tpl["not_found"].format(place=place), place=place)

    forecast = await get_forecast(geo["lat"], geo["lon"], day_offset)
    alerts = check_alerts(forecast, lang)
    when = "aaj" if (lang == "hi" and day_offset == 0) else \
           "kal" if (lang == "hi" and day_offset == 1) else \
           ("today" if day_offset == 0 else "tomorrow" if day_offset == 1 else forecast["date"])

    if param == "alert":
        reply = "\n".join(tpl["alert"].format(place=geo["name"], alert_text=a) for a in alerts) \
            if alerts else tpl["no_alert"].format(place=geo["name"])
    else:
        desc = WEATHER_DESC.get(forecast["weathercode"], "variable conditions")
        reply = tpl["forecast"].format(
            place=geo["name"], when=when, desc=desc,
            temp=round((forecast["temp_max"] + forecast["temp_min"]) / 2),
            rain=forecast["rain_chance"], wind=round(forecast["wind_max"]),
        )
        if alerts:
            reply += "\n" + "\n".join(tpl["alert"].format(place=geo["name"], alert_text=a) for a in alerts)

    return ChatResponse(
        reply=reply, place=geo["name"], date=forecast["date"],
        data=forecast, alerts=alerts,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}