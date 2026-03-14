"""
Tool implementations for Nova 2 Sonic voicebot.
Mirrors the TypeScript tools in src/tools/.
"""

import datetime
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tools")


# ─────────────────────────────────────────────────────────────────────────────
# Base Tool
# ─────────────────────────────────────────────────────────────────────────────

class BaseTool:
    name: str = ""
    description: str = ""

    def get_spec(self) -> dict:
        raise NotImplementedError

    def execute(self, params: dict) -> Any:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# DateTime Tool
# ─────────────────────────────────────────────────────────────────────────────

class DateTimeTool(BaseTool):
    name = "dateTime"
    description = "Returns the current date and time"

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Optional timezone name (e.g. 'America/New_York'). Defaults to UTC.",
                            }
                        },
                        "required": [],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        now = datetime.datetime.utcnow()
        return {
            "utc": now.isoformat() + "Z",
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "dayOfWeek": now.strftime("%A"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Weather Tool  (Open-Meteo free API)
# ─────────────────────────────────────────────────────────────────────────────

class WeatherTool(BaseTool):
    name = "weather"
    description = "Get current weather and forecasts for a location using Open-Meteo API"

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number", "description": "Latitude"},
                            "longitude": {"type": "number", "description": "Longitude"},
                            "location_name": {"type": "string", "description": "Human readable location name"},
                        },
                        "required": ["latitude", "longitude"],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        lat = params.get("latitude", 0)
        lon = params.get("longitude", 0)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,apparent_temperature,precipitation,weather_code,"
            f"wind_speed_10m,wind_direction_10m,relative_humidity_2m"
            f"&temperature_unit=celsius"
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            current = data.get("current", {})
            return {
                "location": params.get("location_name", f"{lat},{lon}"),
                "temperature_c": current.get("temperature_2m"),
                "feels_like_c": current.get("apparent_temperature"),
                "humidity_pct": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "time": current.get("time"),
            }
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia Tool
# ─────────────────────────────────────────────────────────────────────────────

class WikipediaTool(BaseTool):
    name = "wikipedia"
    description = "Search and retrieve summaries from Wikipedia"

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query or article title"},
                            "sentences": {"type": "integer", "description": "Number of sentences to return (default 3)"},
                        },
                        "required": ["query"],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        query = params.get("query", "")
        sentences = params.get("sentences", 3)
        encoded = urllib.parse.quote(query)
        url = (
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NovaSonicVoicebot/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            extract = data.get("extract", "")
            # Truncate to N sentences
            import re
            sentence_list = re.split(r'(?<=[.!?])\s+', extract)
            truncated = " ".join(sentence_list[:sentences])
            return {
                "title": data.get("title", ""),
                "summary": truncated,
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "thumbnail": data.get("thumbnail", {}).get("source", ""),
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Fallback: search
                return self._search_fallback(query, sentences)
            return {"error": str(e)}
        except Exception as e:
            return {"error": str(e)}

    def _search_fallback(self, query: str, sentences: int) -> dict:
        encoded = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&srlimit=1"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NovaSonicVoicebot/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            results = data.get("query", {}).get("search", [])
            if results:
                title = results[0]["title"]
                return self.execute({"query": title, "sentences": sentences})
            return {"error": "No Wikipedia article found for query"}
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Location Search Tool  (Nominatim OpenStreetMap)
# ─────────────────────────────────────────────────────────────────────────────

class LocationSearchTool(BaseTool):
    name = "locationSearch"
    description = "Find places, addresses, and coordinates using OpenStreetMap Nominatim"

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Place name or address to search"},
                            "limit": {"type": "integer", "description": "Max results (default 3)"},
                        },
                        "required": ["query"],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        query = params.get("query", "")
        limit = params.get("limit", 3)
        encoded = urllib.parse.quote(query)
        url = (
            f"https://nominatim.openstreetmap.org/search"
            f"?q={encoded}&format=json&limit={limit}&addressdetails=1"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NovaSonicVoicebot/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            places = []
            for item in data:
                places.append({
                    "name": item.get("display_name", ""),
                    "latitude": float(item.get("lat", 0)),
                    "longitude": float(item.get("lon", 0)),
                    "type": item.get("type", ""),
                    "country": item.get("address", {}).get("country", ""),
                })
            return {"query": query, "results": places}
        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Tool  (extended thinking stub)
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningTool(BaseTool):
    name = "reasoning"
    description = "Extended thinking for complex multi-step reasoning questions. Use this when you need to think carefully before answering."

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "The question or problem to reason about"},
                            "context": {"type": "string", "description": "Optional additional context"},
                        },
                        "required": ["question"],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        # The actual reasoning is done by Nova Sonic itself — this tool provides
        # a structured scaffold for it to use.
        question = params.get("question", "")
        context = params.get("context", "")
        return {
            "instruction": "Please reason step-by-step to answer the following question.",
            "question": question,
            "context": context,
            "format": "Think through this carefully before giving your final answer.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Transcript Correction Tool
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptCorrectionTool(BaseTool):
    name = "transcriptCorrection"
    description = "Fix ASR transcription errors. Use when you notice likely speech recognition mistakes in the user's input."

    def get_spec(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "original": {"type": "string", "description": "The original ASR transcript"},
                            "corrected": {"type": "string", "description": "The corrected transcript"},
                            "corrections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "original": {"type": "string"},
                                        "corrected": {"type": "string"},
                                    },
                                },
                                "description": "List of specific corrections made",
                            },
                        },
                        "required": ["original", "corrected"],
                    }
                },
            }
        }

    def execute(self, params: dict) -> dict:
        return {
            "original": params.get("original", ""),
            "corrected": params.get("corrected", ""),
            "corrections": params.get("corrections", []),
            "applied": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        for tool_cls in [
            DateTimeTool,
            WeatherTool,
            WikipediaTool,
            LocationSearchTool,
            ReasoningTool,
            TranscriptCorrectionTool,
        ]:
            t = tool_cls()
            self._tools[t.name] = t
        logger.info(f"Registered tools: {list(self._tools.keys())}")

    def get_specs(self, enabled_names: List[str]) -> List[dict]:
        """Return Bedrock tool specs for the given list of enabled tool names."""
        specs = []
        for name in enabled_names:
            tool = self._tools.get(name)
            if tool:
                specs.append(tool.get_spec())
        return specs

    def execute(self, name: str, params: dict) -> Any:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool.execute(params)
        except Exception as e:
            logger.error(f"Tool {name} execution error: {e}")
            return {"error": str(e)}

    @property
    def available_tools(self) -> List[str]:
        return list(self._tools.keys())
