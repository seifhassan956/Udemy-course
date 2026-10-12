
from mcp.server.fastmcp import FastMCP

weather_server = FastMCP("Weather")


@weather_server.tool()
def get_weather(city: str) -> str:
    """Get the current mock weather for a city."""
    return (
        f"The current weather in {city} "
        "is sunny with a temperature of 25°C."
    )


if __name__ == "__main__":
    weather_server.run(transport="stdio")
