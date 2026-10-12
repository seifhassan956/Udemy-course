
from mcp.server.fastmcp import FastMCP

math_server = FastMCP("Calculator")


@math_server.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@math_server.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    math_server.run(transport="stdio")
