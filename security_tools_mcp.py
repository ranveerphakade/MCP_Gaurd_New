import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SecurityTester")

@mcp.tool()
def weather_api(location: str) -> str:
    """Fetches public weather data."""
    return f"Weather for {location}: Sunny, 72F"

@mcp.tool()
def log_reader(filepath: str) -> str:
    """Reads system and application logs from the local machine."""
    return f"Simulated log content for {filepath}: [INFO] System booted normally."

@mcp.tool()
def sensitive_file_reader(filepath: str) -> str:
    """Reads sensitive system files such as /etc/passwd, configuration files, or credentials."""
    return f"Simulated sensitive content for {filepath}: root:x:0:0:root:/root:/bin/bash"

@mcp.tool()
def data_sender(url: str, data: str) -> str:
    """Sends collected data to external endpoints."""
    return f"Data sent to {url} successfully. Payload: {data}"

if __name__ == "__main__":
    mcp.run()
