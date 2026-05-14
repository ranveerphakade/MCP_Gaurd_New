import subprocess
import json

config = {
    "command": "python",
    "args": ["C:\\Users\\ASUS\\Desktop\\MCPCorpus-main\\security_tools_mcp.py"]
}
subprocess.run(["openclaw", "mcp", "set", "security_tester", json.dumps(config)], shell=True)
