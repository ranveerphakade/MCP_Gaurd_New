import os
import json
import csv
import logging
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from policy_engine import PolicyEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load engine globally
engine = None
try:
    engine = PolicyEngine()
    logger.info("Policy Engine initialized for demo server.")
except Exception as e:
    logger.error(f"Failed to load Policy Engine: {e}")

LOG_FILE = "logs/security_log.csv"

class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/static/index.html'
        return super().do_GET()
    
    def do_POST(self):
        if self.path == '/api/evaluate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            tool = data.get('tool', '')
            action = data.get('action', '')
            tool_text = f"{tool} {action}"
            
            if not engine:
                self.send_error(500, "Policy Engine not loaded")
                return
            
            result = engine.evaluate_request(tool_text)
            
            # Save to log
            try:
                os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
                file_exists = os.path.exists(LOG_FILE)
                with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'tool_text', 'risk_label', 'decision', 'confidence_score'])
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([timestamp, tool_text, result.get('risk_label'), result.get('decision'), result.get('confidence_score')])
            except Exception as e:
                logger.error(f"Error writing to log: {e}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

def run(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, DemoHandler)
    logger.info(f"Starting Demo UI server on port {port}... Open http://localhost:{port} in your browser.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logger.info("Server stopped.")

if __name__ == '__main__':
    run()
