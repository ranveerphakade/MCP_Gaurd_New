import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Import our previously built Policy Engine which loads the HuggingFace model
from policy_engine import PolicyEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_LOG_FILE = "logs/security_log.csv"

def simulate_tool_execution(tool_text: str):
    """
    Simulates the actual execution of an MCP tool in the host environment.
    """
    print(f"   [EXECUTION] Simulated execution completed for: '{tool_text}'")

def process_tool_request(engine: PolicyEngine, tool_text: str) -> Dict[str, Any]:
    """
    Passes the tool request to the ML policy engine to retrieve a classification result,
    prints the security decision gracefully, and governs execution bounds.
    """
    print("-" * 70)
    print(f"Request: {tool_text}")
    
    # Send tool text to the policy engine
    evaluation = engine.evaluate_request(tool_text)
    
    risk_label = evaluation.get("risk_label", "unknown").upper()
    confidence = evaluation.get("confidence_score", 0.0)
    decision = evaluation.get("decision", "BLOCK").upper()
    
    # Add timestamp for the logging system
    evaluation["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Output cleanly formatted response
    print(f"   [POLICY] Evaluated as {risk_label} (Confidence: {confidence:.2%}) -> Decision: {decision}")
    
    # Handle routing based on the allowed status
    if decision == "ALLOW":
        simulate_tool_execution(tool_text)
    elif decision == "WARN":
        print(f"   [WARNING] Tool flagged as suspicious but proceeding with guarded execution.")
        simulate_tool_execution(tool_text)
    elif decision == "BLOCK":
        print(f"   [ALERT] Execution prevented! Tool flagged as malicious.")
    else:
        print(f"   [ALERT] Unknown security decision '{decision}'. Blocking by default.")
        
    return evaluation
    
def run_simulation() -> List[Dict[str, Any]]:
    """
    Initializes the engine and iterates over a simulated list of MCP requests.
    Returns the accumulated log of evaluation payloads.
    """
    try:
        engine = PolicyEngine()
    except Exception as e:
        logger.error("Simulation aborted: Could not load Policy Engine.")
        return []

    # Hardcoded simulation payloads based on requirements
    simulated_requests = [
        "get_weather fetch weather forecast from API",
        "delete_file remove a local file",
        "run_shell execute bash command on host",
        "read_database query customer table",
        "send_email send automated email"
    ]
    
    print("\n" + "="*70)
    print("             STARTING MCP SECURITY SIMULATION")
    print("="*70)

    security_log = []
    
    for req in simulated_requests:
        eval_result = process_tool_request(engine, req)
        security_log.append(eval_result)
        
    print("-" * 70)
    return security_log

def save_security_log(security_log: List[Dict[str, Any]], output_path: str):
    """
    Transforms the aggregated list of request logs into a structured csv using pandas.
    """
    if not security_log:
        logger.warning("No logs to save.")
        return
        
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(security_log)
        
        # Ensure optimal column ordering
        ordered_cols = ['timestamp', 'tool_text', 'risk_label', 'decision', 'confidence_score']
        # Map any missing columns or extra ones cleanly
        df = df[[col for col in ordered_cols if col in df.columns]]
        
        df.to_csv(output_path, index=False)
        print(f"\n[+] Security log successfully written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write security log: {e}")

def main():
    """
    Main orchestration function running the full simulation pipeline.
    """
    # 1. Run simulation and gather logs
    logs = run_simulation()
    
    if not logs:
        return
        
    # 2. Extract metrics for final reporting
    total_tools = len(logs)
    allowed = sum(1 for log in logs if log.get("decision") == "ALLOW")
    warned = sum(1 for log in logs if log.get("decision") == "WARN")
    blocked = sum(1 for log in logs if log.get("decision") == "BLOCK")

    # 3. Print cleanly formatted final statistics
    print("\n" + "="*70)
    print("                SIMULATION SUMMARY REPORT")
    print("="*70)
    print(f"Total tools evaluated : {total_tools}")
    print(f"Allowed tools         : {allowed}")
    print(f"Warned tools          : {warned}")
    print(f"Blocked tools         : {blocked}")
    print("="*70)
    
    # 4. Save to CSV
    save_security_log(logs, OUTPUT_LOG_FILE)

if __name__ == "__main__":
    main()
