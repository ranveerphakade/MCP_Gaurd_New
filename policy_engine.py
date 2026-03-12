import logging
from typing import Dict, Any, List
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration and model
MODEL_DIR = "models/final_security_classifier"

class PolicyEngine:
    def __init__(self, model_path: str = MODEL_DIR):
        """Initializes the Policy Engine and loads the trained BERT model."""
        logger.info(f"Loading security classification model from {model_path}...")
        try:
            self.classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}. Error: {e}")
            raise

    def evaluate_request(self, tool_text: str) -> Dict[str, Any]:
        """
        Evaluates an MCP tool request payload.
        Maps the predicted label to a security decision (ALLOW, WARN, BLOCK).
        """
        if not tool_text or not tool_text.strip():
            logger.warning("Empty tool text provided. Defaulting to BLOCK decision.")
            return {
                "tool_text": tool_text,
                "risk_label": "unknown",
                "confidence_score": 0.0,
                "decision": "BLOCK",
                "reason": "Empty input text."
            }

        try:
            # Run inference
            result = self.classifier(tool_text[:512])[0] # Truncate to avoid exceeding max sequence length token limits if raw text is long
            
            label_id = result['label']
            score = float(result['score'])

            # Map Model Output Label ID back to human-readable strings
            # In our training dataset: safe=0, suspicious=1, malicious=2
            # HF outputs standard labels like 'LABEL_0', 'LABEL_1', depending on saving configuration
            
            risk_label = "unknown"
            if label_id == "LABEL_0" or label_id == "safe":
                risk_label = "safe"
            elif label_id == "LABEL_1" or label_id == "suspicious":
                risk_label = "suspicious"
            elif label_id == "LABEL_2" or label_id == "malicious":
                risk_label = "malicious"
            else:
                # Fallback if label mapping config exists natively in the loaded model
                risk_label = str(label_id).lower()

            # Policy logic mapping
            if risk_label == "safe":
                decision = "ALLOW"
            elif risk_label == "suspicious":
                decision = "WARN"
            elif risk_label == "malicious":
                decision = "BLOCK"
            else:
                # Edge case, default to secure option
                decision = "BLOCK" 

            return {
                "tool_text": tool_text,
                "risk_label": risk_label,
                "confidence_score": round(score, 4),
                "decision": decision
            }

        except Exception as e:
            logger.error(f"Error evaluating request: {e}")
            return {
                "tool_text": tool_text,
                "risk_label": "error",
                "confidence_score": 0.0,
                "decision": "BLOCK",
                "reason": str(e)
            }

def main():
    """Test block to evaluate example MCP tool requests."""
    print("="*60)
    print("           MCP TOOL POLICY ENGINE INITIALIZATION")
    print("="*60)
    
    try:
        engine = PolicyEngine()
    except Exception as e:
        print("Failed to initialize PolicyEngine. Ensure the model has been trained and exists in models/security_classifier/")
        return

    # Example test cases
    test_cases = [
        "Read weather information from a public API endpoint. read_only_weather_api",
        "Modifies local database records and updates the user table schema. sql_database_modifier",
        "Executes arbitrary bash shell commands on the host machine. system_command_runner",
    ]

    print("\n" + "="*60)
    print("                 EVALUATION RESULTS")
    print("="*60)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}]")
        print(f"Tool Text : {test_text}")
        
        result = engine.evaluate_request(test_text)
        
        risk = result['risk_label'].upper()
        decision = result['decision']
        confidence = result['confidence_score']
        
        print(f"Risk Label: {risk} (Confidence: {confidence:.2%})")
        print(f"Decision  : >> {decision} <<")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
