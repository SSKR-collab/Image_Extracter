import re
import os
from image_extractor.base_analyzer import BaseAnalyzer

try:
    import yara
    HAS_YARA = True
except ImportError:
    HAS_YARA = False


class SecurityScanner(BaseAnalyzer):
    """
    Scans OCR text and binary contents for security risks: secrets (API keys, JWTs),
    suspicious commands, IP addresses, external URLs, and runs optional YARA rules.
    """
    VERSION = "1.1.0"

    # Default risk weights, tuneable via analyzer config
    DEFAULT_RISK_WEIGHTS = {
        "embedded_executable": 50,
        "valid_zip_payload": 35,
        "private_key_detected": 45,
        "aws_key_detected": 40,
        "jwt_token_detected": 25,
        "generic_secret_detected": 20,
        "suspicious_command": 35,
        "suspicious_url": 15,
        "custom_png_chunk": 5,
        "high_file_entropy": 10,
        "high_entropy_overlay": 30,
        "yara_rule_match": 40
    }

    # Regular expressions for secrets and credentials
    SECRETS_PATTERNS = {
        "aws_access_key": r"\b(AKIA|ASCA|AOAG|ACCA)[0-9A-Z]{16}\b",
        "jwt_token": r"\beyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*\b",
        "private_key": r"-----BEGIN [A-Z ]+ PRIVATE KEY-----",
        "google_api_key": r"\bAIza[yY][a-zA-Z0-9-_]{35}\b",
        "generic_api_key": r"\b(api_key|client_secret|access_token|db_password|db_pass|aws_secret|secret_key)\b"
    }

    # Suspicious shell and scripting execution markers
    COMMANDS_PATTERNS = {
        "powershell": r"\b(powershell|pwsh)\b",
        "download_helper": r"\b(curl|wget|Invoke-WebRequest|iwr|wget.exe|curl.exe)\b",
        "reverse_shell": r"\b(bash\s+-i|sh\s+-i|nc\s+-e|nc\s+-c)\b",
        "command_exec": r"\b(cmd\.exe|/bin/bash|/bin/sh)\b"
    }

    # URL and IP matching patterns
    URL_PATTERN = r"\bhttps?://[^\s\"']+"
    IP_PATTERN = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    def analyze(self, file_path: str, img, context: dict) -> dict:
        results = {
            "facts": {},
            "indicators": [],
            "assessments": {},
            "errors": []
        }

        # Load configurable weights
        weights = self.DEFAULT_RISK_WEIGHTS.copy()
        if "risk_weights" in self.config:
            weights.update(self.config["risk_weights"])

        # Fetch OCR text from context if available
        ocr_text = context.get("ocr_engine", {}).get("facts", {}).get("raw_text", "")
        
        # Read file bytes for signature searching
        try:
            with open(file_path, "rb") as f:
                # Scan first and last 2MB for string scanning to prevent overhead
                size = os.path.getsize(file_path)
                limit = 2 * 1024 * 1024
                if size > 2 * limit:
                    file_bytes = f.read(limit)
                    f.seek(-limit, os.SEEK_END)
                    file_bytes += f.read()
                else:
                    file_bytes = f.read()
            binary_text = file_bytes.decode("utf-8", errors="ignore")
        except Exception as e:
            binary_text = ""
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "warning",
                "message": f"Could not read bytes for string security scan: {str(e)}"
            })

        # Combined text to scan
        scan_text = f"{ocr_text}\n{binary_text}"

        # 1. Scan for Secrets
        found_secrets = []
        for sec_type, pattern in self.SECRETS_PATTERNS.items():
            matches = re.findall(pattern, scan_text, re.IGNORECASE)
            if matches:
                # Deduplicate and register
                matches = list(set(matches))
                found_secrets.append({
                    "type": sec_type,
                    "count": len(matches),
                    "matches": [str(m)[:30] + "..." if len(str(m)) > 30 else str(m) for m in matches]
                })
                
                # Register Indicators
                severity = "high" if sec_type in ("aws_access_key", "private_key") else "medium"
                results["indicators"].append({
                    "type": f"secret_{sec_type}",
                    "description": f"Potential credentials/secret of type '{sec_type}' discovered.",
                    "severity": severity
                })
        results["facts"]["secrets"] = found_secrets

        # 2. Scan for Suspicious Commands
        found_commands = []
        for cmd_type, pattern in self.COMMANDS_PATTERNS.items():
            matches = re.findall(pattern, scan_text, re.IGNORECASE)
            if matches:
                matches = list(set(matches))
                found_commands.append({
                    "type": cmd_type,
                    "matches": matches
                })
                results["indicators"].append({
                    "type": "suspicious_command",
                    "description": f"Suspicious execution command or tool '{cmd_type}' found in strings.",
                    "severity": "high" if cmd_type == "reverse_shell" else "medium"
                })
        results["facts"]["suspicious_commands"] = found_commands

        # 3. Scan for URLs and IPs
        found_urls = re.findall(self.URL_PATTERN, scan_text)
        found_ips = re.findall(self.IP_PATTERN, scan_text)
        
        # Filter local loopbacks if needed
        found_ips = [ip for ip in found_ips if not ip.startswith(("127.0.0", "0.0.0"))]
        
        results["facts"]["urls"] = list(set(found_urls))
        results["facts"]["ips"] = list(set(found_ips))
        
        if found_urls:
            results["indicators"].append({
                "type": "suspicious_url",
                "description": f"Found {len(results['facts']['urls'])} external URL links inside the document.",
                "severity": "low"
            })
        if found_ips:
            results["indicators"].append({
                "type": "suspicious_ip",
                "description": f"Found {len(results['facts']['ips'])} IP addresses inside the document.",
                "severity": "medium"
            })

        # 4. Optional YARA rule support
        results["facts"]["yara_matches"] = []
        if HAS_YARA and "yara_rules_path" in self.config:
            yara_path = self.config["yara_rules_path"]
            if os.path.exists(yara_path):
                try:
                    rules = yara.compile(yara_path)
                    # Check against both file bytes and text
                    matches = rules.match(data=file_bytes)
                    for m in matches:
                        results["facts"]["yara_matches"].append(m.rule)
                        results["indicators"].append({
                            "type": "yara_rule_match",
                            "description": f"YARA rule match: {m.rule}",
                            "severity": "high"
                        })
                except Exception as e:
                    results["errors"].append({
                        "plugin": self.get_name(),
                        "severity": "warning",
                        "message": f"YARA scanning failed: {str(e)}"
                    })
        elif "yara_rules_path" in self.config:
            results["errors"].append({
                "plugin": self.get_name(),
                "severity": "warning",
                "message": "yara-python not installed, skipping configured YARA rules check."
            })

        # 5. Risk Assessment Calculation
        # Gather indicators from all active plugins in context + current plugin
        all_indicators = results["indicators"].copy()
        for p_name, p_out in context.items():
            if "indicators" in p_out:
                all_indicators.extend(p_out["indicators"])

        # Calculate score based on indicators
        risk_score = 0
        reasons = []
        for ind in all_indicators:
            ind_type = ind["type"]
            weight = weights.get(ind_type, 10) # default weight 10 if not defined
            risk_score += weight
            reasons.append(f"{ind['description']} (+{weight})")

        # Cap score at 100
        risk_score = min(risk_score, 100)

        # Classify Risk Level
        if risk_score <= 15:
            risk_level = "Low"
        elif risk_score <= 45:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Determine confidence of risk score
        confidence = 0.90 if len(all_indicators) > 0 else 0.95

        results["assessments"]["security_risk"] = {
            "score": risk_score,
            "level": risk_level,
            "confidence": confidence,
            "reason_summary": reasons if reasons else ["No security indicators triggered."]
        }

        return results
