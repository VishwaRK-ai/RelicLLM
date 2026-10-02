import re

class FinfluencerDetector:
    def __init__(self):
        # SEBI strictly prohibits these phrases in financial promotions
        self.red_flag_keywords = [
            "guaranteed", "sure shot", "sure-shot", "100% return", "multibagger",
            "jackpot", "zero risk", "risk free", "risk-free", "upper circuit", 
            "double your money", "rocket", "confirm profit"
        ]
        
    def analyze_tip(self, text: str):
        report = {
            "is_registered": False,
            "registration_number": None,
            "red_flags_found": [],
            "risk_score": 0, # 0 to 100
            "verdict": ""
        }
        
        # 1. Check for valid SEBI Registration Format
        # Investment Advisers (INA) or Research Analysts (INH) followed by 9 digits
        reg_match = re.search(r'\bIN[AH]\d{9}\b', text, re.IGNORECASE)
        if reg_match:
            report["is_registered"] = True
            report["registration_number"] = reg_match.group(0).upper()
        
        # 2. Heuristic Keyword Search
        text_lower = text.lower()
        for word in self.red_flag_keywords:
            if word in text_lower:
                report["red_flags_found"].append(word)
                report["risk_score"] += 35 # Heavy penalty for prohibited language
                
        # Cap risk score at 100
        report["risk_score"] = min(report["risk_score"], 100)
        
        # 3. Verdict Generation
        if not report["is_registered"] and report["risk_score"] > 0:
            report["verdict"] = "CRITICAL WARNING: Unregistered tipster using prohibited speculative language."
        elif not report["is_registered"]:
            report["verdict"] = "CAUTION: No SEBI Registration (INA/INH) found. This is likely illegal financial advice."
        elif report["is_registered"] and report["risk_score"] > 0:
            report["verdict"] = "WARNING: Registered entity using prohibited 'guaranteed return' language. This violates SEBI advertising codes."
        else:
            report["verdict"] = "PASS: SEBI Registration found and no blatant scam keywords detected. (Always verify the INA/INH number on SEBI's portal)."
            
        return report

# Quick Terminal Test
if __name__ == "__main__":
    detector = FinfluencerDetector()
    
    # Simulate a Telegram scam message
    fake_tip = "Buy Suzlon now! 100% guaranteed upper circuit tomorrow. Jackpot multibagger stock. No risk!"
    print("Test 1 (Scam):", detector.analyze_tip(fake_tip))
    
    # Simulate a compliant SEBI analyst
    compliant_tip = "Reliance Q2 earnings show steady growth. Recommendation: HOLD. SEBI Reg: INH000012345"
    print("\nTest 2 (Compliant):", detector.analyze_tip(compliant_tip))