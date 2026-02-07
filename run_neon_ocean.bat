@echo off
TITLE NeonOcean V4.0 Startup
ECHO ==================================================
ECHO      NEON OCEAN: HYBRID INTELLIGENCE SYSTEM
ECHO      V4.1 (Groq AI + Hybrid ML + Live Data)
ECHO ==================================================
ECHO.
ECHO [1/3] Navigating to Project Command Center...
cd /d "C:\Users\moham\.gemini\antigravity\scratch"

ECHO [2/3] Initializing AI Neuromorphic Core...
ECHO [3/3] Launching Dashboard...
ECHO.
streamlit run app/main.py --server.address 0.0.0.0
PAUSE
