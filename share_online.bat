@echo off
TITLE Share NeonOcean Online
ECHO ==================================================
ECHO      SHARING NEON OCEAN WITH THE WORLD
ECHO      (Powered by localtunnel)
ECHO ==================================================
ECHO.
ECHO [IMPORTANT] Ensure your Streamlit app is ALREADY running!
ECHO [IMPORTANT] If you are asked for a "Tunnel Password", use the IP below:
ECHO.
curl -4 icanhazip.com
ECHO.
ECHO [INFO] Generating public URL...
call npx -y localtunnel --port 8501
PAUSE
