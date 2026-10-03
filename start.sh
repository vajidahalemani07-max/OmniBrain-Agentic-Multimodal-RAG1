#!/bin/bash
export PYTHONPATH=.

# 1. Backend ko dedicated python script se chalao
python run_backend.py &

# 2. 5 second wait taaki backend port 8000 par listen karna shuru kar de
sleep 5

# 3. Streamlit ko Render ke public port par chalao
streamlit run frontend/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false