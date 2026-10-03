import os
import sys

# Render ke PORT environment variable ko override karo taaki Uvicorn 10000 na le
os.environ["PORT"] = "8000"

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)