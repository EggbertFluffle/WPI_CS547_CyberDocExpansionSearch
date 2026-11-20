import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import rag_npy

# To run: uvicorn app:app --host 0.0.0.0 --port 8000

app = FastAPI( 
   title="CyberDoc Expansion Search", 
   version="1.0.0" 
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


pipeline = rag_npy.load_pipeline()  # loads the index and models once, at startup
lock = threading.Lock()  # the models can't safely handle two requests at once

@app.get("/")
async def root():
    return {"message": "Welcome to the CyberDoc Expansion Search API!"}

@app.get("/search")
def search(query: str):
    try:
        with lock:
            results = rag_npy.search_exploits(query, *pipeline)
    except SystemExit as exc:  # rag_npy raises this when no retrieved doc is found in the corpus
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "results": [
            {
                "rank": i,
                "exploit_id": r["exploit_id"],
                "file": r["path"],
                "explanation": r["explanation"],
            }
            for i, r in enumerate(results, start=1)
        ]
    }