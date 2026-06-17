from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import random

app = FastAPI()

# Global state
state = {
    "is_manual": False,
    "active_limit": 50,
    "traffic_density": 30, # Percentage 0-100
}

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

class ModeUpdate(BaseModel):
    is_manual: bool

class LimitUpdate(BaseModel):
    limit: int

class TrafficUpdate(BaseModel):
    density: int

def calculate_limit_from_traffic(density: int) -> int:
    """Logic to determine speed limit based on traffic density."""
    if density > 85:
        return 20   # Gridlock
    elif density > 70:
        return 40   # Heavy traffic
    elif density > 50:
        return 60   # Moderate traffic
    elif density > 30:
        return 80   # Light traffic
    elif density > 15:
        return 100  # Clear road
    else:
        return 120  # Empty road

async def traffic_simulator():
    """Simulates realistic traffic density fluctuations."""
    while True:
        # Gradually shift density by -5 to +5 percent
        change = random.randint(-5, 5)
        new_density = max(0, min(100, state["traffic_density"] + change))
        state["traffic_density"] = new_density
        
        # If in AUTO mode, update the limit based on this new density
        if not state["is_manual"]:
            new_limit = calculate_limit_from_traffic(new_density)
            if new_limit != state["active_limit"]:
                state["active_limit"] = new_limit
                print(f"Traffic Update: Density {new_density}% -> Setting limit to {new_limit} km/h")
        
        await asyncio.sleep(5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(traffic_simulator())

@app.get("/")
async def get_index():
    return FileResponse("speed-limit-manager.html")

@app.get("/api/status")
async def get_status():
    return state

@app.post("/api/traffic")
async def set_traffic(data: TrafficUpdate):
    """Manual override for testing traffic density."""
    if data.density < 0 or data.density > 100:
        raise HTTPException(status_code=400, detail="Density must be between 0 and 100")
    
    state["traffic_density"] = data.density
    if not state["is_manual"]:
        state["active_limit"] = calculate_limit_from_traffic(data.density)
        
    return state

@app.post("/api/mode")
async def update_mode(data: ModeUpdate):
    state["is_manual"] = data.is_manual
    if not data.is_manual:
        state["active_limit"] = calculate_limit_from_traffic(state["traffic_density"])
    return state

@app.get("/api/test")
async def get_test():
    return {"message": "Test endpoint is working!"}

@app.post("/api/limit")
async def update_limit(data: LimitUpdate):
    if data.limit < 5 or data.limit > 200:
        raise HTTPException(status_code=400, detail="Limit must be between 5 and 200")
    if not state["is_manual"]:
        raise HTTPException(status_code=400, detail="Cannot set limit in auto mode")
        
    state["active_limit"] = data.limit
    return state

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
