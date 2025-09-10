import os
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, Header, Query
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv
import requests


load_dotenv()


API_KEY = os.getenv("API_KEY", "")
CSV_PATH = os.getenv("LOADS_CSV", "Sample_Loads_Dataset__FDE_.csv")
FMCSA_API_URL = os.getenv("FMCSA_API_URL", "https://mobile.fmcsa.dot.gov/qc/services/carriers/" )
FMCSA_API_KEY = os.getenv("FMCSA_API_KEY", "")

app = FastAPI(title="FDE Loads API", version="1.0.0")

def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server missing API_KEY configuration")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

class Load(BaseModel):
    load_id: str
    origin: str
    destination: str
    pickup_datetime: str
    delivery_datetime: str
    equipment_type: str
    loadboard_rate: float
    notes: str
    weight: int
    commodity_type: str
    num_of_pieces: int
    miles: int
    dimensions: str

def read_df() -> pd.DataFrame:
    # For Vercel deployment, use absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, CSV_PATH)
    
    if not os.path.exists(csv_file):
        raise HTTPException(status_code=500, detail=f"CSV not found at {csv_file}")
    df = pd.read_csv(csv_file)
    # Normalize datetime columns
    for col in ["pickup_datetime", "delivery_datetime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/mcnumber/verify", dependencies=[Depends(verify_api_key)])
def verify_mc(mc: str = Query(..., description="MC number to verify")):
    if not mc:
        raise HTTPException(status_code=400, detail="Missing MC number")
    # Replace :dotNumber in the URL with the provided MC/DOT number
    url = FMCSA_API_URL.replace(":mcNumber", str(mc))
    # Remove quotes if present from .env value
    url = url.strip('"')
    print(f"Fetching FMCSA data from {url}")
    url = f"{url}?webKey={FMCSA_API_KEY}"
    print(f"Full URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"FMCSA response data: {data}")
        content = data.get("content")
        if not content or not isinstance(content, dict) or "carrier" not in content:
            raise HTTPException(status_code=404, detail={"error": "Carrier not found or unexpected FMCSA response", "raw_response": data})
        carrier_info = content["carrier"]
        return {
            "mc_number": mc,
            "carrier_name": carrier_info.get("legalName"),
            "dba_name": carrier_info.get("dbaName"),
            "status": carrier_info.get("carrierOperation", {}).get("carrierOperationDesc") if carrier_info.get("carrierOperation") else None,
            "dot_number": carrier_info.get("dotNumber"),
            "entity_type": carrier_info.get("entityType") if "entityType" in carrier_info else None,
            "active": carrier_info.get("allowedToOperate"),
            "address": {
                "street": carrier_info.get("phyStreet"),
                "city": carrier_info.get("phyCity"),
                "state": carrier_info.get("phyState"),
                "zip": carrier_info.get("phyZipcode"),
                "country": carrier_info.get("phyCountry")
            }
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail={"error": "Error fetching data from FMCSA", "details": str(e)})

@app.get("/loads/search", response_model=List[Load], dependencies=[Depends(verify_api_key)])
def get_loads(
    load_id: Optional[str] = Query(None, description="Filter by load ID"),
    origin: Optional[str] = Query(None, description="Filter by origin city/state"),
    destination: Optional[str] = Query(None, description="Filter by destination city/state"),
    equipment_type: Optional[str] = Query(None, description="Filter by equipment type (e.g., 'Dry Van', 'Reefer', 'Flatbed')"),
    commodity_type: Optional[str] = Query(None, description="Filter by commodity type"),
    min_rate: Optional[float] = Query(None, description="Minimum loadboard rate"),
    max_rate: Optional[float] = Query(None, description="Maximum loadboard rate"),
    min_weight: Optional[int] = Query(None, description="Minimum weight"),
    max_weight: Optional[int] = Query(None, description="Maximum weight"),
    min_miles: Optional[int] = Query(None, description="Minimum miles"),
    max_miles: Optional[int] = Query(None, description="Maximum miles"),
    pickup_date: Optional[str] = Query(None, description="Filter by pickup date (YYYY-MM-DD format)"),
    delivery_date: Optional[str] = Query(None, description="Filter by delivery date (YYYY-MM-DD format)"),
    limit: Optional[int] = Query(None, description="Limit number of results returned")
):
    """
    Retrieve loads with optional filtering parameters.
    All parameters are optional - if none provided, returns all loads.
    """
    try:
        df = read_df()
        
        # Apply filters if provided
        if load_id:
            df = df[df['load_id'].str.contains(load_id, case=False, na=False)]
        
        if origin:
            df = df[df['origin'].str.contains(origin, case=False, na=False)]
        
        if destination:
            df = df[df['destination'].str.contains(destination, case=False, na=False)]
        
        if equipment_type:
            df = df[df['equipment_type'].str.contains(equipment_type, case=False, na=False)]
        
        if commodity_type:
            df = df[df['commodity_type'].str.contains(commodity_type, case=False, na=False)]
        
        if min_rate is not None:
            df = df[df['loadboard_rate'] >= min_rate]
        
        if max_rate is not None:
            df = df[df['loadboard_rate'] <= max_rate]
        
        if min_weight is not None:
            df = df[df['weight'] >= min_weight]
        
        if max_weight is not None:
            df = df[df['weight'] <= max_weight]
        
        if min_miles is not None:
            df = df[df['miles'] >= min_miles]
        
        if max_miles is not None:
            df = df[df['miles'] <= max_miles]
        
        if pickup_date:
            try:
                pickup_dt = pd.to_datetime(pickup_date)
                df = df[df['pickup_datetime'].dt.date == pickup_dt.date()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid pickup_date format. Use YYYY-MM-DD")
        
        if delivery_date:
            try:
                delivery_dt = pd.to_datetime(delivery_date)
                df = df[df['delivery_datetime'].dt.date == delivery_dt.date()]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid delivery_date format. Use YYYY-MM-DD")
        
        # Apply limit if provided
        if limit is not None and limit > 0:
            df = df.head(limit)
        
        # Convert datetime columns back to string for JSON serialization
        df_copy = df.copy()
        df_copy['pickup_datetime'] = df_copy['pickup_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        df_copy['delivery_datetime'] = df_copy['delivery_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Convert to list of dictionaries
        loads = df_copy.to_dict('records')
        
        return loads
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving loads: {str(e)}")

# Create a handler for Vercel
def handler(request):
    return app

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)