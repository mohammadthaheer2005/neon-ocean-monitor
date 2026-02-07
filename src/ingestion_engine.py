import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatelliteStream:
    """
    The 'Nerve Center' of NeonOcean.
    Connects to live OPeNDAP/Thredds servers to ingest multi-dimensional ocean data.
    """
    
    def __init__(self, use_synthetic_fallback=True):
        self.use_synthetic_fallback = use_synthetic_fallback
        # NOAA OISST V2.1 (Optimum Interpolation Sea Surface Temperature) - Real-time
        self.sst_source_url = "https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/V2.1/AVHRR/{year}{month}/oisst-avhrr-v02r01.{date}.nc"
        
    def fetch_live_sst(self, date=None, lat=None, lon=None):
        """
        Fetches the global Sea Surface Temperature (SST) field for a specific date.
        If date is None, fetches yesterday's data.
        Optional lat/lon allows for regional slicing.
        """
        if date is None:
            date = datetime.now() - timedelta(days=2)
            
        date_str = date.strftime("%Y%m%d")
        year_str = date.strftime("%Y")
        month_str = date.strftime("%m")
        
        url = self.sst_source_url.format(year=year_str, month=month_str, date=date_str)
        logger.info(f"🛰️  Connecting to NOAA Satellite Feed: {url}")
        
        try:
            ds = xr.open_dataset(url, decode_times=False)
            sst_grid = ds['sst']
            
            # Dynamic regional slicing (default to Indian Ocean if none provided)
            target_lat = lat if lat is not None else 10.0
            target_lon = lon if lon is not None else 80.0
            
            # Slice a 10x10 degree box around the target
            regional_sst = sst_grid.sel(
                lat=slice(target_lat - 5, target_lat + 5), 
                lon=slice(target_lon - 5, target_lon + 5)
            )
            
            logger.info(f"✅ Live Data Packet Received for location ({target_lat}, {target_lon})")
            return regional_sst
            
        except Exception as e:
            logger.error(f"❌ Connection Failed: {e}")
            if self.use_synthetic_fallback:
                logger.warning("⚠️  Switching to SYNTHETIC GENERATION MODE")
                return self._generate_synthetic_ocean(date, lat, lon)
            else:
                raise ConnectionError("Could not fetch live data.")

    def _generate_synthetic_ocean(self, date, lat=None, lon=None):
        """
        Generates a high-fidelity synthetic ocean grid centered on provided coordinates.
        Temperatures are varied to ensure location matters (Equator = Warm, Poles = Cold).
        """
        target_lat = lat if lat is not None else 10.0
        target_lon = lon if lon is not None else 80.0
        
        lats = np.linspace(target_lat - 5, target_lat + 5, 100)
        lons = np.linspace(target_lon - 5, target_lon + 5, 160)
        
        data = np.zeros((1, 1, len(lats), len(lons)))
        for i, x in enumerate(lats):
            for j, y in enumerate(lons):
                # Simulated pattern: Equator (0 deg) is warmest (~31C), 
                # temperature drops as lat increases/decreases.
                # Linear drop: 32C at Equator - 0.4C per degree of latitude
                base_temp = 32.0 - (abs(x) * 0.4)
                
                # Add location-specific eddies and noise
                eddy_noise = np.sin(x*0.8) * np.cos(y*0.8) * 3.0
                seasonal_drift = np.sin(date.timetuple().tm_yday / 365 * 2 * np.pi) * 2.0
                
                data[0, 0, i, j] = base_temp + eddy_noise + seasonal_drift + np.random.normal(0, 0.5)
                
        ds = xr.Dataset(
            data_vars=dict(sst=(["time", "zlev", "lat", "lon"], data)),
            coords=dict(lat=lats, lon=lons, time=[0], zlev=[0]),
            attrs=dict(description="Synthetic NeonOcean Digital Twin Data")
        )
        return ds['sst']

if __name__ == "__main__":
    # Test the ingestion
    stream = SatelliteStream()
    data = stream.fetch_live_sst()
    print("Preview of Ingested Ocean Data:")
    print(data)
