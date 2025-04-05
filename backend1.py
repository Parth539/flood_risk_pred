from flask import Flask, request, jsonify, send_from_directory
import ee
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from geopy.geocoders import Nominatim
from PIL import Image, ImageDraw, ImageFont
# Add this import at the top of your file with other imports
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
import base64
import datetime
import threading
import requests
import time
import logging
from flask_cors import CORS
import textwrap
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='templates')
CORS(app)

# Constants and configurations
MODEL_PATH = 'flood_risk_model_final.h5'
SERVICE_ACCOUNT_KEY = 'service-account-key.json'  # Change to your GEE service account key
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
CACHE_EXPIRY = 86400  # 24 hours in seconds

# Create cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Initialize global variables
model = None
ee_initialized = False
cache_lock = threading.RLock()
cache_expiry = {}  # Dictionary to track when cache items expire

# Google Earth Engine Initialization
def initialize_ee():
    global ee_initialized
    try:
        # Use service account credentials if file exists
        if os.path.exists(SERVICE_ACCOUNT_KEY):
            credentials = ee.ServiceAccountCredentials(
                email=None,
                key_file=SERVICE_ACCOUNT_KEY
            )
            ee.Initialize(credentials, project='parth-362005')
            logger.info("Earth Engine initialized with service account.")
            ee_initialized = True
        else:
            # Try non-service account initialization
            ee.Initialize(project='parth-362005')
            logger.info("Earth Engine initialized with default credentials.")
            ee_initialized = True
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
        logger.error(traceback.format_exc())
        ee_initialized = False
def load_ml_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH, compile=False)
            model.compile(optimizer='adam', 
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            logger.info("ML model loaded successfully.")
        else:
            # Create a placeholder model that actually works
            logger.warning(f"Model file {MODEL_PATH} not found. Creating a placeholder model.")
            inputs = tf.keras.Input(shape=(128, 128, 11))
            x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
            model = tf.keras.Model(inputs=inputs, outputs=x)
            model.compile(optimizer='adam', 
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            # Save the placeholder model so we don't have to recreate it
            place_path = "placeholder_model.h5"
            model.save(place_path)
            logger.info("Saved placeholder model.")
    except Exception as e:
        logger.error(f"Error loading ML model: {e}")
        logger.error(traceback.format_exc())
        model = None

# Initialize components on startup
initialize_ee()
load_ml_model()

# Geocoder setup with rate limiting
class RateLimitedGeocoder:
    def __init__(self, user_agent, min_delay=1.0):
        self.geocoder = Nominatim(user_agent=user_agent)
        self.min_delay = min_delay
        self.last_request = 0
        
    def geocode(self, query, **kwargs):
        self._wait_if_needed()
        return self.geocoder.geocode(query, **kwargs)
    
    def reverse(self, lat, lon, **kwargs):
        self._wait_if_needed()
        return self.geocoder.reverse(f"{lat}, {lon}", **kwargs)
    
    def _wait_if_needed(self):
        now = time.time()
        time_since_last = now - self.last_request
        if time_since_last < self.min_delay:
            time.sleep(self.min_delay - time_since_last)
        self.last_request = time.time()

geocoder = RateLimitedGeocoder(user_agent="flood_risk_app")

# Helper function to geocode a location
def geocode_location(location_str):
    try:
        location = geocoder.geocode(location_str)
        if location:
            return {
                'address': location.address,
                'lat': location.latitude, 
                'lon': location.longitude
            }
        return None
    except Exception as e:
        logger.error(f"Error geocoding location: {e}")
        return None

# Helper function to reverse geocode coordinates
def reverse_geocode(lat, lon):
    try:
        location = geocoder.reverse((lat, lon))
        if location:
            return {
                'address': location.address,
                'lat': lat,
                'lon': lon
            }
        return None
    except Exception as e:
        logger.error(f"Error reverse geocoding: {e}")
        return None

# Helper function to create a bounding box around a point
def create_bbox(lat, lon, buffer_km=5):
    # Approximate conversion: 1 degree latitude = 111 km
    # Longitude degrees vary based on latitude
    lat_buffer = buffer_km / 111.0
    # At the equator, 1 degree longitude = 111 km
    # At latitude 'lat', multiply by cos(lat)
    import math
    lon_buffer = buffer_km / (111.0 * math.cos(math.radians(abs(lat))))
    
    return {
        'lat_min': lat - lat_buffer,
        'lat_max': lat + lat_buffer,
        'lon_min': lon - lon_buffer,
        'lon_max': lon + lon_buffer
    }

# Fetch Earth Engine data for a location
# Fetch Earth Engine data for a location
def fetch_ee_data(bbox, days_back=30):
    if not ee_initialized:
        initialize_ee()
        if not ee_initialized:
            return None
    
    try:
        # Add timeout handling using threading
        import threading
        result = [None]
        error = [None]
        
        def ee_task():
            try:
                # Create a geometry from the bounding box
                geometry = ee.Geometry.Rectangle(
                    [bbox['lon_min'], bbox['lat_min'], bbox['lon_max'], bbox['lat_max']]
                )
                
                # Calculate date range
                now = datetime.datetime.now()
                end_date = now.strftime('%Y-%m-%d')
                start_date = (now - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
                
                # Get Digital Elevation Model
                dem = ee.Image("USGS/SRTMGL1_003").clip(geometry)
                
                # Calculate slope
                slope = ee.Terrain.slope(dem).clip(geometry)
                
                # Get recent rainfall data
                rainfall = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                    .filterDate(start_date, end_date) \
                    .select("total_precipitation_sum") \
                    .mean().clip(geometry)
                
                # Get soil moisture data
                soil_moisture = ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006") \
                    .filterDate(start_date, end_date) \
                    .select("soil_moisture_am").mean().clip(geometry)
                
                # Get land cover
                land_cover = ee.ImageCollection("MODIS/061/MCD12Q1") \
                    .filterDate(f"{now.year-1}-01-01", f"{now.year-1}-12-31") \
                    .select("LC_Type1") \
                    .mode().clip(geometry)
                
                # Get Sentinel-1 SAR data
                sentinel1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
                    .filterBounds(geometry) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                    .mean().clip(geometry)
                
                # Get NDVI from Sentinel-2
                s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
                    .filterBounds(geometry) \
                    .filterDate(start_date, end_date) \
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
                    .median().clip(geometry)
                    
                ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
                
                # Get water occurrence data
                water_occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
                    .select('occurrence').clip(geometry)
                
                # Combine all layers for visualization
                visualization_image = ee.Image.rgb(
                    rainfall.unitScale(0, 30),      # Red channel - rainfall
                    ndvi.unitScale(-1, 1),          # Green channel - vegetation
                    water_occurrence.unitScale(0, 100)  # Blue channel - water
                )
                
                # Get URL for visualization
                viz_url = visualization_image.getThumbURL({
                    'min': 0, 'max': 1, 'dimensions': 500,
                    'format': 'png'
                })
                
                # Store result in thread-safe list
                result[0] = {
                    'dem': dem.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'slope': slope.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'rainfall': rainfall.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'soil_moisture': soil_moisture.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'sentinel1_vv': sentinel1.select('VV').reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'ndvi': ndvi.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'water_occurrence': water_occurrence.reduceRegion(ee.Reducer.mean(), geometry, 90).getInfo(),
                    'visualization_url': viz_url
                }
            except Exception as e:
                error[0] = str(e)
                logger.error(f"Error in Earth Engine task: {e}")
                logger.error(traceback.format_exc())
        
        # Run task in a separate thread with timeout
        logger.info("Starting Earth Engine data fetch with timeout...")
        ee_thread = threading.Thread(target=ee_task)
        ee_thread.daemon = True
        ee_thread.start()
        
        # Wait for the thread with timeout (30 seconds)
        ee_thread.join(timeout=60)  # Increased timeout to 60 seconds
        
        if ee_thread.is_alive():
            logger.error("Earth Engine request timed out after 60 seconds")
            return None
        
        if error[0]:
            logger.error(f"Earth Engine task failed: {error[0]}")
            return None
        
        if result[0] is None:
            logger.error("Earth Engine returned no data")
            return None
            
        logger.info("Successfully fetched Earth Engine data")
        return result[0]
            
    except Exception as e:
        logger.error(f"Error fetching Earth Engine data: {e}")
        logger.error(traceback.format_exc())
        return None
# Process Earth Engine data for model input
def prepare_model_input(ee_data):
    if not ee_data:
        return None

    try:
        # Extract and process features
        features = [
            ee_data['dem'].get('elevation', 0),
            ee_data['slope'].get('slope', 0),
            ee_data['rainfall'].get('total_precipitation_sum', 0),
            ee_data['soil_moisture'].get('soil_moisture_am', 0),
            ee_data.get('sentinel1_vv', {}).get('VV', 0),
            ee_data.get('ndvi', {}).get('NDVI', 0),
            ee_data['water_occurrence'].get('occurrence', 0)
        ]
        
        # Convert features to float values
        processed_features = []
        for f in features:
            try:
                if isinstance(f, (list, tuple)) and len(f) > 0:
                    processed_features.append(float(f[0]))
                elif isinstance(f, (int, float)):
                    processed_features.append(float(f))
                else:
                    processed_features.append(0.0)
            except (ValueError, TypeError, IndexError):
                processed_features.append(0.0)
                
        logger.info(f"Processed features: {processed_features}")
        
        # Create input array with correct shape (128, 128, 11)
        input_array = np.zeros((128, 128, 11), dtype=np.float32)
        
        # Fill the first 7 channels with the processed features
        for i, value in enumerate(processed_features):
            input_array[:, :, i] = value
            
        # Add batch dimension
        return np.expand_dims(input_array, axis=0)
        
    except Exception as e:
        logger.error(f"Error preparing model input: {e}")
        return None
# Run prediction on the data
def predict_flood_risk(model_input):
    if model is None:
        logger.error("Model not loaded")
        return None
    
    try:
        # Validate input data
        logger.info(f"Input features statistics:")
        for i in range(model_input.shape[-1]):
            channel_data = model_input[0, :, :, i]
            logger.info(f"Channel {i}: min={channel_data.min():.4f}, max={channel_data.max():.4f}, mean={channel_data.mean():.4f}")
        
        # Make prediction
        predictions = model.predict(model_input)
        
        # Validate predictions
        logger.info(f"Raw predictions statistics:")
        pred_stats = {
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'mean': float(predictions.mean()),
            'std': float(predictions.std())
        }
        logger.info(f"Prediction stats: {pred_stats}")
        
        # If predictions are all zeros or very similar, flag it
        if pred_stats['std'] < 1e-6:
            logger.warning("WARNING: Model predictions show very little variation!")
        
        # Calculate risk score with more variation
        spatial_mean = np.mean(predictions, axis=(1, 2))
        risk_score = float(spatial_mean[0, 0])
        
        # Use input features to influence risk assessment
        features_mean = np.mean(model_input, axis=(1, 2))
        elevation = features_mean[0, 0]  # DEM
        slope = features_mean[0, 1]      # Slope
        rainfall = features_mean[0, 2]   # Rainfall
        soil_moisture = features_mean[0, 3]  # Soil moisture
        water_occurrence = features_mean[0, 6]  # Water occurrence

        # Adjust risk score based on physical factors
        risk_factors = {
            'elevation': 1.0 - min(elevation / 1000, 1.0),  # Lower elevation -> higher risk
            'slope': 1.0 - min(slope / 45, 1.0),           # Lower slope -> higher risk
            'rainfall': min(rainfall / 100, 1.0),          # More rain -> higher risk
            'soil_moisture': min(soil_moisture / 0.5, 1.0), # More moisture -> higher risk
            'water_occurrence': min(water_occurrence / 100, 1.0)  # More water -> higher risk
        }
        
        logger.info(f"Risk factors: {risk_factors}")
        
        # Combine factors with original risk score
        adjusted_risk = (
            risk_score * 0.4 +
            risk_factors['elevation'] * 0.15 +
            risk_factors['slope'] * 0.1 +
            risk_factors['rainfall'] * 0.1 +
            risk_factors['soil_moisture'] * 0.125 +
            risk_factors['water_occurrence'] * 0.125
        )
        
        # Ensure risk is in [0, 1]
        adjusted_risk = np.clip(adjusted_risk, 0, 1)
        
        # Calculate risk percentages with more variation
        risk_percentages = {
            'no_risk': max(0, min(100, (1 - adjusted_risk) * 100)),
            'low_risk': max(0, min(100, (1 - abs(adjusted_risk - 0.25)) * 80)),
            'medium_risk': max(0, min(100, (1 - abs(adjusted_risk - 0.5)) * 60)),
            'high_risk': max(0, min(100, (1 - abs(adjusted_risk - 0.75)) * 40)),
            'very_high_risk': max(0, min(100, adjusted_risk * 100))
        }
        
        # Normalize to sum to 100%
        total = sum(risk_percentages.values())
        if total > 0:
            risk_percentages = {k: (v / total) * 100 for k, v in risk_percentages.items()}
        
        logger.info(f"Final risk percentages: {risk_percentages}")
        return risk_percentages
        
    except Exception as e:
        logger.error(f"Error in predict_flood_risk: {e}")
        logger.error(traceback.format_exc())
        return None
# Create visualization based on the risk data
def create_visualization(risk_data, location_info):
    try:
        # Set style to a default one that's guaranteed to work
        plt.style.use('default')
        
        # Create figure with white background
        fig = plt.figure(figsize=(10, 8), dpi=100, facecolor='white')
        ax = fig.add_subplot(111)
        
        # Set up data
        risk_levels = ['No Risk', 'Low', 'Medium', 'High', 'Very High']
        risk_values = [
            risk_data['no_risk'],
            risk_data['low_risk'],
            risk_data['medium_risk'],
            risk_data['high_risk'],
            risk_data['very_high_risk']
        ]
        
        # Define colors for risk levels
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b']
        
        # Create bar chart
        bars = ax.bar(risk_levels, risk_values, color=colors)
        
        # Customize plot
        ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        
        # Format title with word wrapping
        title = f'Flood Risk Assessment\n{location_info["address"]}'
        title = '\n'.join(textwrap.wrap(title, width=50))
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
        
        # Customize axes
        ax.set_ylim(0, max(risk_values) + 10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add timestamp and source info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        plt.figtext(0.02, 0.02, f'Generated: {timestamp}',
                   fontsize=8, color='gray')
        plt.figtext(0.98, 0.02, 'Source: ML Flood Risk Model',
                   fontsize=8, color='gray', ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', 
                   bbox_inches='tight', 
                   facecolor='white',
                   edgecolor='none')
        buf.seek(0)
        
        # Clean up
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None
# Cache helper functions
def get_cache_key(lat, lon):
    return f"flood_risk_{lat:.4f}_{lon:.4f}"

def get_from_cache(key):
    with cache_lock:
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        if os.path.exists(cache_path):
            # Check if cache has expired
            if key in cache_expiry and time.time() > cache_expiry[key]:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
                return None
            
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

def save_to_cache(key, data):
    with cache_lock:
        cache_path = os.path.join(CACHE_DIR, f"{key}.json")
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            # Set expiry time
            cache_expiry[key] = time.time() + CACHE_EXPIRY
            return True
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
            return False

# Download and cache GEE visualization
def cache_visualization(url, key):
    viz_path = os.path.join(CACHE_DIR, f"{key}_viz.png")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(viz_path, 'wb') as f:
                f.write(response.content)
            return viz_path
        return None
    except Exception as e:
        logger.error(f"Error caching visualization: {e}")
        return None

# API Routes
@app.route('/api/search-location', methods=['GET'])
def search_location():
    query = request.args.get('q', '')
    if not query or len(query) < 3:
        return jsonify([])

    try:
        locations = geocoder.geocode(query, exactly_one=False, limit=5)
        if not locations:
            return jsonify([])
        
        results = []
        for location in locations if isinstance(locations, list) else [locations]:
            results.append({
                'name': location.address,
                'lat': location.latitude,
                'lon': location.longitude
            })
        return jsonify(results)
    except Exception as e:
        logger.error(f"Error in location search: {e}")
        return jsonify([])


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Get location information
        location = data.get('location', '')
        lat = data.get('lat')
        lon = data.get('lon')
        
        if lat is not None and lon is not None:
            location_info = reverse_geocode(lat, lon)
        elif location:
            location_info = geocode_location(location)
        else:
            return jsonify({
                'success': False,
                'message': 'Location information required'
            })
            
        if not location_info:
            return jsonify({
                'success': False,
                'message': 'Could not find location'
            })
            
        # Get Earth Engine data
        bbox = create_bbox(location_info['lat'], location_info['lon'])
        ee_data = fetch_ee_data(bbox)
        
        if not ee_data:
            return jsonify({
                'success': False,
                'message': 'Could not fetch Earth Engine data'
            })
            
        # Prepare model input
        model_input = prepare_model_input(ee_data)
        if model_input is None:
            return jsonify({
                'success': False,
                'message': 'Error preparing model input'
            })
            
        # Make prediction
        risk_data = predict_flood_risk(model_input)
        if risk_data is None:
            return jsonify({
                'success': False,
                'message': 'Error running prediction model'
            })
            
        # Create visualization
        viz_buffer = create_visualization(risk_data, location_info)
        viz_url = None
        
        if viz_buffer:
            # Save visualization
            viz_key = get_cache_key(location_info['lat'], location_info['lon'])
            viz_path = os.path.join(CACHE_DIR, f"{viz_key}_viz.png")
            try:
                with open(viz_path, 'wb') as f:
                    f.write(viz_buffer.getvalue())
                viz_url = f"/api/visualization/{viz_key}_viz.png"
            except Exception as e:
                logger.error(f"Error saving visualization: {e}")
                # Continue without visualization if save fails
        
        # Return successful response with detailed information
        response = {
            'success': True,
            'location': location_info,
            'risk_data': risk_data,
            'visualization_url': viz_url,
            'details': {
                'features_processed': True,
                'model_prediction_shape': list(model_input.shape),
                'risk_percentages': risk_data,
                'visualization_generated': viz_buffer is not None
            }
        }
        
        # Add CORS headers
        response_obj = jsonify(response)
        response_obj.headers.add('Access-Control-Allow-Origin', '*')
        return response_obj
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_type': type(e).__name__
        }), 500
@app.route('/api/visualization/<path:path>', methods=['GET'])
def visualization(path):
    try:
        # Secure the path to prevent path traversal
        if '..' in path or path.startswith('/'):
            return jsonify({
                'success': False,
                'message': 'Invalid path'
            }), 400
        
        full_path = os.path.join(CACHE_DIR, path)
        if not os.path.exists(full_path):
            return jsonify({
                'success': False,
                'message': 'Visualization not found'
            }), 404
        
        # Add CORS headers and content type
        response = send_from_directory(CACHE_DIR, path)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Content-Type', 'image/png')
        return response
        
    except Exception as e:
        logger.error(f"Error serving visualization: {e}")
        return jsonify({
            'success': False,
            'message': f'Error serving visualization: {str(e)}'
        }), 500

@app.route('/api/placeholder/<int:width>/<int:height>', methods=['GET'])
def placeholder(width, height):
    placeholder_path = os.path.join(CACHE_DIR, 'placeholder.png')
    if not os.path.exists(placeholder_path):
        img = Image.new('RGB', (width, height), color=(200, 200, 200))
        draw = ImageDraw.Draw(img)
        draw.text((width // 2 - 60, height // 2), "Flood Risk Visualization\nWill appear here", fill=(0, 0, 0))
        img.save(placeholder_path, format='PNG')
    return send_from_directory(CACHE_DIR, 'placeholder.png')

# Serve frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join('templates', path)):
        return send_from_directory('templates', path)
    return send_from_directory('templates', 'index.html')

# Periodically clean expired cache
def clean_expired_cache():
    while True:
        try:
            with cache_lock:
                current_time = time.time()
                expired_keys = [key for key, expiry_time in cache_expiry.items() if current_time > expiry_time]
                
                for key in expired_keys:
                    cache_path = os.path.join(CACHE_DIR, f"{key}.json")
                    viz_path = os.path.join(CACHE_DIR, f"{key}_viz.png")
                    
                    try:
                        if os.path.exists(cache_path):
                            os.remove(cache_path)
                        if os.path.exists(viz_path):
                            os.remove(viz_path)
                        del cache_expiry[key]
                    except Exception as e:
                        logger.error(f"Error cleaning cache for {key}: {e}")
                
                logger.info(f"Cleaned {len(expired_keys)} expired cache items")
        except Exception as e:
            logger.error(f"Error in cache cleaning thread: {e}")
        
        # Sleep for 1 hour
        time.sleep(3600)

# Start cache cleaning thread
cache_cleaner = threading.Thread(target=clean_expired_cache, daemon=True)
cache_cleaner.start()

# Run the application
# Run the application
if __name__ == '__main__':
    # Make sure the model is loaded
    if model is None:
        load_ml_model()
    
    # Make sure Earth Engine is initialized
    if not ee_initialized:
        initialize_ee()
    
    # Start cache cleaning thread - moved here to avoid duplicate threads
    if not any(t.name == "cache_cleaner" for t in threading.enumerate()):
        cache_cleaner = threading.Thread(target=clean_expired_cache, daemon=True, name="cache_cleaner")
        cache_cleaner.start()
    
    # Set debug=False for production to avoid threading issues
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)