# 🌊 FloodSense – AI-Powered Flood Risk Prediction System

FloodSense is an AI-driven web application that predicts flood risk for any location using satellite data, machine learning, and geospatial analysis. It integrates **Google Earth Engine**, a deep learning **U-Net model**, and an interactive **web frontend** to help individuals, communities, and policymakers make informed decisions about flood preparedness.

---

## 🚀 Features

- 🔍 **Location-Based Search:** Enter a location or click on the map to view flood risk.
- 🛰️ **Data Integration:** Utilizes real-time environmental data (elevation, slope, rainfall, soil moisture, etc.) from GEE.
- 📊 **Risk Visualization:** Displays risk levels with bar charts and interactive maps.
- 🗺️ **Interactive Interface:** Built with Leaflet.js for map-based exploration.
- 💡 **Smart Predictions:** Uses deep learning (U-Net) trained on multi-source geospatial data.

---

## 🛠️ Tech Stack

**Machine Learning:**
- Python, TensorFlow, Keras, NumPy, scikit-learn, rasterio

**Backend:**
- Flask, Google Earth Engine API, geopy, Matplotlib

**Frontend:**
- HTML, CSS, JavaScript, Leaflet.js, Font Awesome

**Other Tools:**
- Postman, Jupyter Notebook, Git

---

## 🧠 How It Works

### 1. **Data Collection & Preprocessing**
- **Sources:** GEE datasets (DEM, rainfall, soil moisture, Sentinel imagery, etc.)
- **Tools:** Earth Engine API, rasterio, NumPy
- **Output:** Clipped and preprocessed GeoTIFF files ready for model training

### 2. **Model Development**
- **Architecture:** U-Net with batch normalization & dropout
- **Input Shape:** `(256, 256, 11)` – 11 geospatial features
- **Training:** Binary cross-entropy + Dice loss, data augmentation, model checkpoints
- **Output:** Binary flood risk predictions

### 3. **Backend (API)**
- Flask API with endpoints:
  - `/api/search-location`: Geolocation search
  - `/api/predict`: GEE data fetch + model prediction
  - `/api/visualization`: Risk chart rendering
- Caching results for 24 hours to reduce latency

### 4. **Frontend**
- Map-based interface with location search
- Displays prediction results and risk visualizations
- Responsive design with modern CSS styling

---

## 📈 Results

- ✅ Accurate and consistent predictions validated with visual and quantitative metrics
- 🌐 Real-time risk assessment via API integration with GEE
- 📱 Responsive web interface tested across devices
- ⚡ Caching reduced API latency by ~80% on repeated queries

---

## 📌 Challenges

- ⚠️ Missing historical flood masks → Used water occurrence as proxy
- 🔄 GEE timeouts → Handled with request timeouts and data caching
- 📉 Initial overfitting in model → Mitigated with dropout and batch normalization
- 🖥️ Mobile responsiveness required advanced CSS tweaks

---

## 🔮 Future Enhancements

- 📊 Ensemble models & transfer learning for improved prediction accuracy
- ☁️ Cloud deployment with async processing and load balancing
- 📥 User accounts, saved searches, and alert notifications
- 🔔 Real-time flood alerts via email/SMS
- 🌍 Crowdsourced flood event reporting

---

## 📁 Project Structure

```bash
📁 project-root/
├── 📁 templates/
│   └── index.html                 # Frontend HTML template
├── README.md                      # Project documentation
├── backend1.py                    # Flask backend script
├── flood.ipynb                    # Jupyter notebook for model training/evaluation
├── requirements.txt               # Python dependencies
├── service-account-key.json       # GEE service account credentials (keep this secure!)

```


---


## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Google Earth Engine](https://earthengine.google.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Leaflet.js](https://leafletjs.com/)
- [OpenStreetMap / Nominatim](https://nominatim.openstreetmap.org/)
