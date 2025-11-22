# 🚗 Smart Traffic Monitoring System

**AI-Powered Vehicle Detection with Speed Estimation, GPS Tracking & Advanced Analytics**

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 🌟 Features

### Core Detection
- **🎯 Vehicle Detection**: Cars, trucks, buses, motorcycles using YOLOv8
- **📊 Vehicle Counting**: Accurate counting with SORT tracking algorithm
- **⚡ Speed Estimation**: Real-time speed calculation in km/h
- **🚨 Violation Detection**: Automatic flagging of speeding vehicles
- **🔄 No Duplicates**: Advanced tracking prevents duplicate counting

### GPS & Location
- **🗺️ Click-on-Map Selection**: Simply click on map to set location (no manual entry!)
- **📍 Drag & Adjust**: Fine-tune marker position by dragging
- **🌍 Auto Location Name**: Reverse geocoding automatically gets address
- **🌐 Multi-Language Support**: Handles Kannada, Hindi, Tamil, and all Unicode addresses
- **📝 Dual Storage**: Full address for reports, simplified for video overlay

### Speed & Calibration
- **🎯 Integrated Calibration Tool**: Visual calibration interface
- **📐 Click-to-Measure**: Click 2 points on known distance
- **🔢 Auto Calculation**: Automatic pixels-per-meter calculation
- **📋 One-Click Copy**: Copy calibration value to clipboard
- **💡 Built-in Tips**: Reference distances for common objects

### Analytics Dashboard
- **📊 Overview Statistics**: 5 key metrics at a glance
- **📈 Speed Distribution Chart**: Bar chart showing speed ranges
- **📍 Location Comparison**: Compare traffic across different locations
- **⚠️ Top Violations Table**: List worst speed offenders
- **📄 Reports History**: All processing sessions in one table
- **🔄 Auto Refresh**: Updates every 30 seconds

### Data Export
- **🎬 Processed Videos**: Download with speed overlays
- **📄 JSON Reports**: Detailed data with all vehicle information
- **🌐 UTF-8 Support**: Proper encoding for all languages
- **📊 Multi-Report Analysis**: Compare data across sessions

---
### 🎥 Demo

Here’s a quick look at the system in action 👇

![Demo](demo.gif)


## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Good quality video files

### Quick Setup

1. **Install Flask** (only new dependency needed)
   ```bash
   pip install Flask
   ```

2. **Create Project Structure**
   ```bash
   mkdir templates
   # Place HTML files in templates folder
   ```

3. **Verify Files**
   ```
   Your-Project/
   ├── web_interface_enhanced.py
   ├── start_enhanced.py
   ├── sort.py
   ├── yolov8n.pt
   └── templates/
       ├── index_final.html
       ├── calibration.html
       └── analytics.html
   ```

4. **Run the System**
   ```bash
   python start_enhanced.py
   ```

5. **Access**
   - Main Interface: http://localhost:5000
   - Calibration: http://localhost:5000/calibration
   - Analytics: http://localhost:5000/analytics

---

## 🚀 Quick Start Guide

### Step 1: Set GPS Location (30 seconds)

1. Open main interface: http://localhost:5000
2. Look at the map on the right side
3. **Click anywhere on the map** where you want to monitor
4. Marker appears automatically
5. Status changes to "Active" (green)
6. Location name appears below map

**Pro Tip:** Drag the marker to fine-tune the exact position!

---

### Step 2: Calibrate Speed (2 minutes)

1. Go to: http://localhost:5000/calibration
2. Upload your video file
3. First frame displays automatically
4. **Click 2 points** on a known distance (e.g., lane marking)
5. Enter real distance in meters (e.g., 3.5 for lane width)
6. Click "Calculate"
7. **Copy the result** (e.g., 8.7)

**Common Reference Distances:**
- Lane Width: 3.5 meters
- Road Marking: 3.0 meters
- Car Length: 4.5 meters
- Truck Length: 8.0 meters

---

### Step 3: Process Video (depends on video length)

1. Go back to main interface: http://localhost:5000
2. Upload your video (drag & drop or click)
3. Settings:
   - **Pixels per Meter**: Paste your calibration value
   - **Speed Limit**: Set appropriate limit (e.g., 60 km/h)
   - **Confidence**: Leave at 0.30 (adjust if needed)
4. Click **"🚀 Process Video"**
5. Wait for processing (progress shown)
6. Download processed video and JSON report

---

### Step 4: View Analytics (anytime)

1. Go to: http://localhost:5000/analytics
2. See comprehensive statistics:
   - Total vehicles across all sessions
   - Speed violations count
   - Average speeds
   - Speed distribution chart
   - Location comparison
   - Top 10 worst offenders

**Auto Updates:** Dashboard refreshes every 30 seconds!

---

## 🎯 Key Features Explained

### 1. Click-on-Map GPS Selection

✅ Click on map → Done!
   - Marker appears
   - Location detected automatically
   - GPS coordinates saved
   - Status: Active
```

**How it works:**
- Uses OpenStreetMap and Leaflet.js
- Reverse geocoding via Nominatim API
- Drag marker to adjust position
- Works with any location worldwide

---

### 2. Integrated Calibration Tool

**Purpose:** Calculate accurate pixels-per-meter for speed estimation

**Process:**
```
Video Frame → Click 2 Points → Enter Distance → Get Value
```

**Example:**
```
1. Lane marking visible: 3 meters long
2. Click start of marking
3. Click end of marking
4. System measures: 25.5 pixels
5. Enter: 3 meters
6. Result: 8.5 pixels per meter
```

**Why it matters:**
- Accurate speed calculations
- Adapts to any camera angle
- Works for any video resolution
- One-time calibration per camera position

---

### 3. Speed Estimation System

**How it works:**
```
1. Track vehicle position across frames
2. Calculate distance traveled (in pixels)
3. Convert to meters using calibration
4. Calculate speed: distance/time × 3.6 = km/h
```

**Features:**
- Real-time speed display on video
- Color coding: Purple = normal, Red = speeding
- Historical speed tracking
- Average speed calculation
- Violation detection

**Accuracy factors:**
- Proper calibration (most important!)
- Stable camera position
- Good video quality
- Correct FPS value

---

### 4. Analytics Dashboard

**What you see:**

**Overview Cards:**
```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ 📁 Reports   │ 🚗 Vehicles  │ ⚠️ Violations│ ⚡ Avg Speed │
│     15       │     453      │      89      │   58.3 km/h  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Charts:**
1. **Speed Distribution**: Shows how many vehicles in each speed range
2. **Location Comparison**: Compare traffic at different locations

**Tables:**
1. **Top Violations**: 10 worst speed offenders
2. **All Reports**: Complete history with dates and locations

**Use cases:**
- Identify problem areas
- Track trends over time
- Generate reports for authorities
- Compare different locations
- Find peak violation times

---

## 📊 Understanding the Display

### Video Overlay

**Information Panel (Top-Left):**
```
┌─────────────────────────────┐
│ VEHICLE COUNT: 23           │  ← Total crossed line
│ AVG SPEED: 55.3 km/h        │  ← Average of tracked vehicles
│ SPEEDING: 3                 │  ← Vehicles over limit
│ LIMIT: 60 km/h              │  ← Your speed limit
│ GPS: Banashankari, Bengaluru│  ← Location (English only)
│ 12.969792, 77.575404        │  ← Coordinates
└─────────────────────────────┘
```

**Vehicle Labels:**
```
┌─────────────────────────┐
│ ID:23 | 65.3 km/h       │  ← Vehicle ID and speed
└─────────────────────────┘
```

**Color Coding:**
- 🟣 **Purple Box** = Normal speed (within limit)
- 🔴 **Red Box** = Speeding (over limit)
- 🔵 **Red Line** = Counting line
- 🟢 **Green Flash** = Vehicle counted (line flashes green)

---

### Statistics Dashboard

**Four Key Metrics:**

1. **Total Vehicles** 🚗
   - Vehicles that crossed the counting line
   - Final count for the session
   - This is your main result

2. **Speed Violations** ⚠️
   - Vehicles exceeding speed limit
   - Counted only once per vehicle
   - Used for violation rate calculation

3. **Average Speed** ⚡
   - Mean speed of all tracked vehicles
   - Only vehicles with valid speed readings
   - Useful for traffic flow analysis

4. **Vehicles Tracked** 🔄
   - Currently tracked vehicles
   - Changes as vehicles enter/leave frame
   - Usually 0 at end of video

---

## ⚙️ Configuration Guide

### Speed Settings

**Confidence Threshold (0.1 - 0.9):**
- **0.15-0.25**: More detections, more false positives
- **0.25-0.35**: ✅ **Recommended** - Best balance
- **0.35-0.50**: Fewer false positives, may miss vehicles

**Speed Limit (km/h):**
- School Zone: 20-30 km/h
- Residential: 30-40 km/h
- City Road: 50-60 km/h
- Highway: 80-100 km/h
- Expressway: 100-120 km/h

**Pixels per Meter:**
- Must be calibrated for each camera position!
- Typical range: 5-20
- Highway (far camera): 15-25
- City street (close): 5-10

---

### SORT Tracker Parameters

Located in `web_interface_enhanced.py`:
```python
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
```

**Parameters:**
- `max_age=20`: Keep tracking for 20 frames without detection
- `min_hits=3`: Need 3 detections before counting
- `iou_threshold=0.3`: Overlap threshold for matching

**When to adjust:**
- Heavy traffic → Increase `max_age` to 30
- Fast vehicles → Decrease `min_hits` to 2
- Many false detections → Increase `min_hits` to 4

---

## 📁 Project Structure

```
Smart-Traffic-Monitoring/
│
├── 📄 README.md                    ← This file
├── 📄 web_interface_enhanced.py    ← Backend server
├── 📄 start_enhanced.py            ← Launcher script
├── 📄 sort.py                      ← SORT tracking algorithm
├── 🗂️ yolov8n.pt                   ← YOLO model weights
│
├── 📁 templates/                   ← HTML templates
│   ├── index_final.html            ← Main interface
│   ├── calibration.html            ← Calibration tool
│   └── analytics.html              ← Analytics dashboard
│
├── 📁 uploads/                     ← Uploaded videos (temp)
├── 📁 static/
│   ├── results/                    ← Processed videos
│   └── calibration/                ← Calibration frames
│
└── 📁 data/
    └── logs/                       ← JSON reports
```

**Auto-created folders:**
- `uploads/` - Temporary storage for uploaded videos
- `static/results/` - Processed videos with overlays
- `static/calibration/` - Extracted frames for calibration
- `data/logs/` - JSON reports for each session

---

## 📊 Performance

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| YOLOv8n | 6MB | ~45ms | ~1.2ms | Good |
| YOLOv8s | 22MB | ~65ms | ~1.4ms | Better |
| YOLOv8m | 50MB | ~95ms | ~2.1ms | Great |
| YOLOv8l | 84MB | ~120ms | ~2.8ms | Excellent |

## 📄 Report Format

### JSON Report Structure

```json
{
  "timestamp": "2025-01-15 14:30:22",
  "total_vehicles": 45,
  "speed_violations": 8,
  "speed_limit": 60,
  "location": {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "location_name": "Banashankari, ಬೆಂಗಳೂರು ಪಟ್ಟಣದ ನಗರ ನಿಗಮ, Bengaluru",
    "location_display": "Banashankari, Bengaluru",
    "road_name": "Road",
    "enabled": true
  },
  "vehicles": [
    {
      "id": 1,
      "timestamp": "2025-01-15 14:30:25",
      "speed": 55.3,
      "latitude": 12.9716,
      "longitude": 77.5946,
      "location": "Banashankari, Bengaluru"
    }
  ],
  "violations": [
    {
      "id": 23,
      "speed": 78.5,
      "time": "2025-01-15 14:30:45",
      "location": "Banashankari, Bengaluru"
    }
  ]
}
```

**Key Fields:**
- `location_name`: Full address (all languages)
- `location_display`: Simplified for video (English only)
- `vehicles`: All vehicles that crossed the line
- `violations`: Only vehicles exceeding speed limit

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Map Not Loading
**Symptoms:** Blank or gray map area

**Solutions:**
```
✓ Check internet connection (map needs internet)
✓ Wait 5 seconds for map to load
✓ Refresh page (F5)
✓ Try different browser (Chrome recommended)
✓ Disable ad-blockers temporarily
```

---

#### 2. GPS Not Setting
**Symptoms:** Status stays "Click on Map"

**Solutions:**
```
✓ Make sure you CLICK (not hover) on map
✓ Look for marker appearing
✓ Try clicking again
✓ Check browser console (F12) for errors
✓ Verify internet connection (needs geocoding)
```

---

#### 3. Video Showing "????????"
**Symptoms:** Location shows as question marks in video

**Solution:** ✅ **Already Fixed!**
- System now automatically extracts English parts
- Video shows simplified English-only location
- Full address still available in reports and web interface

---

#### 4. Speed Showing 0 or Wrong
**Symptoms:** All speeds are 0 or wildly inaccurate

**Solutions:**
```
✓ Recalibrate pixels_per_meter (most common issue!)
✓ Use calibration tool: /calibration
✓ Ensure camera is stable (not moving)
✓ Check video quality (HD recommended)
✓ Verify FPS value is correct
```

**Verification:**
```
If speed seems off:
1. Process short test clip
2. Compare displayed speeds with known speeds
3. Adjust pixels_per_meter value
4. Reprocess and verify
```

---

#### 5. Analytics Chart Labels Overlapping
**Symptoms:** Unreadable text on location chart

**Solution:** ✅ **Already Fixed!**
- Labels now rotated 45 degrees
- Reduced font size
- Hover shows full address
- Table cells wrap text properly

---

#### 6. Video Not Playing in Browser
**Symptoms:** Processed video doesn't play

**Solutions:**
```
✓ Try different browser (Chrome best)
✓ Use download button and play locally
✓ Check if file exists in static/results/
✓ Verify file size is not 0
✓ Try VLC player for downloaded file
```

---

#### 7. Port Already in Use
**Symptoms:** Error: "Address already in use"

**Solution:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

---

#### 8. Calibration Canvas Blank
**Symptoms:** No image shows after uploading video

**Solutions:**
```
✓ Use valid video format (MP4, AVI, MOV)
✓ Try shorter video (under 1 minute)
✓ Check file size (under 100MB)
✓ Verify video is not corrupted
✓ Try different video
```

---

## 💡 Pro Tips

### For Best Results

**Video Quality:**
```
✓ HD resolution (720p or 1080p)
✓ Good lighting (daylight or proper artificial)
✓ Stable camera (tripod or fixed mount)
✓ Clear view (no obstacles blocking vehicles)
✓ Appropriate angle (30-45° from perpendicular)
```

**Speed Accuracy:**
```
✓ Calibrate carefully using calibration tool
✓ Test with known speeds first
✓ Recalibrate if camera position changes
✓ Use clear reference distances
✓ Verify results look reasonable
```

**GPS Setup:**
```
✓ Click directly on monitoring spot
✓ Use satellite view for accuracy
✓ Drag marker to fine-tune
✓ Verify location name is correct
✓ Coordinates will auto-update
```

**Processing:**
```
✓ Start with short test clip (30 seconds)
✓ Verify settings before long video
✓ Check if speeds look reasonable
✓ Adjust confidence if needed
✓ Monitor progress during processing
```

---

## 🌍 Multi-Language Support

### Supported Languages

The system properly handles addresses in:
- ✅ English
- ✅ ಕನ್ನಡ (Kannada)
- ✅ हिन्दी (Hindi)
- ✅ தமிழ் (Tamil)
- ✅ తెలుగు (Telugu)
- ✅ മലയാളം (Malayalam)
- ✅ ગુજરાતી (Gujarati)
- ✅ मराठी (Marathi)
- ✅ বাংলা (Bengali)
- ✅ ਪੰਜਾਬੀ (Punjabi)
- ✅ All Unicode characters

### How it Works

**Dual Location Storage:**
1. **Full Address** - Used in:
   - Web interface display
   - JSON reports
   - Analytics dashboard
   - All visible text

2. **Simplified Address** - Used in:
   - Video overlay (OpenCV limitation)
   - Automatically extracts English parts
   - Example: "Banashankari, Bengaluru"

**Why Two Versions?**
- OpenCV can't display Unicode characters
- Video needs readable overlay
- Reports need full accurate data
- Best of both worlds!

---

## 📊 Use Cases

### 1. Traffic Studies
```
Purpose: Analyze traffic patterns
Data: Speed distribution, peak hours
Duration: 1 week to 1 month
Output: Comprehensive reports with charts
```

### 2. Speed Enforcement Support
```
Purpose: Document violations
Data: Speed, time, location, vehicle ID
Duration: Continuous monitoring
Output: Violation reports with GPS tags
```

### 3. Road Safety Assessment
```
Purpose: Evaluate road design
Data: Average speeds, violation rates
Duration: Before/after comparisons
Output: Safety recommendations
```

### 4. School Zone Monitoring
```
Purpose: Ensure student safety
Data: Speeds during school hours
Duration: Daily during school days
Output: Compliance reports
```

### 5. Highway Monitoring
```
Purpose: Traffic flow analysis
Data: Volume, speeds, violations
Duration: 24/7 monitoring
Output: Flow optimization recommendations
```

---

## 🎓 Example Workflow

### Scenario: Highway Speed Monitoring

**Day 1: Setup (30 minutes)**
```
1. Set GPS:
   - Open main interface
   - Click on highway location on map
   - Verify: "GPS Active" status
   
2. Calibrate:
   - Go to /calibration
   - Upload test video
   - Click 2 points on lane marking (3.5m)
   - Get result: 9.2 pixels/meter
   - Copy value
   
3. Test:
   - Process 30-second test clip
   - Verify speeds look reasonable
   - Adjust if needed
```

**Week 1: Data Collection (Daily)**
```
1. Upload daily videos
2. Use saved calibration value
3. Set speed limit: 100 km/h
4. Process videos
5. Download reports
```

**Weekend: Analysis**
```
1. Open /analytics
2. View statistics:
   - 543 total vehicles
   - 87 violations (16% rate)
   - Average speed: 96 km/h
   - Peak violations: 2pm-4pm
3. Generate charts:
   - Speed distribution shows most at 90-110 km/h
   - Location comparison shows consistent pattern
4. Top offender: 142 km/h (Vehicle #234)
```

**Decision:**
```
Based on data:
- 16% violation rate is high
- Peak time is 2-4pm
- Recommend increased enforcement
- Consider speed limit review
```

---

## 🔐 Privacy & Ethics

### Data Handling
- No personal data collected
- No facial recognition
- No license plate reading
- GPS location is for reference only
- Reports stored locally only

### Intended Use
- ✅ Traffic studies
- ✅ Road safety assessment
- ✅ Infrastructure planning
- ✅ Research purposes

### Not Intended For
- ❌ Automated enforcement without human review
- ❌ Real-time ticketing
- ❌ Individual tracking
- ❌ Privacy invasion

### Recommendations
- Use for aggregate analysis
- Human review of all violations
- Inform public if deployed
- Follow local regulations
- Respect privacy laws

---

## 🛠️ Technical Specifications

### System Requirements
```
Minimum:
- Python 3.8+
- 4GB RAM
- 2GB free disk space
- CPU: Dual-core 2.0GHz

Recommended:
- Python 3.10+
- 8GB+ RAM
- 10GB+ free disk space
- CPU: Quad-core 3.0GHz+
- GPU: Optional (faster processing)
```

### Supported Video Formats
```
✅ MP4 (H.264, H.265)
✅ AVI
✅ MOV
✅ MKV
✅ Most common formats OpenCV supports
```

### Supported Browsers
```
✅ Chrome (recommended)
✅ Firefox
✅ Safari
✅ Edge
✅ Opera
✅ Mobile browsers
```

### Performance
```
Processing Speed:
- CPU only: 5-10 FPS
- With GPU: 20-30 FPS

Video Length:
- 1 minute video: ~2-5 minutes processing
- 10 minute video: ~20-50 minutes processing
- Depends on resolution and hardware
```

---

## 📚 Additional Resources

### Documentation Files
```
📄 README.md                    - This file
📄 COMPLETE_SYSTEM_GUIDE.md     - Detailed usage guide
📄 LOCATION_DISPLAY_FIX.md      - Multi-language support info
📄 OPENCV_DISPLAY_FIX.md        - Technical fixes explanation
```

### External Resources
```
🔗 YOLOv8: https://docs.ultralytics.com/
🔗 SORT Algorithm: https://github.com/abewley/sort
🔗 OpenCV: https://opencv.org/
🔗 Leaflet.js: https://leafletjs.com/
🔗 Chart.js: https://www.chartjs.org/
```

---

## 🤝 Contributing

### Ways to Contribute
- Report bugs or issues
- Suggest new features
- Improve documentation
- Share your use cases
- Provide feedback

### Feature Requests
If you'd like additional features:
- Direction detection (incoming/outgoing)
- Vehicle type classification display
- Export to Excel/CSV
- Email notifications for violations
- Multiple counting lines
- Time-based filtering

---

## 📝 License

This project uses:
- **YOLOv8**: AGPL-3.0 License
- **SORT**: GPL-3.0 License
- **Flask**: BSD-3-Clause License
- **OpenCV**: Apache 2.0 License

---

## ⚠️ Disclaimer

This software is provided "as is" for research and educational purposes. The accuracy of speed measurements depends on proper calibration and setup. This system is not intended to replace certified speed enforcement equipment. Always verify critical measurements with professional-grade equipment. Consult local regulations before deploying for any official use.

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Start system
python start_enhanced.py

# Access pages
Main:        http://localhost:5000
Calibration: http://localhost:5000/calibration
Analytics:   http://localhost:5000/analytics
```

### Default Settings
```
Confidence: 0.30
Speed Limit: 60 km/h
Pixels/Meter: 8.5 (must calibrate!)
GPS: Bangalore (12.9716, 77.5946)
```

### Keyboard Shortcuts (in terminal)
```
Ctrl+C: Stop server
F5: Refresh page
F12: Open browser console (for debugging)
```

---


## 🚀 What's Next?

Future enhancements could include:
- Real-time webcam support
- Multiple counting lines
- Vehicle type classification
- Direction detection
- CSV/Excel export
- Email alerts
- Database integration
- Mobile app

---

## ✅ Checklist for First-Time Users

```
Setup:
[ ] Python 3.8+ installed
[ ] Flask installed (pip install Flask)
[ ] All files in correct folders
[ ] templates/ folder created
[ ] 3 HTML files in templates/

First Run:
[ ] Run: python start_enhanced.py
[ ] Open: http://localhost:5000
[ ] Click on map to set GPS
[ ] Status shows "Active"
[ ] Go to /calibration
[ ] Upload test video
[ ] Calibrate pixels_per_meter

First Processing:
[ ] Return to main page
[ ] Upload video
[ ] Enter calibration value
[ ] Set speed limit
[ ] Process video
[ ] Download results

Verify:
[ ] Video shows speeds (not "?")
[ ] JSON report created
[ ] Analytics page loads
[ ] Charts display properly
```
**Made with ❤️ for Smart Traffic Management**
