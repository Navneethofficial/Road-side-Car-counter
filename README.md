# 🚗 Roadside Car Counter - Web Interface

A modern, AI-powered vehicle detection and counting system with an intuitive web interface. Built with YOLOv8 and SORT tracking algorithm for accurate real-time vehicle counting.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-red.svg)

## 🌟 Features

* **🌐 Modern Web Interface**: Clean, responsive design with drag-and-drop functionality
* **📹 Video Processing**: Upload and process videos for vehicle counting
* **📷 Live Webcam Detection**: Real-time vehicle detection using your webcam
* **🎯 Multi-Vehicle Support**: Detects cars, trucks, buses, and motorbikes
* **📊 Real-time Statistics**: Live count updates and processing metrics
* **⚙️ Adjustable Settings**: Customizable confidence threshold
* **📱 Mobile Friendly**: Works on desktop, tablet, and mobile devices
* **🔄 SORT Tracking**: Advanced object tracking to prevent duplicate counting

### 🎥 Demo

Here’s a quick look at the system in action 👇

![Demo](demo.gif)


## 🚀 Quick Start

### Option 1: Quick Launch (Recommended)

```bash
python start_web.py
```

The browser will automatically open to `http://localhost:5000`

### Option 2: Manual Start

```bash
python web_interface.py
```

Then open your browser to `http://localhost:5000`

## 📦 Installation

### Prerequisites

* Python 3.8 or higher
* Webcam (optional, for live detection)
* 4GB+ RAM recommended

### Step-by-Step Setup

1. **Clone or download the project**
   ```bash
   cd Roadside-Car-Counter
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify files**
   Ensure you have:
   - `sort.py` - SORT tracker implementation
   - `yolov8n.pt` - YOLO model (auto-downloads if missing)
   - `web_interface.py` - Flask backend
   - `templates/index.html` - Web interface

5. **Run the application**
   ```bash
   python start_web.py
   ```

## 📁 Project Structure

```
Roadside-Car-Counter/
├── 📄 start_web.py              # Quick launcher script
├── 🌐 web_interface.py          # Flask web backend
├── 🔍 sort.py                   # SORT tracking algorithm
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
│
├── 📁 templates/
│   └── 🌐 index.html           # Web interface HTML
│
├── 📁 uploads/                  # Uploaded videos (auto-created)
├── 📁 static/
│   └── 📁 results/             # Processed videos (auto-created)
│
├── 🗂️ yolov8n.pt               # YOLO model weights
├── 🎬 cars.mp4                  # Sample video (optional)
├── 🖼️ graphics.png              # UI graphics (optional)
└── 🖼️ mask-950x480.png          # Region mask (optional)
```

## 🎯 How to Use

### Video Upload Mode

1. **Upload Video**
   - Drag and drop a video file to the upload zone, OR
   - Click the upload zone to browse and select a video
   - Supported formats: MP4, AVI, MOV

2. **Adjust Settings**
   - Set confidence threshold (0.1 - 0.9)
   - Lower values detect more objects but may include false positives
   - Recommended: 0.25-0.35 for optimal results

3. **Process Video**
   - Click "Process Video" button
   - Wait for processing to complete
   - View results with vehicle count overlay

4. **Download Results**
   - Processed video plays automatically
   - Right-click video to download

### Live Webcam Mode

1. **Start Webcam**
   - Click "Start Webcam" button
   - Allow browser to access your camera

2. **Monitor Detection**
   - Watch live vehicle detection
   - Real-time count updates
   - FPS metrics displayed

3. **Stop Detection**
   - Click "Stop Webcam" to end session

### Reset Counter

- Click "Reset Counter" button to clear the vehicle count
- Useful when starting a new counting session

## ⚙️ Configuration

### Confidence Threshold

Controls detection sensitivity:

| Threshold | Use Case | Pros | Cons |
|-----------|----------|------|------|
| 0.15-0.25 | Dense traffic, small vehicles | More detections | More false positives |
| 0.25-0.35 | **Recommended** | Balanced | Best overall |
| 0.35-0.50 | Clear conditions, large vehicles | Fewer false positives | May miss some vehicles |

### Counting Line

The counting line is pre-configured in `web_interface.py`:

```python
counting_line = [370, 297, 750, 297]  # [x1, y1, x2, y2]
```

Adjust these coordinates based on your video's perspective.

### SORT Tracker Parameters

In `web_interface.py`:

```python
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
```

- **max_age**: Frames to keep tracking without detection (20)
- **min_hits**: Minimum detections before counting (3)
- **iou_threshold**: Overlap threshold for matching (0.3)

## 🛠️ Advanced Usage

### Custom Model

Replace `yolov8n.pt` with other YOLO models:

- `yolov8s.pt` - Small (better accuracy, slower)
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (best accuracy)

Update in `web_interface.py`:
```python
def initialize_model(model_name='yolov8s.pt'):  # Change here
```

### Add Region Masking

To focus detection on specific road areas:

1. Create a mask image (white = detect, black = ignore)
2. Load mask in `web_interface.py`:
```python
mask = cv2.imread('mask.png')
imgRegion = cv2.bitwise_and(img, mask)
```

### Change Port

To run on a different port:

```python
app.run(debug=False, host='0.0.0.0', port=8080)  # Change 5000 to 8080
```

## 🐛 Troubleshooting

### "Module not found" errors

```bash
pip install -r requirements.txt --force-reinstall
```

### Webcam not working

- Check browser permissions (allow camera access)
- Ensure no other application is using the webcam
- Try a different browser

### Video processing slow

- Use smaller video files
- Reduce video resolution
- Use a faster YOLO model (yolov8n is fastest)
- Close other applications

### Import errors with 'sort'

Make sure `sort.py` is in the same directory as `web_interface.py`

### Port already in use

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

## 📊 Performance

| Model | Size | Speed (CPU) | Speed (GPU) | Accuracy |
|-------|------|-------------|-------------|----------|
| YOLOv8n | 6MB | ~45ms | ~1.2ms | Good |
| YOLOv8s | 22MB | ~65ms | ~1.4ms | Better |
| YOLOv8m | 50MB | ~95ms | ~2.1ms | Great |
| YOLOv8l | 84MB | ~120ms | ~2.8ms | Excellent |

## 🔒 Security Notes

- This is designed for local use
- Don't expose to public internet without authentication
- Uploaded videos are stored temporarily in `uploads/`
- Clear `uploads/` and `static/results/` periodically

### 🧩 Future Improvements

* 🚦 Add speed estimation for tracked vehicles
* 🛰️ Integrate GPS for roadside monitoring
* 📉 Store daily traffic logs in a database
* 💻 Deploy using Streamlit or Flask for live dashboard visualization

---

## 📝 License

This project uses:
- YOLOv8 (AGPL-3.0)
- SORT tracker (GPL-3.0)
- Flask (BSD-3-Clause)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add vehicle type classification display
- [ ] Export count data to CSV/Excel
- [ ] Multi-line counting support
- [ ] Speed estimation
- [ ] Direction tracking
- [ ] Database integration for historical data

## 📧 Support

For issues or questions:

1. Check the Troubleshooting section
2. Ensure all dependencies are installed
3. Verify file structure matches documentation

## 🎓 Acknowledgments

- **Ultralytics** for YOLOv8
- **Alex Bewley** for SORT tracking algorithm
- **Flask** for web framework

---

Made with ❤️ for smart traffic monitoring

**Happy Counting! 🚗💨**

### 🧩 Future Improvements

* 🚦 Add speed estimation for tracked vehicles
* 🛰️ Integrate GPS for roadside monitoring
* 📉 Store daily traffic logs in a database
* 💻 Deploy using Streamlit or Flask for live dashboard visualization

---


