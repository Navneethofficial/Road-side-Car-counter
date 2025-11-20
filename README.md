

## 🚗 Roadside Vehicle Detection and Counting System

This project implements a **real-time vehicle detection and counting system** using the **YOLOv8 (You Only Look Once)** model and the **SORT (Simple Online and Realtime Tracking)** algorithm.
It can process both **live camera feeds** and **recorded videos** to detect, track, and count moving vehicles such as cars, buses, trucks, and motorbikes.

---
### 🎥 Demo

Here’s a quick look at the system in action 👇

![Demo](demo.gif)

---

### ⚙️ Key Features

* ✅ Real-time object detection using **YOLOv8**
* 🚘 Vehicle tracking and ID assignment with **SORT**
* 📊 Accurate counting of vehicles crossing a predefined line
* 🧠 Mask-based region selection for focused detection
* 🖼️ Graphic overlays showing vehicle counts dynamically
* 💾 Video processing and output saving capabilities

---

### 🧰 Technologies Used

* **Python**
* **OpenCV** — for image processing and visualization
* **cvzone** — for beautiful graphics overlays
* **Ultralytics YOLOv8** — for object detection
* **SORT Algorithm (Kalman Filter + IOU)** — for multi-object tracking
* **NumPy**, **Math**, **Matplotlib**, **FilterPy**

---

### 🧠 How It Works

1. The YOLOv8 model detects vehicles in each video frame.
2. The **SORT** algorithm tracks each object across frames using **Kalman filters** and **IOU matching**.
3. When a vehicle crosses the predefined counting line, it increments the total vehicle count.
4. Results are overlaid on the video with bounding boxes, IDs, and the total count display.

---

### 🖥️ Project Structure

```
📁 Roadside_Car_Counter/
│
├── main.py                 # Main detection and counting script
├── sort.py                 # SORT algorithm implementation
├── yolov8n.pt              # YOLOv8 model weights
├── mask.png                # Mask for region of interest
├── graphics.png            # Counter overlay image
├── cars.mp4                # Input video file
└── demo.gif                # Project demo animation
```

---

### ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Roadside_Car_Counter.git
cd Roadside_Car_Counter

# 2. Install dependencies
pip install ultralytics opencv-python cvzone filterpy numpy matplotlib

# 3. Run the project
python main.py
```

---

### 📈 Output Example

When you run the script, you’ll see:

* Detected vehicles with bounding boxes and unique IDs
* A live counter overlay showing the number of cars passing
* (Optional) You can modify the line coordinates to count in any direction

---

### 🧩 Future Improvements

* 🚦 Add speed estimation for tracked vehicles
* 🛰️ Integrate GPS for roadside monitoring
* 📉 Store daily traffic logs in a database
* 💻 Deploy using Streamlit or Flask for live dashboard visualization

---


