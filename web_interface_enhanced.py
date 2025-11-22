from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os
import time
import json
from datetime import datetime
from collections import defaultdict
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('static/calibration', exist_ok=True)
os.makedirs('data/logs', exist_ok=True)

# Global variables
model = None
tracker = None
total_count = []
vehicle_speeds = {}
vehicle_positions = {}
speed_violations = []
traffic_data = []
all_reports = []

# GPS Configuration
gps_location = {
    'latitude': 12.9716,
    'longitude': 77.5946,
    'location_name': 'Click on map to set location',
    'road_name': 'Not set',
    'enabled': False
}

# Speed Configuration
speed_config = {
    'pixels_per_meter': 8.5,
    'fps': 30,
    'speed_limit': 60,
    'enabled': True
}

# YOLO class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def initialize_model(model_name='yolov8n.pt'):
    """Initialize YOLO model and tracker"""
    global model, tracker, total_count, vehicle_speeds, vehicle_positions, speed_violations, traffic_data
    try:
        model = YOLO(model_name)
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        total_count = []
        vehicle_speeds = {}
        vehicle_positions = {}
        speed_violations = []
        traffic_data = []
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def calculate_speed(vehicle_id, current_pos, current_time):
    """Calculate vehicle speed based on position history"""
    global vehicle_positions, vehicle_speeds, speed_config
    
    if not speed_config['enabled']:
        return 0
    
    if vehicle_id not in vehicle_positions:
        vehicle_positions[vehicle_id] = []
    
    vehicle_positions[vehicle_id].append({
        'pos': current_pos,
        'time': current_time
    })
    
    if len(vehicle_positions[vehicle_id]) > 10:
        vehicle_positions[vehicle_id].pop(0)
    
    if len(vehicle_positions[vehicle_id]) < 2:
        return 0
    
    first = vehicle_positions[vehicle_id][0]
    last = vehicle_positions[vehicle_id][-1]
    
    dx = last['pos'][0] - first['pos'][0]
    dy = last['pos'][1] - first['pos'][1]
    distance_pixels = math.sqrt(dx**2 + dy**2)
    
    time_diff = last['time'] - first['time']
    
    if time_diff == 0:
        return 0
    
    distance_meters = distance_pixels / speed_config['pixels_per_meter']
    speed_mps = distance_meters / time_diff
    speed_kmh = speed_mps * 3.6
    
    vehicle_speeds[vehicle_id] = abs(speed_kmh)
    
    return abs(speed_kmh)

def process_frame(img, conf_threshold=0.3, counting_line=[370, 297, 750, 297], frame_number=0):
    """Process a single frame for car detection, counting, and speed estimation"""
    global tracker, total_count, vehicle_speeds, speed_violations, traffic_data, gps_location, speed_config
    
    if model is None or tracker is None:
        return img, len(total_count), {}
    
    current_time = frame_number / speed_config['fps']
    
    results = model(img, stream=True, verbose=False)
    detections = np.empty((0, 5))
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > conf_threshold:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
    
    resultsTracker = tracker.update(detections)
    
    cv2.line(img, (counting_line[0], counting_line[1]), 
             (counting_line[2], counting_line[3]), (0, 0, 255), 3)
    
    frame_stats = {
        'vehicles_detected': len(resultsTracker),
        'avg_speed': 0,
        'speeding_count': 0
    }
    
    total_speed = 0
    speed_count = 0
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        
        cx, cy = x1 + w // 2, y1 + h // 2
        
        speed = calculate_speed(id, (cx, cy), current_time)
        
        if speed > 0:
            total_speed += speed
            speed_count += 1
        
        if speed > speed_config['speed_limit']:
            box_color = (0, 0, 255)
            if id not in [v['id'] for v in speed_violations]:
                speed_violations.append({
                    'id': id,
                    'speed': speed,
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'location': gps_location['location_name'] if gps_location['enabled'] else 'Unknown'
                })
            frame_stats['speeding_count'] += 1
        else:
            box_color = (255, 0, 255)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        
        corner_length = min(w, h) // 5
        cv2.line(img, (x1, y1), (x1 + corner_length, y1), box_color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + corner_length), box_color, 3)
        
        label = f'ID:{id}'
        if speed > 0:
            label += f' | {speed:.1f}km/h'
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(img, (cx, cy), 5, box_color, cv2.FILLED)
        
        if counting_line[0] < cx < counting_line[2] and counting_line[1] - 15 < cy < counting_line[3] + 15:
            if id not in total_count:
                total_count.append(id)
                cv2.line(img, (counting_line[0], counting_line[1]),
                        (counting_line[2], counting_line[3]), (0, 255, 0), 5)
                
                traffic_data.append({
                    'id': id,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'speed': speed if speed > 0 else 'N/A',
                    'latitude': gps_location['latitude'] if gps_location['enabled'] else 'N/A',
                    'longitude': gps_location['longitude'] if gps_location['enabled'] else 'N/A',
                    'location': gps_location['location_name'] if gps_location['enabled'] else 'Unknown'
                })
    
    if speed_count > 0:
        frame_stats['avg_speed'] = total_speed / speed_count
    
    panel_height = 180 if gps_location['enabled'] else 140
    cv2.rectangle(img, (10, 10), (350, panel_height), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (350, panel_height), (255, 255, 255), 2)
    
    y_offset = 35
    cv2.putText(img, 'VEHICLE COUNT', (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, str(len(total_count)), (250, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    y_offset += 30
    if frame_stats['avg_speed'] > 0:
        cv2.putText(img, f"AVG SPEED: {frame_stats['avg_speed']:.1f} km/h", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img, f"SPEEDING: {frame_stats['speeding_count']}", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if frame_stats['speeding_count'] > 0 else (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(img, f"LIMIT: {speed_config['speed_limit']} km/h", (20, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if gps_location['enabled']:
        y_offset += 30
        # Use ASCII-safe version for OpenCV display
        location_display = gps_location['location_name']
        # Remove non-ASCII characters for OpenCV compatibility
        location_ascii = location_display.encode('ascii', 'ignore').decode('ascii')
        if not location_ascii:
            location_ascii = 'Location Set'
        
        cv2.putText(img, f"GPS: {location_ascii[:30]}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_offset += 20
        cv2.putText(img, f"{gps_location['latitude']:.6f}, {gps_location['longitude']:.6f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return img, len(total_count), frame_stats

@app.route('/')
def index():
    """Render main page"""
    return render_template('index_final.html')

@app.route('/analytics')
def analytics():
    """Render analytics dashboard"""
    return render_template('analytics.html')

@app.route('/calibration')
def calibration():
    """Render calibration page"""
    return render_template('calibration.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and processing"""
    global speed_config
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    conf_threshold = float(request.form.get('confidence', 0.3))
    
    speed_config['pixels_per_meter'] = float(request.form.get('pixels_per_meter', 8.5))
    speed_config['speed_limit'] = int(request.form.get('speed_limit', 60))
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
    file.save(filename)
    
    initialize_model()
    
    output_filename = f'output_{int(time.time())}.mp4'
    output_path = os.path.join('static', 'results', output_filename)
    
    os.makedirs(os.path.join('static', 'results'), exist_ok=True)
    
    cap = cv2.VideoCapture(filename)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(cap.get(cv2.CAP_PROP_FPS), 20.0)
    speed_config['fps'] = fps
    
    fourcc_list = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    
    for fourcc_code in fourcc_list:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if out.isOpened():
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, (frame_width, frame_height))
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, count, stats = process_frame(frame, conf_threshold, frame_number=frame_count)
        out.write(processed_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    # Save traffic report
    report_filename = f'traffic_report_{int(time.time())}.json'
    report_path = os.path.join('data', 'logs', report_filename)
    
    report_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_vehicles': len(total_count),
        'speed_violations': len(speed_violations),
        'speed_limit': speed_config['speed_limit'],
        'location': gps_location.copy(),
        'vehicles': traffic_data,
        'violations': speed_violations
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    return jsonify({
        'success': True,
        'output_video': f'/static/results/{output_filename}',
        'total_count': len(total_count),
        'frames_processed': frame_count,
        'speed_violations': len(speed_violations),
        'report_file': report_filename
    })

@app.route('/set_gps_from_map', methods=['POST'])
def set_gps_from_map():
    """Set GPS location from map click"""
    global gps_location
    
    try:
        data = request.json
        lat = float(data.get('latitude'))
        lng = float(data.get('longitude'))
        location_name = data.get('location_name', f'Location_{int(time.time())}')
        
        gps_location['latitude'] = lat
        gps_location['longitude'] = lng
        gps_location['location_name'] = location_name
        gps_location['road_name'] = data.get('road_name', 'Road')
        gps_location['enabled'] = True
        
        return jsonify({'success': True, 'gps': gps_location})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/upload_calibration_frame', methods=['POST'])
def upload_calibration_frame():
    """Upload video frame for calibration"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    
    file = request.files['video']
    video_path = os.path.join('uploads', 'calibration_video.mp4')
    file.save(video_path)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame_path = os.path.join('static', 'calibration', 'frame.jpg')
        cv2.imwrite(frame_path, frame)
        return jsonify({'success': True, 'frame_path': '/static/calibration/frame.jpg'})
    
    return jsonify({'error': 'Could not read video'}), 400

@app.route('/calculate_calibration', methods=['POST'])
def calculate_calibration():
    """Calculate pixels per meter from calibration data"""
    try:
        data = request.json
        pixels = float(data.get('pixels'))
        meters = float(data.get('meters'))
        
        pixels_per_meter = pixels / meters
        
        return jsonify({
            'success': True,
            'pixels_per_meter': round(pixels_per_meter, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_analytics_data')
def get_analytics_data():
    """Get all analytics data"""
    global all_reports
    
    # Load all reports from logs directory
    reports_dir = 'data/logs'
    all_reports = []
    
    if os.path.exists(reports_dir):
        for filename in os.listdir(reports_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(reports_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        report['filename'] = filename
                        all_reports.append(report)
                except:
                    continue
    
    # Calculate aggregate statistics
    total_vehicles = sum(r.get('total_vehicles', 0) for r in all_reports)
    total_violations = sum(r.get('speed_violations', 0) for r in all_reports)
    
    all_speeds = []
    for report in all_reports:
        for vehicle in report.get('vehicles', []):
            speed = vehicle.get('speed')
            if isinstance(speed, (int, float)):
                all_speeds.append(speed)
    
    avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    
    # Speed distribution
    speed_ranges = {
        '0-40': sum(1 for s in all_speeds if 0 <= s <= 40),
        '41-60': sum(1 for s in all_speeds if 40 < s <= 60),
        '61-80': sum(1 for s in all_speeds if 60 < s <= 80),
        '81-100': sum(1 for s in all_speeds if 80 < s <= 100),
        '100+': sum(1 for s in all_speeds if s > 100)
    }
    
    # Location stats
    location_stats = {}
    for report in all_reports:
        loc = report.get('location', {})
        if isinstance(loc, dict):
            loc_name = loc.get('location_name', 'Unknown')
            if loc_name not in location_stats:
                location_stats[loc_name] = {
                    'vehicles': 0,
                    'violations': 0
                }
            location_stats[loc_name]['vehicles'] += report.get('total_vehicles', 0)
            location_stats[loc_name]['violations'] += report.get('speed_violations', 0)
    
    # Top violations
    all_violations = []
    for report in all_reports:
        all_violations.extend(report.get('violations', []))
    
    top_violations = sorted(all_violations, key=lambda x: x.get('speed', 0), reverse=True)[:10]
    
    return jsonify({
        'success': True,
        'total_reports': len(all_reports),
        'total_vehicles': total_vehicles,
        'total_violations': total_violations,
        'avg_speed': round(avg_speed, 1),
        'speed_distribution': speed_ranges,
        'location_stats': location_stats,
        'top_violations': top_violations,
        'reports': all_reports
    })

@app.route('/get_gps')
def get_gps():
    """Get current GPS location"""
    return jsonify(gps_location)

@app.route('/get_stats')
def get_stats():
    """Get current statistics"""
    avg_speed = sum(vehicle_speeds.values()) / len(vehicle_speeds) if vehicle_speeds else 0
    
    return jsonify({
        'total_count': len(total_count),
        'speed_violations': len(speed_violations),
        'avg_speed': round(avg_speed, 1),
        'active_vehicles': len(vehicle_speeds)
    })

@app.route('/download_report/<filename>')
def download_report(filename):
    """Download traffic report"""
    filepath = os.path.join('data', 'logs', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/reset_count', methods=['POST'])
def reset_count():
    """Reset the counter"""
    initialize_model()
    return jsonify({'success': True})

if __name__ == '__main__':
    initialize_model()
    print("🚗 Smart Traffic Monitoring System")
    print("=" * 60)
    print("📍 Main Interface: http://localhost:5000")
    print("📊 Analytics: http://localhost:5000/analytics")
    print("🎯 Calibration: http://localhost:5000/calibration")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)