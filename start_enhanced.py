#!/usr/bin/env python3
"""
Smart Traffic Monitoring System - Complete System
With Clickable Map, Integrated Calibration & Analytics
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def print_header():
    print("\n" + "="*70)
    print("  🚗 SMART TRAFFIC MONITORING SYSTEM - COMPLETE")
    print("  With Map Selection, Calibration & Analytics")
    print("="*70 + "\n")

def check_files():
    print("📁 Checking required files...")
    
    required = {
        'sort.py': 'SORT tracking algorithm',
        'web_interface_enhanced.py': 'Backend server',
        'templates/index_final.html': 'Main interface',
        'templates/calibration.html': 'Calibration tool',
        'templates/analytics.html': 'Analytics dashboard'
    }
    
    missing = []
    for file, desc in required.items():
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - {desc}")
            missing.append(file)
    
    if missing:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        return False
    
    print()
    return True

def create_directories():
    print("📂 Setting up directories...")
    dirs = ['uploads', 'static/results', 'static/calibration', 'templates', 'data/logs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ {d}")
    print()

def open_browser():
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def main():
    print_header()
    
    if not check_files():
        print("\n💡 Make sure all required files are in place!")
        print("\nRequired files:")
        print("  - web_interface_enhanced.py")
        print("  - sort.py")
        print("  - templates/index_final.html")
        print("  - templates/calibration.html")
        print("  - templates/analytics.html")
        sys.exit(1)
    
    create_directories()
    
    print("🚀 Starting Complete System...")
    print("="*70)
    print("📍 Main Interface:    http://localhost:5000")
    print("🎯 Calibration Tool:  http://localhost:5000/calibration")
    print("📊 Analytics:         http://localhost:5000/analytics")
    print("="*70)
    print("\n✨ NEW FEATURES:")
    print("  • 🗺️  Click on map to set GPS location")
    print("  • 🎯 Integrated calibration tool")
    print("  • 📊 Comprehensive analytics dashboard")
    print("  • 📈 Speed distribution charts")
    print("  • 📍 Location-wise analysis")
    print("\nPress Ctrl+C to stop")
    print("="*70 + "\n")
    
    Timer(1.5, open_browser).start()
    
    try:
        from web_interface_enhanced import app
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
        print("📊 Reports saved in: data/logs/")
        print("🎬 Videos saved in: static/results/\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

if __name__ == '__main__':
    main()