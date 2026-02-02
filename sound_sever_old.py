# Project: Sound Alert Pro
# Author: Brandon Whitehead
# Original Creation Date: September 14, 2025

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy import signal
from pushbullet import Pushbullet
from sound_config import SOUND_NAMES

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

is_monitoring = False
interpreter = None
recent_alerts = []
enabled_sounds = ["Fire alarm", "Smoke detector", "Doorbell", "Baby crying", "Emergency sirens"]
last_notification_sound = ""
last_notification_time = 0
NOTIFICATION_COOLDOWN = 30 

# Pushbullet configuration
PB_API_KEY = "o.v0UboC4HraDHEf6kyYF3TSu25JcAQfWA"

def load_model():
    global interpreter
    interpreter = tf.lite.Interpreter(model_path='yamnet.tflite')
    interpreter.allocate_tensors()
    print("✅ AI Model loaded")

def sound_monitoring_thread():
    global is_monitoring, interpreter, last_notification_sound, last_notification_time
    
    DURATION = 2
    SAMPLE_RATE = 48000
    CONFIDENCE_THRESHOLD = 0.25
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("🎧 Sound monitoring started in background...")
    
    while is_monitoring:
        try:
            # Record audio
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1,
                          dtype='float32',
                          device='hw:1,0')
            sd.wait()
            audio_data = np.squeeze(audio)

            # Resample from 48000 Hz to 16000 Hz
            if SAMPLE_RATE != 16000:
                number_of_samples = int(len(audio_data) * 16000 / SAMPLE_RATE)
                audio_data = signal.resample(audio_data, number_of_samples)

            # Trim to exactly 15600 samples for YAMNet
            if len(audio_data) > 15600:
                audio_data = audio_data[:15600]
            elif len(audio_data) < 15600:
                audio_data = np.pad(audio_data, (0, 15600 - len(audio_data)))

            # Run AI model
            interpreter.set_tensor(input_details[0]['index'], [audio_data])
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])
            
            # Get the top prediction
            mean_scores = np.mean(scores, axis=0)
            top_class = np.argmax(mean_scores)
            confidence = mean_scores[top_class]
            
            # If we detect something significant
            if confidence > CONFIDENCE_THRESHOLD:
                sound_name = SOUND_NAMES.get(top_class, f"Unknown sound #{top_class}")
                
                # Only process if this sound is enabled
                if len(enabled_sounds) == 0 or sound_name in enabled_sounds:
                    alert_data = {
                        'sound': sound_name,
                        'confidence': float(confidence),
                        'timestamp': time.strftime('%H:%M:%S'),
                        'id': int(time.time() * 1000)
                    }
                    print(f"🔊 Detected: {sound_name} ({confidence:.1%})")
                    
                    # Store alert (keep last 20)
                    recent_alerts.append(alert_data)
                    if len(recent_alerts) > 20:
                        recent_alerts.pop(0)
                    
                    # Send Pushbullet notification with throttling
                    current_time = time.time()
                    should_send = True
                    if (sound_name == last_notification_sound and 
                        current_time - last_notification_time < NOTIFICATION_COOLDOWN):
                        should_send = False
                        print(f"⏳ Notification throttled: {sound_name}")
                    
                    if should_send:
                        try:
                            pb = Pushbullet(PB_API_KEY)
                            pb.push_note(
                                f"🔊 {sound_name} detected", 
                                f"{confidence:.1%} confidence - Check Sound Alert Pro"
                            )
                            print(f"📱 Pushbullet notification sent: {sound_name}")
                            last_notification_sound = sound_name
                            last_notification_time = current_time
                        except Exception as e:
                            print(f"❌ Push notification failed: {e}")
                    
                    # Also emit via WebSocket
                    socketio.emit('sound_alert', alert_data)
                
        except Exception as e:
            print(f"Error in sound monitoring: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return "Sound Alert Pro Server is Running! Use /start to begin monitoring."

@app.route('/start')
def start_monitoring():
    global is_monitoring
    if not is_monitoring:
        is_monitoring = True
        thread = threading.Thread(target=sound_monitoring_thread)
        thread.daemon = True
        thread.start()
        return "Sound monitoring STARTED"
    return "Sound monitoring is already running"

@app.route('/stop')
def stop_monitoring():
    global is_monitoring
    is_monitoring = False
    return "Sound monitoring STOPPED"

@app.route('/api/alerts')
def get_alerts():
    """Return recent alerts as formatted HTML"""
    try:
        active_alerts = len(recent_alerts)
        monitored_sounds = len(enabled_sounds)
        critical_alerts = len([a for a in recent_alerts if any(word in a.get('sound', '') for word in ['Fire', 'Smoke', 'Siren', 'Alarm'])])
        available_sounds = len(SOUND_NAMES)
        
        html = """
        <html>
        <head>
            <title>Sound Alert Pro - Dashboard</title>
            <style>
                body { 
                    font-family: 'Arial', sans-serif; 
                    background: #0f0f0f; 
                    color: #e0e0e0;
                    padding: 20px;
                    margin: 0;
                }
                .dashboard-container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .dashboard-header {
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 30px;
                    background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
                    border-radius: 15px;
                    border-left: 5px solid #4CAF50;
                }
                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 20px;
                    margin-bottom: 40px;
                }
                .stat-card {
                    background: linear-gradient(135deg, #1e1e1e, #2a2a2a);
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    border: 1px solid #333;
                }
                .stat-number {
                    font-size: 2.5em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .stat-active { color: #4CAF50; }
                .stat-monitored { color: #2196F3; }
                .stat-critical { color: #f44336; }
                .stat-available { color: #FF9800; }
                .stat-label {
                    font-size: 0.9em;
                    color: #aaa;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .section {
                    background: linear-gradient(135deg, #1a1a1a, #252525);
                    padding: 25px;
                    border-radius: 12px;
                    margin-bottom: 25px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }
                .section-header {
                    font-size: 1.3em;
                    font-weight: bold;
                    margin-bottom: 20px;
                    color: #4CAF50;
                    border-bottom: 2px solid #333;
                    padding-bottom: 10px;
                }
                .alert-item {
                    background: #2a2a2a;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 4px solid #4CAF50;
                    transition: transform 0.2s;
                }
                .alert-item:hover {
                    transform: translateX(5px);
                    background: #2f2f2f;
                }
                .alert-critical {
                    border-left-color: #f44336;
                    background: #3a2a2a;
                }
                .alert-critical:hover {
                    background: #3f2f2f;
                }
                .health-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 15px;
                }
                .health-item {
                    background: #2a2a2a;
                    padding: 15px;
                    border-radius: 8px;
                }
                .health-status {
                    color: #4CAF50;
                    font-weight: bold;
                }
                .profile-tag {
                    display: inline-block;
                    background: #333;
                    padding: 8px 15px;
                    margin: 5px;
                    border-radius: 20px;
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1 style="margin:0;font-size:2.5em;">🔊 SOUND ALERT PRO</h1>
                    <p style="margin:10px 0 0 0;color:#aaa;">Real-time Audio Monitoring Dashboard</p>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number stat-active">""" + str(active_alerts) + """</div>
                        <div class="stat-label">Active Alerts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number stat-monitored">""" + str(monitored_sounds) + """</div>
                        <div class="stat-label">Monitored Sounds</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number stat-critical">""" + str(critical_alerts) + """</div>
                        <div class="stat-label">Critical Alerts</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number stat-available">""" + str(available_sounds) + """</div>
                        <div class="stat-label">Available Sounds</div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-header">🚨 RECENT ALERTS</div>
        """
        
        # Add recent alerts (newest first)
        for alert in reversed(recent_alerts[-8:]):
            is_critical = any(word in alert.get('sound', '') for word in ['Fire', 'Smoke', 'Siren', 'Alarm'])
            alert_class = "alert-item alert-critical" if is_critical else "alert-item"
            
            html += f"""
                    <div class="{alert_class}">
                        <div style="font-weight:bold;font-size:1.1em;">{alert.get('sound', 'Unknown')}</div>
                        <div style="color:#aaa;margin-top:5px;">
                            Confidence: <strong>{alert.get('confidence', 0)*100:.1f}%</strong> | 
                            Time: <strong>{alert.get('timestamp', 'Unknown')}</strong>
                        </div>
                    </div>
            """
        
        html += """
                </div>
                
                <div class="health-grid">
                    <div class="section">
                        <div class="section-header">⚙️ SYSTEM HEALTH</div>
                        <div class="health-item">
                            <strong>Monitoring Status:</strong> 
                            <span class="health-status">""" + ("ACTIVE ✅" if is_monitoring else "INACTIVE ❌") + """</span>
                        </div>
                        <div class="health-item">
                            <strong>Total Sounds Processed:</strong> """ + str(active_alerts) + """
                        </div>
                        <div class="health-item">
                            <strong>Server Status:</strong> <span class="health-status">OPERATIONAL</span>
                        </div>
                        <div class="health-item">
                            <strong>Push Notifications:</strong> <span class="health-status">ENABLED</span>
                        </div>
                    </div>
                    
                    <div class="section">
                        <div class="section-header">🔊 ACTIVE PROFILES</div>
                        <div style="padding:15px;">
        """
        
        # Show enabled sound profiles
        for sound in enabled_sounds[:6]:
            html += f'<span class="profile-tag">{sound}</span>'
        
        if not enabled_sounds:
            html += '<span class="profile-tag">All Sounds Enabled</span>'
            
        html += """
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return html
        
    except Exception as e:
        return f"<html><body style='background:#0f0f0f;color:white;padding:20px;'><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"

@app.route('/api/enabled_sounds')
def set_enabled_sounds():
    global enabled_sounds
    sounds_param = request.args.get('sounds', '')
    enabled_sounds = sounds_param.split(',') if sounds_param else []
    print(f"✅ Enabled sounds updated: {enabled_sounds}")
    return jsonify({"status": "updated", "sounds": enabled_sounds})

@socketio.on('connect')
def handle_connect():
    print("📱 Client connected")
    emit('status', {'message': 'Connected to Sound Alert Pro'})

@socketio.on('disconnect')
def handle_disconnect():
    print("📱 Client disconnected")

if __name__ == '__main__':
    print("🚀 Starting Sound Alert Pro Server...")
    load_model()
    print("🌐 Server running at: http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
