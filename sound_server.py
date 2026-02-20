# Project: Sound Alert Pro
# Author: Brandon Whitehead
# Original Creation Date: September 14, 2025

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import sounddevice as sd
import numpy as np
import tensorflow as tf
from scipy import signal
from sound_config import SOUND_NAMES
import csv
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

is_monitoring = False
interpreter = None
recent_alerts = []
enabled_sounds = []

last_notification_times = {}
NOTIFICATION_COOLDOWN = 30

LOG_DIR = "sound_logs"
os.makedirs(LOG_DIR, exist_ok=True)

schedule_config = {
    "enabled": False,
    "start_time": "22:00",
    "end_time": "07:00",
    "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
}

sound_stats = {
    "hourly_counts": {},
    "sound_frequency": {},
    "daily_timeline": [],
    "session_start": None
}

CRITICAL_SOUNDS = {
    "Fire alarm": 10,
    "Smoke detector, smoke alarm": 10,
    "Siren": 9,
    "Police car (siren)": 9,
    "Ambulance (siren)": 9,
    "Fire engine, fire truck (siren)": 9,
    "Civil defense siren": 9,
    "Shatter": 8,
    "Glass": 8,
    "Explosion": 8,
    "Baby cry, infant cry": 7,
    "Crying, sobbing": 7,
    "Screaming": 7,
    "Car alarm": 6,
    "Doorbell": 3,
    "Knock": 3,
    "Ding-dong": 3
}

def load_model():
    global interpreter
    interpreter = tf.lite.Interpreter(model_path='yamnet.tflite')
    interpreter.allocate_tensors()
    print("AI Model loaded")



def log_to_csv(sound_name, confidence, priority, timestamp_str):
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = os.path.join(LOG_DIR, f"sound_log_{today}.csv")
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "sound", "confidence", "priority", "date"])
        writer.writerow([timestamp_str, sound_name, f"{confidence:.4f}", priority, today])


def is_within_schedule():
    if not schedule_config["enabled"]:
        return True
    now = datetime.now()
    day_name = now.strftime("%a")
    if day_name not in schedule_config["days"]:
        return False
    current_time = now.strftime("%H:%M")
    start = schedule_config["start_time"]
    end = schedule_config["end_time"]
    if start <= end:
        return start <= current_time <= end
    else:
        return current_time >= start or current_time <= end


def update_stats(sound_name, confidence, timestamp_str):
    hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
    sound_stats["hourly_counts"][hour_key] = sound_stats["hourly_counts"].get(hour_key, 0) + 1
    sound_stats["sound_frequency"][sound_name] = sound_stats["sound_frequency"].get(sound_name, 0) + 1
    sound_stats["daily_timeline"].append({
        "timestamp": timestamp_str,
        "sound": sound_name,
        "confidence": float(confidence),
        "epoch": time.time()
    })
    if len(sound_stats["daily_timeline"]) > 200:
        sound_stats["daily_timeline"] = sound_stats["daily_timeline"][-200:]


def sound_monitoring_thread():
    global is_monitoring

    DURATION = 2
    SAMPLE_RATE = 48000
    CONFIDENCE_THRESHOLD = 0.25

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sound_stats["session_start"] = time.strftime('%Y-%m-%d %H:%M:%S')
    print("Sound monitoring started in background...")

    while is_monitoring:
        try:
            if not is_within_schedule():
                time.sleep(10)
                continue

            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE,
                          channels=1,
                          dtype='float32',
                          device='hw:1,0')
            sd.wait()
            audio_data = np.squeeze(audio)

            if SAMPLE_RATE != 16000:
                number_of_samples = int(len(audio_data) * 16000 / SAMPLE_RATE)
                audio_data = signal.resample(audio_data, number_of_samples)

            if len(audio_data) > 15600:
                audio_data = audio_data[:15600]
            elif len(audio_data) < 15600:
                audio_data = np.pad(audio_data, (0, 15600 - len(audio_data)))

            interpreter.set_tensor(input_details[0]['index'], audio_data)
            interpreter.invoke()
            scores = interpreter.get_tensor(output_details[0]['index'])
            scores = scores.copy()

            mean_scores = np.mean(scores, axis=0)
            top_class = np.argmax(mean_scores)
            confidence = mean_scores[top_class]

            if confidence > CONFIDENCE_THRESHOLD:
                sound_name = SOUND_NAMES.get(top_class, f"Unknown sound #{top_class}")
                priority = CRITICAL_SOUNDS.get(sound_name, 1)

                print(f"Detected: {sound_name} ({confidence:.1%}) [Priority: {priority}]")

                is_enabled = (len(enabled_sounds) == 0 or sound_name in enabled_sounds)

                if is_enabled:
                    timestamp_str = time.strftime('%H:%M:%S')

                    alert_data = {
                        'sound': sound_name,
                        'confidence': float(confidence),
                        'timestamp': timestamp_str,
                        'priority': priority,
                        'id': int(time.time() * 1000)
                    }

                    log_to_csv(sound_name, confidence, priority, timestamp_str)
                    update_stats(sound_name, confidence, timestamp_str)

                    # Per-sound cooldown
                    current_time = time.time()
                    last_time = last_notification_times.get(sound_name, 0)

                    # Priority-based cooldown
                    cooldown = NOTIFICATION_COOLDOWN
                    if priority >= 8:
                        cooldown = 10
                    elif priority >= 6:
                        cooldown = 20

                    should_notify = (current_time - last_time) >= cooldown

                    if should_notify:
                        last_notification_times[sound_name] = current_time
                        print(f"  Alert sent to connected clients")
                    else:
                        remaining = int(cooldown - (current_time - last_time))
                        print(f"  Throttled: {sound_name} ({remaining}s remaining)")

                    # Always store alert and emit via WebSocket
                    recent_alerts.append(alert_data)
                    if len(recent_alerts) > 50:
                        recent_alerts.pop(0)

                    # WebSocket push to phone/dashboard (works over local WiFi!)
                    socketio.emit('sound_alert', alert_data)
                else:
                    print(f"  Not enabled - Pi only")

        except Exception as e:
            print(f"Error in sound monitoring: {e}")
            time.sleep(1)


@app.route('/')
def index():
    return "Sound Alert Pro Server is Running! Use /dashboard for the full UI."

@app.route('/test')
def test():
    return "TEST WORKS"

@app.route('/start')
def start_monitoring():
    global is_monitoring
    if not is_monitoring:
        is_monitoring = True
        thread = threading.Thread(target=sound_monitoring_thread)
        thread.daemon = True
        thread.start()
        return jsonify({"status": "started", "message": "Sound monitoring STARTED"})
    return jsonify({"status": "already_running", "message": "Sound monitoring is already running"})

@app.route('/stop')
def stop_monitoring():
    global is_monitoring
    is_monitoring = False
    return jsonify({"status": "stopped", "message": "Sound monitoring STOPPED"})

@app.route('/api/enabled_sounds')
def set_enabled_sounds():
    global enabled_sounds
    sounds_param = request.args.get('sounds', '')
    enabled_sounds = sounds_param.split(',') if sounds_param else []
    print(f"Enabled sounds updated: {enabled_sounds}")
    return jsonify({"status": "updated", "sounds": enabled_sounds})


@app.route('/api/stats')
def get_stats():
    return jsonify({
        "hourly_counts": sound_stats["hourly_counts"],
        "sound_frequency": sound_stats["sound_frequency"],
        "recent_timeline": sound_stats["daily_timeline"][-50:],
        "session_start": sound_stats["session_start"],
        "total_alerts": len(recent_alerts),
        "is_monitoring": is_monitoring
    })

@app.route('/api/alerts_json')
def get_alerts_json():
    return jsonify(recent_alerts[-20:])

@app.route('/api/schedule', methods=['GET', 'POST'])
def manage_schedule():
    global schedule_config
    if request.method == 'POST':
        data = request.get_json()
        if data:
            schedule_config.update(data)
            print(f"Schedule updated: {schedule_config}")
        return jsonify({"status": "updated", "schedule": schedule_config})
    return jsonify(schedule_config)

@app.route('/api/logs')
def get_logs():
    files = []
    if os.path.exists(LOG_DIR):
        for f in sorted(os.listdir(LOG_DIR), reverse=True):
            if f.endswith('.csv'):
                filepath = os.path.join(LOG_DIR, f)
                size = os.path.getsize(filepath)
                files.append({"filename": f, "size_bytes": size})
    return jsonify(files)

@app.route('/api/logs/<filename>')
def download_log(filename):
    return send_from_directory(LOG_DIR, filename, as_attachment=True)

@app.route('/api/priority_sounds')
def get_priority_sounds():
    return jsonify(CRITICAL_SOUNDS)


@app.route('/api/alerts')
def get_alerts():
    alert_items_html = ""
    for alert in reversed(recent_alerts):
        priority = alert.get('priority', 1)
        if priority >= 8:
            a_class = "alert alert-critical"
            badge_bg = "#f44336"
            badge_label = "CRITICAL"
        elif priority >= 6:
            a_class = "alert alert-high"
            badge_bg = "#FF9800"
            badge_label = "HIGH"
        else:
            a_class = "alert"
            badge_bg = "#2196F3"
            badge_label = "LOW"

        conf_pct = f"{alert['confidence'] * 100:.1f}"
        alert_items_html += (
            '<div class="' + a_class + '">'
            '<span class="alert-name">' + alert['sound'] + '</span>'
            '<span class="priority-badge" style="background:' + badge_bg + '">' + badge_label + ' (' + str(priority) + ')</span>'
            '<div class="alert-meta">'
            'Confidence: <strong>' + conf_pct + '%</strong> | '
            'Time: <strong>' + alert['timestamp'] + '</strong>'
            '</div></div>'
        )

    if not recent_alerts:
        alert_items_html = '<div class="empty-state">No alerts yet. Start monitoring to see detections here.</div>'

    html = ALERTS_HTML.replace("__ALERT_ITEMS__", alert_items_html)
    return html


@app.route('/dashboard')
def dashboard():
    active_alerts = len(recent_alerts)
    monitored_sounds = len(enabled_sounds)
    critical_alerts = len([a for a in recent_alerts if a.get('priority', 0) >= 8])
    available_sounds = len(SOUND_NAMES)

    if is_monitoring:
        mon_status_class = "status-active"
        mon_status_text = "ACTIVE"
    else:
        mon_status_class = "status-inactive"
        mon_status_text = "INACTIVE"

    sched_checked = "checked" if schedule_config["enabled"] else ""

    profile_tags_html = ""
    for sound in enabled_sounds[:8]:
        profile_tags_html += '<span class="profile-tag">' + sound + '</span>'
    if not enabled_sounds:
        profile_tags_html = '<span class="profile-tag">All Sounds Enabled (No Filter)</span>'

    html = DASHBOARD_HTML
    html = html.replace("__ACTIVE_ALERTS__", str(active_alerts))
    html = html.replace("__MONITORED_SOUNDS__", str(monitored_sounds))
    html = html.replace("__CRITICAL_ALERTS__", str(critical_alerts))
    html = html.replace("__AVAILABLE_SOUNDS__", str(available_sounds))
    html = html.replace("__MON_STATUS_CLASS__", mon_status_class)
    html = html.replace("__MON_STATUS_TEXT__", mon_status_text)
    html = html.replace("__SCHEDULE_START__", schedule_config["start_time"])
    html = html.replace("__SCHEDULE_END__", schedule_config["end_time"])
    html = html.replace("__SCHEDULE_CHECKED__", sched_checked)
    html = html.replace("__PROFILE_TAGS__", profile_tags_html)
    return html


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('status', {'message': 'Connected to Sound Alert Pro', 'monitoring': is_monitoring})

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


ALERTS_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sound Alert Pro - Phone Alerts</title>
    <!--
        OFFLINE NOTE: Socket.IO loaded from CDN. For no-internet use,
        pre-download and serve from /static/socket.io.min.js (see dashboard comments).
    -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            padding: 16px;
            padding-bottom: 80px;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #0d1117, #161b22);
            border-radius: 12px;
            border: 1px solid #30363d;
        }
        .header h2 { color: #4CAF50; margin-bottom: 6px; font-size: 1.4em; }
        .header p { color: #8b949e; font-size: 0.85em; }
        .connection-bar {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            margin-top: 10px;
        }
        .conn-dot {
            width: 10px; height: 10px; border-radius: 50%;
            background: #f44336; display: inline-block;
        }
        .conn-dot.connected { background: #4CAF50; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .conn-text { font-size: 0.85em; color: #8b949e; }

        .alert {
            background: #161b22;
            padding: 16px;
            margin: 10px 0;
            border-radius: 10px;
            border: 1px solid #30363d;
            border-left: 5px solid #4CAF50;
            transition: transform 0.15s;
        }
        .alert:hover { transform: translateX(4px); }
        .alert-critical { border-left-color: #f44336; background: #1a1215; }
        .alert-high { border-left-color: #FF9800; background: #1a1812; }
        .alert-name { font-weight: 600; font-size: 1.1em; }
        .alert-meta { color: #8b949e; font-size: 0.9em; margin-top: 6px; }
        .priority-badge {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.75em; color: white; margin-left: 8px; font-weight: 600;
        }
        .empty-state { text-align: center; color: #8b949e; padding: 40px; font-size: 1.1em; }
        .alert-new { animation: flashIn 0.5s ease-out; }
        @keyframes flashIn {
            0% { background: #2a4a2a; transform: translateX(10px); }
            100% { background: #161b22; transform: translateX(0); }
        }

        .flash-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9999;
            justify-content: center; align-items: center; flex-direction: column;
            font-size: 1.5em; font-weight: bold; text-align: center; padding: 40px;
            cursor: pointer;
        }
        .flash-overlay.active { display: flex; animation: flashOverlay 0.3s ease-out; }
        .flash-overlay .flash-sound { font-size: 1.8em; margin-bottom: 10px; }
        .flash-overlay .flash-detail { font-size: 0.6em; opacity: 0.9; }
        .flash-overlay .flash-dismiss { font-size: 0.5em; margin-top: 20px; opacity: 0.7; }
        @keyframes flashOverlay { 0% { opacity: 0; } 100% { opacity: 1; } }

        .enable-btn {
            display: block; width: 100%; padding: 14px; margin-bottom: 16px;
            background: #238636; color: white; border: none; border-radius: 10px;
            font-size: 1em; font-weight: 600; cursor: pointer; text-align: center;
            transition: background 0.2s;
        }
        .enable-btn:hover { background: #2ea043; }
        .enable-btn.active { background: #1f6feb; }
    </style>
</head>
<body>
    <!-- Fullscreen flash overlay -->
    <div class="flash-overlay" id="flashOverlay" onclick="dismissFlash()">
        <div class="flash-sound" id="flashSound"></div>
        <div class="flash-detail" id="flashDetail"></div>
        <div class="flash-dismiss">Tap to dismiss</div>
    </div>

    <div class="header">
        <h2>&#x1F50A; Sound Alert Pro</h2>
        <p>Live Alerts &mdash; Enabled Sounds Only</p>
        <div class="connection-bar">
            <span class="conn-dot" id="connDot"></span>
            <span class="conn-text" id="connText">Connecting...</span>
        </div>
    </div>

    <button class="enable-btn" id="enableBtn" onclick="enableAlerts()">
        &#x1F514; Tap to Enable Vibration &amp; Sound Alerts
    </button>

    <div id="alertFeed">
        __ALERT_ITEMS__
    </div>

<script>
    var socket = io();
    var audioCtx = null;
    var alertsEnabled = false;
    var flashTimeout = null;

    socket.on('connect', function() {
        document.getElementById('connDot').className = 'conn-dot connected';
        document.getElementById('connText').textContent = 'Live - Connected';
        document.getElementById('connText').style.color = '#4CAF50';
    });
    socket.on('disconnect', function() {
        document.getElementById('connDot').className = 'conn-dot';
        document.getElementById('connText').textContent = 'Disconnected';
        document.getElementById('connText').style.color = '#f44336';
    });

    function enableAlerts() {

        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        alertsEnabled = true;

        var btn = document.getElementById('enableBtn');
        btn.textContent = '\\u2705 Vibration & Sound Alerts ON';
        btn.className = 'enable-btn active';

        if (navigator.vibrate) {
            navigator.vibrate(100);
        }
        playBeep(3);
    }

    function playBeep(priority) {
        if (!alertsEnabled || !audioCtx) return;
        if (audioCtx.state === 'suspended') audioCtx.resume();

        var osc = audioCtx.createOscillator();
        var gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);

        if (priority >= 8) {
            osc.frequency.setValueAtTime(880, audioCtx.currentTime);
            osc.frequency.setValueAtTime(440, audioCtx.currentTime + 0.15);
            osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.3);
            osc.frequency.setValueAtTime(440, audioCtx.currentTime + 0.45);
            gain.gain.setValueAtTime(0.5, audioCtx.currentTime);
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.6);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.6);
        } else if (priority >= 6) {
            osc.frequency.setValueAtTime(660, audioCtx.currentTime);
            gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
            gain.gain.setValueAtTime(0, audioCtx.currentTime + 0.12);
            gain.gain.setValueAtTime(0.4, audioCtx.currentTime + 0.2);
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.35);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.35);
        } else {
            osc.frequency.setValueAtTime(520, audioCtx.currentTime);
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
            gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.2);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.2);
        }
    }

    function showFlash(data) {
        var priority = data.priority || 1;
        var overlay = document.getElementById('flashOverlay');
        document.getElementById('flashSound').textContent = data.sound;
        document.getElementById('flashDetail').textContent =
            (data.confidence * 100).toFixed(1) + '% confidence at ' + data.timestamp;

        if (priority >= 8) {
            overlay.style.background = 'rgba(244, 67, 54, 0.95)';
        } else if (priority >= 6) {
            overlay.style.background = 'rgba(255, 152, 0, 0.95)';
        } else {
            overlay.style.background = 'rgba(76, 175, 80, 0.92)';
        }
        overlay.style.color = 'white';
        overlay.className = 'flash-overlay active';

        if (flashTimeout) clearTimeout(flashTimeout);
        var duration = priority >= 8 ? 6000 : priority >= 6 ? 4000 : 2500;
        flashTimeout = setTimeout(dismissFlash, duration);

        if (alertsEnabled && navigator.vibrate) {
            if (priority >= 8) {
                navigator.vibrate([200, 100, 200, 100, 400]);
            } else if (priority >= 6) {
                navigator.vibrate([200, 100, 200]);
            } else {
                navigator.vibrate(150);
            }
        }
    }

    function dismissFlash() {
        document.getElementById('flashOverlay').className = 'flash-overlay';
        if (flashTimeout) { clearTimeout(flashTimeout); flashTimeout = null; }
    }

    socket.on('sound_alert', function(data) {
        var priority = data.priority || 1;

        showFlash(data);
        playBeep(priority);

        var alertClass = 'alert alert-new';
        var badgeColor = '#2196F3';
        var badgeText = 'LOW';
        if (priority >= 8) { alertClass = 'alert alert-critical alert-new'; badgeColor = '#f44336'; badgeText = 'CRITICAL'; }
        else if (priority >= 6) { alertClass = 'alert alert-high alert-new'; badgeColor = '#FF9800'; badgeText = 'HIGH'; }

        var html = '<div class="' + alertClass + '">' +
            '<span class="alert-name">' + data.sound + '</span>' +
            '<span class="priority-badge" style="background:' + badgeColor + '">' + badgeText + ' (' + priority + ')</span>' +
            '<div class="alert-meta">' +
            'Confidence: <strong>' + (data.confidence * 100).toFixed(1) + '%</strong> | ' +
            'Time: <strong>' + data.timestamp + '</strong></div></div>';

        var feed = document.getElementById('alertFeed');
        var empty = feed.querySelector('.empty-state');
        if (empty) empty.remove();
        feed.insertAdjacentHTML('afterbegin', html);

        while (feed.children.length > 30) {
            feed.removeChild(feed.lastChild);
        }
    });
</script>
</body>
</html>'''


DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sound Alert Pro - Live Dashboard</title>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .dashboard-container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        .dashboard-header {
            text-align: center; padding: 30px;
            background: linear-gradient(135deg, #0d1117, #161b22);
            border-radius: 16px; border: 1px solid #30363d;
            margin-bottom: 24px; position: relative; overflow: hidden;
        }
        .dashboard-header::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #4CAF50, #2196F3, #FF9800, #f44336);
        }
        .dashboard-header h1 { font-size: 2.2em; margin-bottom: 8px; }
        .live-badge {
            display: inline-block; background: #4CAF50; color: white;
            padding: 4px 14px; border-radius: 20px; font-size: 0.8em;
            animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }

        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; margin-bottom: 24px;
        }
        .stat-card {
            background: linear-gradient(135deg, #0d1117, #161b22);
            padding: 24px; border-radius: 12px; text-align: center;
            border: 1px solid #30363d; transition: transform 0.2s, border-color 0.2s;
        }
        .stat-card:hover { transform: translateY(-2px); border-color: #4CAF50; }
        .stat-number { font-size: 2.5em; font-weight: bold; }
        .stat-label { font-size: 0.85em; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
        .green { color: #4CAF50; } .blue { color: #2196F3; }
        .red { color: #f44336; } .orange { color: #FF9800; }

        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

        .section {
            background: linear-gradient(135deg, #0d1117, #161b22);
            padding: 24px; border-radius: 12px; border: 1px solid #30363d;
        }
        .section-header {
            font-size: 1.15em; font-weight: 600; margin-bottom: 16px;
            color: #c9d1d9; border-bottom: 1px solid #21262d; padding-bottom: 10px;
        }

        .alert-item {
            background: #161b22; padding: 14px; margin: 8px 0; border-radius: 8px;
            border-left: 4px solid #4CAF50; transition: transform 0.15s, background 0.15s;
        }
        .alert-item:hover { transform: translateX(4px); background: #1c2128; }
        .alert-critical { border-left-color: #f44336; }
        .alert-high { border-left-color: #FF9800; }
        .alert-name { font-weight: 600; font-size: 1.05em; }
        .alert-meta { color: #8b949e; font-size: 0.9em; margin-top: 4px; }
        .priority-badge {
            display: inline-block; padding: 2px 10px; border-radius: 12px;
            font-size: 0.75em; color: white; margin-left: 8px; font-weight: 600;
        }
        .alert-new { animation: flashIn 0.5s ease-out; }
        @keyframes flashIn {
            0% { background: #2a4a2a; transform: translateX(10px); }
            100% { background: #161b22; transform: translateX(0); }
        }

        .chart-container { position: relative; height: 250px; margin-top: 10px; }

        .health-row { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #21262d; }
        .health-row:last-child { border-bottom: none; }
        .health-status { font-weight: 600; }
        .status-active { color: #4CAF50; }
        .status-inactive { color: #f44336; }

        .profile-tag {
            display: inline-block; background: #21262d; border: 1px solid #30363d;
            padding: 6px 14px; margin: 4px; border-radius: 20px; font-size: 0.85em;
        }

        .schedule-form { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .schedule-form label { color: #8b949e; font-size: 0.9em; }
        .schedule-form input, .schedule-form select {
            background: #0d1117; border: 1px solid #30363d; color: white;
            padding: 8px 12px; border-radius: 6px; width: 100%;
        }

        .btn {
            padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer;
            font-weight: 600; font-size: 0.95em; transition: all 0.2s;
        }
        .btn-green { background: #238636; color: white; }
        .btn-green:hover { background: #2ea043; }
        .btn-red { background: #da3633; color: white; }
        .btn-red:hover { background: #f85149; }
        .btn-blue { background: #1f6feb; color: white; }
        .btn-blue:hover { background: #388bfd; }

        .controls-bar { display: flex; gap: 12px; justify-content: center; margin-bottom: 24px; flex-wrap: wrap; }

        .alert-list { max-height: 400px; overflow-y: auto; }
        .alert-list::-webkit-scrollbar { width: 6px; }
        .alert-list::-webkit-scrollbar-track { background: transparent; }
        .alert-list::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

        .log-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px; background: #161b22; border-radius: 6px; margin: 6px 0;
        }
        .log-item a { color: #58a6ff; text-decoration: none; }
        .log-item a:hover { text-decoration: underline; }

        .flash-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 9999;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            font-size: 1.5em;
            font-weight: bold;
            text-align: center;
            padding: 40px;
            cursor: pointer;
        }
        .flash-overlay.active {
            display: flex;
            animation: flashOverlay 0.3s ease-out;
        }
        .flash-overlay .flash-sound { font-size: 1.8em; margin-bottom: 10px; }
        .flash-overlay .flash-detail { font-size: 0.6em; opacity: 0.9; }
        .flash-overlay .flash-dismiss { font-size: 0.5em; margin-top: 20px; opacity: 0.7; }
        @keyframes flashOverlay {
            0% { opacity: 0; } 100% { opacity: 1; }
        }

        /* Sound toggle for phone */
        .sound-toggle {
            display: inline-block; padding: 6px 16px; border-radius: 20px;
            font-size: 0.85em; cursor: pointer; margin-left: 10px;
            border: 1px solid #30363d; background: #21262d; color: #8b949e;
            transition: all 0.2s;
        }
        .sound-toggle.enabled { background: #238636; color: white; border-color: #238636; }
    </style>
</head>
<body>
    <div class="flash-overlay" id="flashOverlay" onclick="dismissFlash()">
        <div class="flash-sound" id="flashSound"></div>
        <div class="flash-detail" id="flashDetail"></div>
        <div class="flash-dismiss">Tap to dismiss</div>
    </div>

    <div class="dashboard-container">
        <div class="dashboard-header">
            <h1>&#x1F50A; SOUND ALERT PRO</h1>
            <p style="color:#8b949e;margin-bottom:8px;">Real-time AI Audio Monitoring Dashboard</p>
            <span class="live-badge" id="liveBadge">&#x25CF; LIVE</span>
            <span id="connectionStatus" style="margin-left:10px;font-size:0.85em;color:#8b949e;">Connecting...</span>
            <span class="sound-toggle" id="soundToggle" onclick="toggleAlertSound()">&#x1F508; Alerts Muted</span>
        </div>

        <div class="controls-bar">
            <button class="btn btn-green" onclick="startMonitoring()">&#x25B6; Start Monitoring</button>
            <button class="btn btn-red" onclick="stopMonitoring()">&#x25A0; Stop Monitoring</button>
            <button class="btn btn-blue" onclick="refreshStats()">&#x21BB; Refresh Stats</button>
            <button class="btn btn-blue" onclick="exportAlerts()">&#x1F4E5; Export Alerts</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number green" id="statAlerts">__ACTIVE_ALERTS__</div>
                <div class="stat-label">Total Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number blue" id="statMonitored">__MONITORED_SOUNDS__</div>
                <div class="stat-label">Monitored Sounds</div>
            </div>
            <div class="stat-card">
                <div class="stat-number red" id="statCritical">__CRITICAL_ALERTS__</div>
                <div class="stat-label">Critical Alerts</div>
            </div>
            <div class="stat-card">
                <div class="stat-number orange" id="statAvailable">__AVAILABLE_SOUNDS__</div>
                <div class="stat-label">Available Sounds</div>
            </div>
        </div>

        <div class="main-grid">
            <div class="section">
                <div class="section-header">&#x1F6A8; LIVE ALERT FEED</div>
                <div class="alert-list" id="alertFeed">
                    <p style="color:#8b949e;text-align:center;padding:20px;">Waiting for detections...</p>
                </div>
            </div>

            <div class="section">
                <div class="section-header">&#x1F4CA; SOUND FREQUENCY</div>
                <div class="chart-container">
                    <canvas id="frequencyChart"></canvas>
                </div>
            </div>

            <div class="section">
                <div class="section-header">&#x1F4C8; HOURLY ACTIVITY</div>
                <div class="chart-container">
                    <canvas id="hourlyChart"></canvas>
                </div>
            </div>

            <div class="section">
                <div class="section-header">&#x2699;&#xFE0F; SYSTEM HEALTH</div>
                <div class="health-row">
                    <span>Monitoring Status</span>
                    <span class="health-status __MON_STATUS_CLASS__" id="healthMonitoring">__MON_STATUS_TEXT__</span>
                </div>
                <div class="health-row">
                    <span>AI Model</span>
                    <span class="health-status status-active">LOADED</span>
                </div>
                <div class="health-row">
                    <span>Network Mode</span>
                    <span class="health-status status-active">LOCAL (SoundAlertNet)</span>
                </div>
                <div class="health-row">
                    <span>Priority System</span>
                    <span class="health-status status-active">ACTIVE</span>
                </div>
                <div class="health-row">
                    <span>CSV Logging</span>
                    <span class="health-status status-active">ACTIVE</span>
                </div>
                <div class="health-row">
                    <span>Session Start</span>
                    <span class="health-status" id="sessionStart" style="color:#8b949e;">-</span>
                </div>
            </div>

            <div class="section">
                <div class="section-header">&#x1F4C5; SCHEDULED MONITORING</div>
                <div class="schedule-form">
                    <div>
                        <label>Start Time</label>
                        <input type="time" id="schedStart" value="__SCHEDULE_START__">
                    </div>
                    <div>
                        <label>End Time</label>
                        <input type="time" id="schedEnd" value="__SCHEDULE_END__">
                    </div>
                </div>
                <div style="margin-top:12px;">
                    <label style="color:#8b949e;font-size:0.9em;">
                        <input type="checkbox" id="schedEnabled" __SCHEDULE_CHECKED__>
                        Enable scheduled monitoring
                    </label>
                </div>
                <button class="btn btn-green" style="margin-top:12px;width:100%;" onclick="saveSchedule()">Save Schedule</button>
            </div>

            <div class="section">
                <div class="section-header">&#x1F50A; ACTIVE SOUND PROFILES</div>
                <div id="profileTags">__PROFILE_TAGS__</div>
            </div>

            <div class="section">
                <div class="section-header">&#x1F4C1; SOUND LOG FILES</div>
                <div id="logFiles">
                    <p style="color:#8b949e;">Loading logs...</p>
                </div>
            </div>
        </div>
    </div>

<script>

    var socket = io();

    socket.on('connect', function() {
        document.getElementById('connectionStatus').textContent = 'Connected';
        document.getElementById('connectionStatus').style.color = '#4CAF50';
    });
    socket.on('disconnect', function() {
        document.getElementById('connectionStatus').textContent = 'Disconnected';
        document.getElementById('connectionStatus').style.color = '#f44336';
        document.getElementById('liveBadge').style.background = '#da3633';
    });

    var audioCtx = null;
    var alertSoundEnabled = false;

    function initAudio() {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    function playAlertBeep(priority) {
        if (!alertSoundEnabled || !audioCtx) return;


        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        var oscillator = audioCtx.createOscillator();
        var gainNode = audioCtx.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        if (priority >= 8) {
            oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
            oscillator.frequency.setValueAtTime(440, audioCtx.currentTime + 0.15);
            oscillator.frequency.setValueAtTime(880, audioCtx.currentTime + 0.3);
            oscillator.frequency.setValueAtTime(440, audioCtx.currentTime + 0.45);
            gainNode.gain.setValueAtTime(0.5, audioCtx.currentTime);
            gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.6);
            oscillator.start(audioCtx.currentTime);
            oscillator.stop(audioCtx.currentTime + 0.6);
        } else if (priority >= 6) {
            oscillator.frequency.setValueAtTime(660, audioCtx.currentTime);
            gainNode.gain.setValueAtTime(0.4, audioCtx.currentTime);
            gainNode.gain.setValueAtTime(0, audioCtx.currentTime + 0.12);
            gainNode.gain.setValueAtTime(0.4, audioCtx.currentTime + 0.2);
            gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.35);
            oscillator.start(audioCtx.currentTime);
            oscillator.stop(audioCtx.currentTime + 0.35);
        } else {
            oscillator.frequency.setValueAtTime(520, audioCtx.currentTime);
            oscillator.type = 'sine';
            gainNode.gain.setValueAtTime(0.25, audioCtx.currentTime);
            gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.2);
            oscillator.start(audioCtx.currentTime);
            oscillator.stop(audioCtx.currentTime + 0.2);
        }
    }

    function toggleAlertSound() {
        if (!audioCtx) initAudio();
        alertSoundEnabled = !alertSoundEnabled;
        var btn = document.getElementById('soundToggle');
        if (alertSoundEnabled) {
            btn.textContent = '\\u1F50A Alerts ON';
            btn.className = 'sound-toggle enabled';
            playAlertBeep(3);
        } else {
            btn.textContent = '\\u1F508 Alerts Muted';
            btn.className = 'sound-toggle';
        }
    }

    var flashTimeout = null;

    function showFlash(data) {
        var priority = data.priority || 1;
        var overlay = document.getElementById('flashOverlay');
        var soundEl = document.getElementById('flashSound');
        var detailEl = document.getElementById('flashDetail');

        soundEl.textContent = data.sound;
        detailEl.textContent = (data.confidence * 100).toFixed(1) + '% confidence at ' + data.timestamp;

        if (priority >= 8) {
            overlay.style.background = 'rgba(244, 67, 54, 0.95)';
            overlay.style.color = 'white';
        } else if (priority >= 6) {
            overlay.style.background = 'rgba(255, 152, 0, 0.95)';
            overlay.style.color = 'white';
        } else {
            overlay.style.background = 'rgba(76, 175, 80, 0.92)';
            overlay.style.color = 'white';
        }

        overlay.className = 'flash-overlay active';

        if (flashTimeout) clearTimeout(flashTimeout);
        var duration = priority >= 8 ? 6000 : priority >= 6 ? 4000 : 2500;
        flashTimeout = setTimeout(dismissFlash, duration);

        if (navigator.vibrate) {
            if (priority >= 8) {
                navigator.vibrate([200, 100, 200, 100, 400]);            } else if (priority >= 6) {
                navigator.vibrate([200, 100, 200]);            
 } else {
                navigator.vibrate(150);
            }
        }
    }

    function dismissFlash() {
        document.getElementById('flashOverlay').className = 'flash-overlay';
        if (flashTimeout) { clearTimeout(flashTimeout); flashTimeout = null; }
    }

    var alertCount = __ACTIVE_ALERTS__;
    var criticalCount = __CRITICAL_ALERTS__;

    socket.on('sound_alert', function(data) {
        alertCount++;
        var priority = data.priority || 1;
        if (priority >= 8) criticalCount++;

        document.getElementById('statAlerts').textContent = alertCount;
        document.getElementById('statCritical').textContent = criticalCount;

        showFlash(data);
        playAlertBeep(priority);

        var alertClass = 'alert-item alert-new';
        var badgeColor = '#2196F3';
        var badgeText = 'LOW';
        if (priority >= 8) { alertClass += ' alert-critical'; badgeColor = '#f44336'; badgeText = 'CRITICAL'; }
        else if (priority >= 6) { alertClass += ' alert-high'; badgeColor = '#FF9800'; badgeText = 'HIGH'; }

        var alertHtml = '<div class="' + alertClass + '">' +
            '<span class="alert-name">' + data.sound + '</span>' +
            '<span class="priority-badge" style="background:' + badgeColor + '">' + badgeText + ' (' + priority + ')</span>' +
            '<div class="alert-meta">' +
            'Confidence: <strong>' + (data.confidence * 100).toFixed(1) + '%</strong> | ' +
            'Time: <strong>' + data.timestamp + '</strong>' +
            '</div></div>';

        var feed = document.getElementById('alertFeed');
        if (feed.querySelector('p')) feed.innerHTML = '';
        feed.insertAdjacentHTML('afterbegin', alertHtml);

        while (feed.children.length > 30) {
            feed.removeChild(feed.lastChild);
        }

        refreshStats();
    });

    var freqChart = null;
    var hourlyChart = null;

    function initCharts() {
        var freqCtx = document.getElementById('frequencyChart').getContext('2d');
        freqChart = new Chart(freqCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#4CAF50', '#2196F3', '#FF9800', '#f44336', '#9C27B0',
                        '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
                    ]
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: '#8b949e', font: { size: 11 } } } }
            }
        });

        var hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        hourlyChart = new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{ label: 'Detections', data: [],
                    backgroundColor: 'rgba(76, 175, 80, 0.6)', borderColor: '#4CAF50', borderWidth: 1
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
                    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    function refreshStats() {
        fetch('/api/stats')
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (freqChart && data.sound_frequency) {
                    var entries = Object.entries(data.sound_frequency);
                    entries.sort(function(a, b) { return b[1] - a[1]; });
                    var sorted = entries.slice(0, 10);
                    freqChart.data.labels = sorted.map(function(s) { return s[0]; });
                    freqChart.data.datasets[0].data = sorted.map(function(s) { return s[1]; });
                    freqChart.update();
                }
                if (hourlyChart && data.hourly_counts) {
                    var hours = Object.entries(data.hourly_counts).slice(-12);
                    hourlyChart.data.labels = hours.map(function(h) { return h[0].split(' ')[1] || h[0]; });
                    hourlyChart.data.datasets[0].data = hours.map(function(h) { return h[1]; });
                    hourlyChart.update();
                }
                if (data.session_start) {
                    document.getElementById('sessionStart').textContent = data.session_start;
                }
            })
            .catch(function(err) { console.log('Stats fetch error:', err); });
    }

    function startMonitoring() {
        fetch('/start').then(function(r) { return r.json(); }).then(function(data) {
            document.getElementById('healthMonitoring').textContent = 'ACTIVE';
            document.getElementById('healthMonitoring').className = 'health-status status-active';
        });
    }
    function stopMonitoring() {
        fetch('/stop').then(function(r) { return r.json(); }).then(function(data) {
            document.getElementById('healthMonitoring').textContent = 'INACTIVE';
            document.getElementById('healthMonitoring').className = 'health-status status-inactive';
        });
    }
    function saveSchedule() {
        var config = {
            enabled: document.getElementById('schedEnabled').checked,
            start_time: document.getElementById('schedStart').value,
            end_time: document.getElementById('schedEnd').value
        };
        fetch('/api/schedule', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        }).then(function(r) { return r.json(); }).then(function(data) { alert('Schedule saved!'); });
    }
    function exportAlerts() {
        fetch('/api/alerts_json').then(function(r) { return r.json(); }).then(function(data) {
            var csvContent = 'Time,Sound,Confidence,Priority\\n';
            data.forEach(function(a) {
                csvContent += a.timestamp + ',' + a.sound + ',' + (a.confidence * 100).toFixed(1) + '%,' + (a.priority || 1) + '\\n';
            });
            var blob = new Blob([csvContent], { type: 'text/csv' });
            var url = URL.createObjectURL(blob);
            var link = document.createElement('a');
            link.href = url; link.download = 'sound_alerts_export.csv'; link.click();
        });
    }
    function loadLogs() {
        fetch('/api/logs').then(function(r) { return r.json(); }).then(function(files) {
            var container = document.getElementById('logFiles');
            if (files.length === 0) {
                container.innerHTML = '<p style="color:#8b949e;">No log files yet. Start monitoring to generate logs.</p>';
                return;
            }
            var html = '';
            files.forEach(function(f) {
                html += '<div class="log-item"><span>' + f.filename + '</span>' +
                    '<a href="/api/logs/' + f.filename + '">Download (' + (f.size_bytes / 1024).toFixed(1) + ' KB)</a></div>';
            });
            container.innerHTML = html;
        });
    }

    initCharts();
    refreshStats();
    loadLogs();
    setInterval(refreshStats, 10000);
    setInterval(loadLogs, 30000);

    fetch('/api/alerts_json').then(function(r) { return r.json(); }).then(function(alerts) {
        alertCount = alerts.length;
        criticalCount = alerts.filter(function(a) { return (a.priority || 0) >= 8; }).length;
        document.getElementById('statAlerts').textContent = alertCount;
        document.getElementById('statCritical').textContent = criticalCount;
        var feed = document.getElementById('alertFeed');
        if (alerts.length > 0) {
            feed.innerHTML = '';
            alerts.reverse().forEach(function(data) {
                var priority = data.priority || 1;
                var alertClass = 'alert-item';
                var badgeColor = '#2196F3'; var badgeText = 'LOW';
                if (priority >= 8) { alertClass += ' alert-critical'; badgeColor = '#f44336'; badgeText = 'CRITICAL'; }
                else if (priority >= 6) { alertClass += ' alert-high'; badgeColor = '#FF9800'; badgeText = 'HIGH'; }
                feed.innerHTML += '<div class="' + alertClass + '"><span class="alert-name">' + data.sound + '</span>' +
                    '<span class="priority-badge" style="background:' + badgeColor + '">' + badgeText + ' (' + priority + ')</span>' +
                    '<div class="alert-meta">Confidence: <strong>' + (data.confidence * 100).toFixed(1) + '%</strong> | ' +
                    'Time: <strong>' + data.timestamp + '</strong></div></div>';
            });
        }
    });
</script>
</body>
</html>'''


if __name__ == '__main__':
    print("Starting Sound Alert Pro Server...")
    load_model()
    print("Server running at: http://0.0.0.0:5000")
    print("Dashboard at: http://0.0.0.0:5000/dashboard")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
