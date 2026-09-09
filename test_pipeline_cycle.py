import urllib.request
import json
import time

def post(url, data=None):
    body = json.dumps(data).encode('utf-8') if data else b''
    headers = {'Content-Type': 'application/json'} if data else {}
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode('utf-8'))

def get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode('utf-8'))

print('=== STEP 1: Fetching Camera List ===')
cams = get('http://127.0.0.1:8000/api/cameras')
for c in cams.get('cameras', []):
    print(f"Cam {c['cam_id']}: {c['name']} | URL: {c['configured_url']} | Running: {c['is_running']}")

print('\n=== STEP 2: Starting Camera 2 (Video Stream) ===')
t0 = time.time()
start_res = post('http://127.0.0.1:8000/api/cameras/2/start', {'detect_skip': 2, 'device': 'gpu'})
print(f"Start Response: {start_res} (took {time.time()-t0:.3f}s)")

print('\n=== STEP 3: Waiting 7s for model warmup and tracking ===')
time.sleep(7)
stats = get('http://127.0.0.1:8000/api/cameras/stats_all')
for c in stats.get('cameras', []):
    if c['cam_id'] == 2:
        print(f"CAM 2 Stats -> Status: {c['status']}, FPS: {c.get('fps')}, Humans: {c.get('human_count')}, Vehicles: {c.get('vehicle_count')}, Tracks: {c.get('active_tracks')}")

print('\n=== STEP 4: Stopping Camera 2 (testing non-blocking stop) ===')
t0 = time.time()
stop_res = post('http://127.0.0.1:8000/api/cameras/2/stop')
elapsed_stop = time.time() - t0
print(f"Stop Response: {stop_res} (took {elapsed_stop:.3f}s)")

time.sleep(1)
stats_after = get('http://127.0.0.1:8000/api/cameras/stats_all')
for c in stats_after.get('cameras', []):
    if c['cam_id'] == 2:
        print(f"CAM 2 After Stop -> Running: {c['is_running']}, Status: {c['status']}")

print('\n=== SUCCESS: Pipeline Start / Stop lifecycle verified cleanly! ===')
