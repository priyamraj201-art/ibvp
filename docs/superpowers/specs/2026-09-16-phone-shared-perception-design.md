# Phone Camera "Shared Perception" — Design

Date: 2026-09-16
Status: Approved for implementation planning

## 1. Purpose

Let a phone join the IBVP live surveillance grid as a camera with **zero app
install** — scan a QR code shown on the dashboard, grant camera access in the
phone's normal browser, and the phone's video appears as a new tile in the
live grid, running through the same detection/tracking/ANPR/FRS pipeline as
every other camera.

This complements, and does not replace, the existing DroidCam/RTSP/webcam/
file ingestion path (`ZeroLagCapture` in `dashboard/stream_server.py`), which
is unchanged.

Scope is LAN/WiFi only — phone and laptop are on the same network. No
internet-facing relay, no cloud infra, no GPS/IMU telemetry, no map. Those
were evaluated (inspired by a reference "Shared Perception" feature in an
unrelated project, `D:\dev\kaya-main`) and explicitly dropped as out of scope
for this fixed-perimeter surveillance use case — see §7.

## 2. Architecture

```
Phone browser --(getUserMedia + WebSocket, JPEG frames)--> FastAPI WebSocket endpoint
                                                                    |
                                                      PhoneCamCapture frame buffer
                                                                    |
                                              CameraPipelineWorker (existing YOLOX/
                                              ByteTrack/ANPR/FRS pipeline, unchanged)
                                                                    |
                                                    Live grid tile (dashboard/live.html)
```

The only new capture primitive is `PhoneCamCapture`, which implements the
same interface as the existing `ZeroLagCapture`:
`start()`, `read_latest() -> (bool, np.ndarray | None)`, `stop()`,
`is_opened`, `source_label`. `CameraPipelineWorker._run_loop()`
(`dashboard/stream_server.py`) gets one conditional branch: if the camera's
`url` starts with `phonecam://`, construct a `PhoneCamCapture` bound to that
device id instead of a `ZeroLagCapture` bound to the URL. Everything
downstream — tracker, alert system, FRS/ANPR pipelines, MJPEG output, the
live grid UI — is unchanged and treats a phone camera exactly like any other
camera.

## 3. New components

| Component | Location | Responsibility |
|---|---|---|
| `PhoneCamCapture` | `dashboard/stream_server.py` (near `ZeroLagCapture`) | Holds the latest decoded frame for one phone device; fed by the WebSocket handler; exposes the `ZeroLagCapture`-compatible interface; tracks a stale-frame timeout. |
| `PhoneDeviceRegistry` | `dashboard/phonecam.py` (new module) | In-memory map of `device_id -> PhoneCamCapture`, plus pending `pair_token -> expiry` entries. Thread-safe (mirrors `CameraManager`'s locking style). |
| Pairing API | `dashboard/routers/live.py` (new routes) | `POST /api/phonecam/pair` generates a token + QR PNG (base64) + join URL. |
| Join page | `dashboard/templates/phonecam_join.html` (new) | Mobile-optimized page: camera preview, name field, "Start Sharing" button, status text. Vanilla JS, no build step, consistent with existing templates. |
| WebSocket ingest | `dashboard/routers/live.py` (new route) | `WS /ws/phonecam/{pair_token}/{device_id}` — validates token, registers/reuses the camera entry, decodes incoming JPEG frames into the matching `PhoneCamCapture`. |
| "+ Add Phone Camera" button + QR modal | `dashboard/templates/live.html` | Calls the pairing API, displays the returned QR image and join URL as text (in case scanning isn't convenient). |

## 4. Pairing flow

1. Dashboard operator clicks **"+ Add Phone Camera"** → `POST /api/phonecam/pair`.
2. Server generates an 8-char url-safe `pair_token`, stores it with a 10-minute
   expiry, detects the machine's LAN-facing IP (not `127.0.0.1`/`0.0.0.0`),
   and returns `{ join_url, qr_png_base64 }` where
   `join_url = http://<lan-ip>:<port>/phonecam/join/<pair_token>`.
3. Dashboard shows the QR code and the raw URL in a modal.
4. Phone scans QR (or the URL is typed manually) → opens
   `GET /phonecam/join/{pair_token}`, which 404s with a clear message if the
   token is expired/unknown.
5. Join page requests camera permission (`getUserMedia`), shows a live local
   preview, and offers an editable device name (default "Phone Camera").
6. On "Start Sharing": the page reads/creates a persistent `device_id` in
   `localStorage`, opens
   `WS /ws/phonecam/{pair_token}/{device_id}`, and starts sending JPEG frames.
7. Server-side, on WebSocket connect:
   - Validate `pair_token` is live and unexpired (404/close otherwise).
   - If `device_id` already maps to a camera entry (reconnect), reuse it.
   - Else, register a new entry via the existing
     `CAMERA_REGISTRY.upsert_camera({..., "url": f"phonecam://{device_id}"})`
     and call `MULTI_CAMERA_MANAGER.start_camera(cam_id)` — the tile appears
     in the live grid with no further action on the laptop.
8. Phone captures frames client-side (canvas/`ImageCapture`, ~10–12 fps,
   resized to cap bandwidth, e.g. max width 960px) and sends each as a binary
   WebSocket message (JPEG bytes). Server decodes with `cv2.imdecode` and
   writes into that device's `PhoneCamCapture` frame slot.

The `pair_token` is a "must have scanned the dashboard's QR on this network"
guard, not a full auth system — consistent with the rest of the project
(DroidCam URLs and the dashboard itself have no authentication today). A
token may be used to join more than one device within its 10-minute window
(so scanning once can add multiple phones), but expires after that window
regardless of use.

## 5. Reconnection & lifecycle

- Backgrounding the tab or a WiFi drop closes the WebSocket. The camera's
  status flips to `OFFLINE` (same status value already used for a dropped
  DroidCam/RTSP stream) rather than deleting the registry entry.
- `PhoneCamCapture.read_latest()` keeps serving the last frame for a short
  grace window (5s) after the socket drops, then reports "no frame" so the
  existing placeholder/offline rendering in `CameraPipelineWorker` takes over
  unchanged.
- Reopening the same join link on the same phone reuses the `device_id` from
  `localStorage`, so it resumes the same tile — no duplicate cameras pile up
  in `cameras.json`.
- Closing the tab entirely behaves the same as backgrounding at the protocol
  level (WebSocket close); the tile stays `OFFLINE` until removed from
  Settings like any other camera — phone cameras are not auto-deleted.

## 6. Error handling

- Camera permission denied on the phone → inline error on the join page, no
  WebSocket is opened.
- Expired/unknown `pair_token` → join page shows "Pairing link expired,
  generate a new QR code from the dashboard."; WebSocket connect with a bad
  token is rejected (close code, no registration side effects).
- A corrupt/undecodable frame (`cv2.imdecode` returns `None`) is dropped and
  logged; it does not crash the WebSocket handler or the pipeline worker —
  mirrors how `ZeroLagCapture` already tolerates bad reads from network
  streams.

## 7. Explicitly out of scope

Evaluated against the reference "Shared Perception" feature in the unrelated
`kaya-main` project and dropped:

- **Internet-facing relay / works-off-LAN** — this project's phone cameras
  are only reachable when phone and laptop share a WiFi network, same as the
  existing DroidCam/RTSP sources.
- **GPS/IMU telemetry, map view, fleet/geofence concepts** — irrelevant to a
  fixed-perimeter surveillance deployment; not built.
- **Postgres/Redis/Auth0/React Native** — none of this project's stack
  changes; the phone side is a single static mobile web page, no app.

## 8. Testing plan

- Unit tests for `PhoneCamCapture`: setting/reading the latest frame, the
  stale-frame timeout behavior, and for `PhoneDeviceRegistry`: token
  generation/expiry and device_id → camera-entry reuse.
- Manual end-to-end pass with a real phone on the same WiFi: scan QR, grant
  camera permission, confirm the tile appears in the live grid and runs
  through YOLOX/ByteTrack/ANPR/FRS, survives a background/foreground cycle
  (tile goes OFFLINE then LIVE again on the same tile), and is removable from
  Settings like any other camera.

## 9. New dependency

- `qrcode` (pure-Python QR PNG generation) added to `requirements.txt`.
  Pillow, already a dependency, is qrcode's only image-backend requirement.
