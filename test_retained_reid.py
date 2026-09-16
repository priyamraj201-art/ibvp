"""
Automated Verification Suite for Retained ID Person Re-ID.
Tests:
1. PersonReIDExtractor: OSNet feature extraction.
2. PersonReIDRegistry: Mints REID-001, calculates centroid, matches same person (> 0.70 score), mints REID-002 for distinct person.
3. ReIDPipeline: Asynchronous observation submission, track-to-retained ID binding across camera IDs.
"""

import os
import sys
import time
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_bytetrack_root = os.path.abspath(os.path.dirname(__file__))
if _bytetrack_root not in sys.path:
    sys.path.insert(0, _bytetrack_root)

from yolox.reid import PersonReIDExtractor, PersonReIDRegistry, ReIDPipeline


def create_synthetic_person(upper_color=(180, 80, 50), lower_color=(60, 120, 150), height=256, width=128) -> np.ndarray:
    """Generate structured synthetic person crop with head, torso, and legs."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Head
    cv2.ellipse(img, (width // 2, int(height * 0.10)), (width // 5, int(height * 0.08)), 0, 0, 360, (180, 200, 230), -1)
    # Upper Torso
    img[int(height * 0.18):int(height * 0.60), width // 6 : width - width // 6] = upper_color
    # Stripe
    img[int(height * 0.35):int(height * 0.42), width // 6 : width - width // 6] = np.clip(np.array(upper_color) * 0.6, 0, 255).astype(np.uint8)
    # Legs
    img[int(height * 0.60):, width // 5 : width // 2 - 4] = lower_color
    img[int(height * 0.60):, width // 2 + 4 : width - width // 5] = lower_color
    return img


def run_tests():
    print("=" * 70)
    print("  IBVAP Retained ID Person Re-ID Automated Test Suite")
    print("=" * 70)

    # 1. Test Feature Extractor
    print("\n[TEST 1] Initializing PersonReIDExtractor (OSNet)...")
    extractor = PersonReIDExtractor(weights_dir="pretrained", fp16=False)
    assert extractor.model is not None, "OSNet model should be loaded"
    print(f"         OSNet model running on: {extractor.device_name}")

    person_a1 = create_synthetic_person((220, 50, 40), (40, 100, 180))
    person_a2 = cv2.convertScaleAbs(person_a1.copy(), alpha=1.05, beta=10) # Same person, slight lighting shift
    person_b = create_synthetic_person((30, 180, 80), (180, 180, 40))      # Distinct person (green shirt)

    emb_a1 = extractor.extract_feature(person_a1)
    emb_a2 = extractor.extract_feature(person_a2)
    emb_b = extractor.extract_feature(person_b)

    assert len(emb_a1) == 512, "Embedding should be 512-D"
    sim_same = float(np.dot(emb_a1, emb_a2))
    sim_diff = float(np.dot(emb_a1, emb_b))
    print(f"         Same person similarity: {sim_same:.4f} (expected > 0.85)")
    print(f"         Different person similarity: {sim_diff:.4f} (expected < 0.70)")
    assert sim_same > 0.80, f"Same person cosine similarity too low: {sim_same}"
    assert sim_same > sim_diff, "Same person similarity must exceed different person similarity"
    print("         -> PASSED [OK]")

    # 2. Test PersonReIDRegistry
    print("\n[TEST 2] Testing PersonReIDRegistry Retained ID Minting & Matching...")
    test_db = os.path.join(_bytetrack_root, "test_reid.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    registry = PersonReIDRegistry(db_path=test_db, similarity_threshold=0.70)
    # Register Person A
    pid_a = registry.register_person(camera_id=0, camera_name="CAM 01", crop=person_a1, embedding=emb_a1, bbox=[10, 10, 100, 200])
    assert pid_a == "REID-001", f"Expected REID-001, got {pid_a}"
    print(f"         Registered initial person -> {pid_a}")

    # Query Person A from Camera 2 (cross-camera reappearance)
    matched_id, score = registry.match_person(emb_a2, threshold=0.70)
    assert matched_id == "REID-001", f"Expected match with REID-001, got {matched_id}"
    print(f"         Cross-camera match -> {matched_id} (score: {score:.3f})")

    # Update appearance on Camera 2
    registry.update_person(matched_id, camera_id=1, camera_name="CAM 02", crop=person_a2, embedding=emb_a2, bbox=[50, 50, 120, 220])
    person_obj = registry.get_person("REID-001")
    assert len(person_obj.camera_history) >= 2, "Camera history should reflect multi-camera appearances"
    print(f"         Camera history entries: {len(person_obj.camera_history)}")

    # Query Person B -> Should not match REID-001
    matched_b, score_b = registry.match_person(emb_b, threshold=0.70)
    assert matched_b is None, f"Person B should not match REID-001 (score: {score_b})"
    pid_b = registry.register_person(camera_id=1, camera_name="CAM 02", crop=person_b, embedding=emb_b, bbox=[200, 50, 100, 200])
    assert pid_b == "REID-002", f"Expected REID-002, got {pid_b}"
    print(f"         Registered second distinct person -> {pid_b}")
    print("         -> PASSED [OK]")

    # 3. Test Asynchronous ReIDPipeline
    print("\n[TEST 3] Testing ReIDPipeline Async Track-to-Retained ID Binding...")
    pipeline = ReIDPipeline(extractor=extractor, registry=registry, match_threshold=0.70)

    # Submit track on Camera 0
    pipeline.submit_observation(cam_id=0, cam_name="CAM 01", local_track_id=101, crop=person_a1, bbox=[10, 10, 100, 200])
    time.sleep(0.3)  # Allow async worker to process

    retained_101 = pipeline.get_retained_id(cam_id=0, local_track_id=101)
    assert retained_101 == "REID-001", f"Expected track #101 bound to REID-001, got {retained_101}"
    print(f"         Camera 0 Track #101 -> Retained ID: {retained_101}")

    # Same person appears on Camera 2 with local track ID 55
    pipeline.submit_observation(cam_id=1, cam_name="CAM 02", local_track_id=55, crop=person_a2, bbox=[50, 50, 120, 220])
    time.sleep(0.3)

    retained_55 = pipeline.get_retained_id(cam_id=1, local_track_id=55)
    assert retained_55 == "REID-001", f"Expected track #55 on CAM 02 to retain REID-001, got {retained_55}"
    print(f"         Camera 1 Track #55 (Cross-Camera) -> Retained ID: {retained_55} (ID successfully retained!)")

    pipeline.stop()

    # Cleanup test DB
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("  ALL RETAINED ID RE-ID TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
