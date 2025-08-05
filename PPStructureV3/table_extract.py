import cv2
import os
from ultralytics import YOLO
from paddleocr import PPStructureV3
import numpy as np
import time

# ✅ Load YOLO model
model = YOLO("../YOLO/best.pt")

# ✅ Initialize PaddleOCR table extractor
table_engine = PPStructureV3()

# ✅ Global table store
tables = []
notes = {} 

def extract_table_name(header, rows):
    """
    Try to extract a meaningful table name based on keywords.
    Fallback to 'unnamed'.
    """
    keywords = ['material', 'specification', 'pressure', 'dimension', 'datasheet', 'schedule']
    text = ' '.join(header or []) + ' ' + ' '.join([' '.join(row) for row in rows])
    text = text.lower()
    for k in keywords:
        if k in text:
            return k
    return "unnamed"


def process_cropped_table(img_crop, bbox):
    result = table_engine.predict(
        img_crop,
        det_kwargs={"use_gpu": False, "lang": "en"},
        structure_version="PP-StructureV3",
        use_layout=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=False,
        use_table=True,
        use_ocr=True,
        use_doc_preprocessing=True,
        use_structure_vqa=False
    )

    print(f"📄 Processed table with bbox {bbox} → result: {result}")

    print(result[0])
    return

    # 🚨 CASE 1: Proper table → result is dict
    if isinstance(result, dict):
        table_list = result.get("table_res_list", [])
        for table in table_list:
            print("=" * 25)
            print(table)
            header = table.get("header", [])
            rows = table.get("cell", [])

            combined_text = ' '.join(header or []) + ' ' + ' '.join([' '.join(row) for row in rows]).lower()
            if "note" in combined_text:
                notes[f"note_{len(notes)+1}"] = {
                    "bbox": bbox,
                    "text": combined_text.strip()
                }
                continue

            name = extract_table_name(header, rows)
            table_obj = {
                "name": name,
                "header": header,
                "rows": rows,
                "bbox": bbox
            }

            existing = next((t for t in tables if t["name"] == name), None)
            if existing:
                existing["rows"].extend(table_obj["rows"])
            else:
                tables.append(table_obj)

    # 🚨 CASE 2: Non-table (like "NOTES") → result is list
    elif isinstance(result, list):
        combined_text = ""
        for page in result:
            ocr_blocks = page.get("overall_ocr_res", []) if isinstance(page, dict) else []
            for block in ocr_blocks:
                if isinstance(block, dict):
                    combined_text += block.get("text", "") + " "
                elif isinstance(block, str):
                    combined_text += block + " "

        if "note" in combined_text.lower():
            notes[f"note_{len(notes)+1}"] = {
                "bbox": bbox,
                "text": combined_text.strip()
            }


    else:
        # Unknown structure; skip
        return



# Dummy warm-up image to fully preload models (avoid loading during prediction)
def warmup_engine(engine):
    dummy_img = np.ones((64, 64, 3), dtype=np.uint8) * 255
    try:
        engine.predict(dummy_img)
    except Exception:
        pass  # ignore errors — we just want to trigger loading

def run_pipeline(image_path):
    """
    Full pipeline: detect tables → crop → parse → structure.
    """
    print(f"🔍 Detecting tables in {image_path}...")
    start = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read {image_path}")
        return

    results = model(image)[0]

    print(f"📦 Detected {len(results.boxes)} tables.")

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image[y1:y2, x1:x2]
        bbox = [x1, y1, x2 - x1, y2 - y1]
        print(f"📏 Processing table at {bbox}...")
        process_cropped_table(crop, bbox)

    print("\n✅ Final structured tables:")
    for t in tables:
        print(f"\n📋 Table: {t['name']}")
        print(f"Header: {t['header']}")
        for row in t['rows']:
            print(row)

    end = time.time()
    print(f"⏱️ Processing time: {end - start:.2f} seconds")

if __name__ == "__main__":
    warmup_engine(table_engine)
    run_pipeline("VP-14780-D-2203-13-0001-A1.png")  # ← Replace with your image path
