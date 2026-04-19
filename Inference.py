import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
import cv2
import numpy as np

def clear_vram():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ── STEP 1: VLM PLANNING ──────────────────────────────────────────────
def run_planner(image_path, instruction):
    print("\n[1/3] running Qwen2-VL planner...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256*28*28,
        max_pixels=512*28*28
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
            {"type": "text", "text": f"""You are a robot action planner.
Given a workspace image and an instruction, output a JSON action plan.

Example 1:
instruction: "pick up the cup and place it on the notebook"
output: {{"actions": [{{"type": "pick", "object": "cup"}}, {{"type": "place", "object": "notebook"}}], "place_mode": "single"}}

Example 2:
instruction: "place the pen between the cup and the book"
output: {{"actions": [{{"type": "pick", "object": "pen"}}, {{"type": "reference", "object": "cup"}}, {{"type": "reference", "object": "book"}}], "place_mode": "between_references"}}

instruction: "{instruction}"
output:"""}
        ]
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt"
    ).to("cuda")

    output = model.generate(**inputs, max_new_tokens=300)
    result = processor.decode(output[0], skip_special_tokens=True)
    print(f"    FULL RESULT: {result}")  # add this


    # finds first { after "assistant\n"
    assistant_idx = result.rfind("assistant")
    json_start = result.find("{", assistant_idx)
    json_end = result.find("\n", json_start)
    if json_end == -1:
        json_end = len(result)
    raw_json = result[json_start:json_end].strip()
    print(f"    raw output: {raw_json}")

    try:
        plan = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"    JSON parse failed: {e}")
        print(f"    raw output was: {raw_json}")
        exit(1)

    print(f"    plan: {plan}")

    del model, processor, inputs, output
    clear_vram()

    return plan

def fallback_parse(result, instruction):
    """if model ignores schema, extract objects manually"""
    import re
    objects = re.findall(r'"object"\s*:\s*"([^"]+)"', result)
    types = re.findall(r'"type"\s*:\s*"([^"]+)"', result)

    if objects and types:
        actions = [{"type": t, "object": o} for t, o in zip(types, objects)]
        place_mode = "between_references" if "reference" in types else "single"
        return {"actions": actions, "place_mode": place_mode}

    # last resort — just pull nouns from instruction
    print("    fallback: extracting from instruction directly")
    words = instruction.lower().split()
    pick_obj = words[words.index("up") + 1] if "up" in words else "object"
    return {
        "actions": [
            {"type": "pick", "object": pick_obj},
        ],
        "place_mode": "single"
    }

# ── STEP 2: DINO DETECTION ────────────────────────────────────────────
def run_detector(image_path, object_name):
    print(f"\n    detecting '{object_name}'...")
    from groundingdino.util.inference import load_model, load_image, predict

    model = load_model(
        "/home/hrigved/miniconda3/envs/vla/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "../phase2/weights/groundingdino_swint_ogc.pth"
    )
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=object_name,
        box_threshold=0.35,
        text_threshold=0.25
    )

    if len(boxes) == 0:
        print(f"    no detection found for '{object_name}'")
        del model
        clear_vram()
        return None, None

    best_idx = logits.argmax()
    best_box = boxes[best_idx]
    print(f"    found '{phrases[best_idx]}' confidence: {logits[best_idx]:.2f}")

    del model
    clear_vram()

    return best_box, image_source

# ── STEP 3: SAM SEGMENTATION ──────────────────────────────────────────
def run_segmenter(image_source, box):
    print(f"    segmenting...")
    from segment_anything import sam_model_registry, SamPredictor

    sam = sam_model_registry["vit_b"](checkpoint="../phase2/weights/sam_vit_b_01ec64.pth")
    sam.to("cuda")
    predictor = SamPredictor(sam)

    image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    h, w, _ = image_source.shape
    box_pixel = torch.tensor([
        (box[0] - box[2]/2) * w,
        (box[1] - box[3]/2) * h,
        (box[0] + box[2]/2) * w,
        (box[1] + box[3]/2) * h,
    ])

    masks, scores, _ = predictor.predict(
        box=box_pixel.numpy(),
        multimask_output=False
    )

    mask = masks[0]
    ys, xs = np.where(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    print(f"    centroid: ({cx}, {cy})  mask pixels: {mask.sum()}")

    del sam, predictor
    clear_vram()

    return mask, cx, cy

# ── STEP 4: VISUALIZE ─────────────────────────────────────────────────
def draw_label(canvas, label, cx, cy, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    cv2.rectangle(canvas, (cx-60-pad, cy-25-th-pad), (cx-60+tw+pad, cy-25+baseline+pad), (20,20,20), -1)
    cv2.putText(canvas, label, (cx-60, cy-25), font, scale, (255,255,255), thickness)
    cv2.line(canvas, (cx-60, cy-25+baseline), (cx-60+tw, cy-25+baseline), color, 2)

def visualize_multi(image_path, results, place_mode, instruction):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    output = image_rgb.copy()
    task_space = np.zeros_like(image_rgb)

    type_colors = {
        "pick":      (0,   200, 100),
        "place":     (0,   100, 255),
        "reference": (255, 165, 0),
    }

    for obj_name, data in results.items():
        if data["mask"] is None:
            continue

        mask   = data["mask"]
        cx, cy = data["cx"], data["cy"]
        color  = type_colors.get(data["type"], (255, 255, 255))
        label  = f"{data['type'].upper()}: {obj_name}"

        # --- visual space overlay ---
        output[mask] = (output[mask] * 0.3 + np.array(color) * 0.7).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 3)
        cv2.circle(output, (cx, cy), 16, (255, 255, 255), -1)
        cv2.line(output, (cx-40, cy), (cx+40, cy), (255, 255, 0), 2)
        cv2.line(output, (cx, cy-40), (cx, cy+40), (255, 255, 0), 2)
        draw_label(output, label, cx, cy, color)

        # --- task space overlay ---
        task_space[mask] = color
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(task_space, contours, -1, (255, 255, 255), 3)
        cv2.circle(task_space, (cx, cy), 16, (255, 255, 255), -1)
        cv2.line(task_space, (cx-40, cy), (cx+40, cy), (255, 255, 0), 2)
        cv2.line(task_space, (cx, cy-40), (cx, cy+40), (255, 255, 0), 2)
        draw_label(task_space, label, cx, cy, color)

    # --- between references midpoint ---
    if place_mode == "between_references":
        refs = [d for d in results.values() if d["type"] == "reference" and d["cx"] is not None]
        if len(refs) >= 2:
            mid_x = int(sum(r["cx"] for r in refs) / len(refs))
            mid_y = int(sum(r["cy"] for r in refs) / len(refs))
            pts = np.array([[mid_x, mid_y-40],[mid_x+40, mid_y],[mid_x, mid_y+40],[mid_x-40, mid_y]])
            for canvas in [output, task_space]:
                cv2.polylines(canvas, [pts], True, (255, 165, 0), 2)
                cv2.putText(canvas, "PLACE HERE", (mid_x - 45, mid_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 4)
                cv2.putText(canvas, "PLACE HERE", (mid_x - 45, mid_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 165, 0), 2)
            print(f"    midpoint place target: ({mid_x}, {mid_y})")

    # --- instruction banner ---
    cv2.putText(output, instruction, (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
    cv2.putText(output, instruction, (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)

    # --- save both ---
    cv2.imwrite("pipeline_result.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    cv2.imwrite("task_space.png",      cv2.cvtColor(task_space, cv2.COLOR_RGB2BGR))
    print("    saved pipeline_result.png and task_space.png")

    # --- side by side ---
    pipeline_img   = cv2.imread("pipeline_result.png")
    taskspace_img  = cv2.imread("task_space.png")
    h1, w1 = pipeline_img.shape[:2]
    h2, w2 = taskspace_img.shape[:2]
    target_h = min(h1, h2, 500)

    pipeline_resized  = cv2.resize(pipeline_img,  (int(w1 * target_h / h1), target_h))
    taskspace_resized = cv2.resize(taskspace_img, (int(w2 * target_h / h2), target_h))

    def add_banner(img, text):
        banner = np.zeros((36, img.shape[1], 3), dtype=np.uint8)
        banner[:] = (28, 28, 28)
        cv2.putText(banner, text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return np.vstack([banner, img])

    pipeline_resized  = add_banner(pipeline_resized,  "visual space")
    taskspace_resized = add_banner(taskspace_resized, "task space")

    divider = np.full((pipeline_resized.shape[0], 4, 3), (50, 50, 50), dtype=np.uint8)
    combined = np.hstack([pipeline_resized, divider, taskspace_resized])
    cv2.imwrite("combined_view.png", combined)
    screen_w = 1920
    max_w = screen_w - 100
    scale_factor = min(max_w / combined.shape[1], 1.0)
    display = cv2.resize(combined, (int(combined.shape[1]*scale_factor), int(combined.shape[0]*scale_factor))) if scale_factor < 1.0 else combined
    cv2.namedWindow("robot perception", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("robot perception", display.shape[1], display.shape[0])
    cv2.imshow("robot perception", display)
    print("    press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ── MAIN ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    #IMAGE       = input("image path: ").strip()
    DEFAULT_IMAGE = "cute-messy-workplace-with-stationary-coffee-flowers.jpg"
    IMAGE = input(f"image path (enter for default '{DEFAULT_IMAGE}'): ").strip()
    IMAGE = IMAGE if IMAGE else DEFAULT_IMAGE
    INSTRUCTION = input("instruction: ").strip()

    print(f"\ninstruction : {INSTRUCTION}")
    print(f"image       : {IMAGE}")

    # step 1 — plan
    plan       = run_planner(IMAGE, INSTRUCTION)
    actions    = plan.get("actions", [])
    place_mode = plan.get("place_mode", "single")

    # step 2+3 — detect and segment each object
    print(f"\n[2/3] detecting and segmenting {len(actions)} objects...")

    results = {}
    for action in actions:
        obj_type = action["type"]
        obj_name = action["object"]

        box, image_source = run_detector(IMAGE, obj_name)

        if box is not None:
            mask, cx, cy = run_segmenter(image_source, box)
            results[obj_name] = {
                "type": obj_type,
                "mask": mask,
                "cx":   cx,
                "cy":   cy,
            }
        else:
            results[obj_name] = {
                "type": obj_type,
                "mask": None,
                "cx":   None,
                "cy":   None,
            }

    # step 4 — visualize
    print("\n[3/3] visualizing...")
    visualize_multi(IMAGE, results, place_mode, INSTRUCTION)

    # final summary
    print("\n" + "="*50)
    print("PIPELINE RESULT")
    print("="*50)
    print(f"instruction : {INSTRUCTION}")
    print(f"place mode  : {place_mode}")
    for obj_name, data in results.items():
        status = f"({data['cx']}, {data['cy']})" if data['cx'] else "NOT FOUND"
        print(f"{data['type']:12s}: {obj_name:20s} {status}")
    print("="*50)