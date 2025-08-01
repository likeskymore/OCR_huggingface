import os
import time
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from tqdm import tqdm
import re

# === CONFIG ===
model_name = "nanonets/Nanonets-OCR-s"
image_folder = "./datatest"
output_folder = "./results_txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# === LOAD MODEL & PROCESSOR ===
print(f"📦 Loading model: {model_name}")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(model_name)

def clean_nanonets_output(text: str) -> str:
    # Remove everything before the assistant's response
    match = re.search(r"assistant\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# === INFERENCE FUNCTION ===
def ocr_with_nanonets(image: Image.Image):
    prompt = """Extract the text from the above document as if you were reading it naturally. \
Return the tables in HTML format. Return LaTeX equations where present. Use <img></img> tags \
for images with descriptions, <signature></signature> for signatures, <watermark></watermark> \
for watermark text, and ☐/☑/☒ for checkbox/radio states."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = processor(text=[inputs], images=[image], padding=True, return_tensors="pt").to(device)
    generated_ids = model.generate(**processed, max_new_tokens=4096)
    raw_output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return clean_nanonets_output(raw_output)

# === RUN ON FOLDER ===
print(f"🔍 Processing folder: {image_folder}")
image_files = [fn for fn in os.listdir(image_folder) if fn.lower().endswith((".png", ".jpg", ".jpeg"))]

start_time = time.time()

for fn in tqdm(image_files, desc="📝 OCR Progress", unit="img"):
    path = os.path.join(image_folder, fn)
    try:
        image = Image.open(path).convert("RGB")
        text = ocr_with_nanonets(image)

        # Save to text file
        base_name = os.path.splitext(fn)[0]
        out_path = os.path.join(output_folder, f"{base_name}_result.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    except Exception as e:
        print(f"{fn} → ERROR: {e}")

end_time = time.time()
elapsed_sec = end_time - start_time
minutes, seconds = divmod(elapsed_sec, 60)

# === SUMMARY ===
print("\n✅ Finished processing.")
print(f"🖼️ Total images: {len(image_files)}")
print(f"⏱️ Time elapsed: {int(minutes)} minutes {int(seconds)} seconds")
