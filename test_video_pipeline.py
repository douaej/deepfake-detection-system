"""
Diagnostic complet du pipeline vidéo.
Lance : python test_video_pipeline.py --video TON_VIDEO.mp4
"""
import sys, argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
args = parser.parse_args()

sys.path.insert(0, '.')
from config import CONFIG
from src.detector import DeepfakeDetector
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

detector = DeepfakeDetector(device=CONFIG.device, load_models=True)

cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\n=== VIDÉO : {args.video} | {total} frames ===\n")

# 1. Afficher la structure du dict face
print("=== STRUCTURE DICT FACE ===")
for idx in range(0, min(total, 30)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.image_processor.detect_faces(rgb)
    if faces:
        print(f"  Clés : {list(faces[0].keys())}")
        print(f"  Vals : {faces[0]}")
        break
cap.release()

# 2. Inférence sur 5 frames
def get_patch(image_rgb, face):
    keys = list(face.keys())
    try:
        if 'bbox' in face:
            x1,y1,x2,y2 = [int(v) for v in face['bbox']]
        elif 'x1' in face:
            x1,y1,x2,y2 = int(face['x1']),int(face['y1']),int(face['x2']),int(face['y2'])
        elif 'xmin' in face:
            x1,y1,x2,y2 = int(face['xmin']),int(face['ymin']),int(face['xmax']),int(face['ymax'])
        else:
            vals = [v for v in face.values() if isinstance(v,(int,float))]
            x1,y1,x2,y2 = int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3])
        h,w = image_rgb.shape[:2]
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        patch = image_rgb[y1:y2, x1:x2]
        return patch if patch.size > 0 else image_rgb
    except:
        return image_rgb

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

cap = cv2.VideoCapture(args.video)
print(f"\n=== INFÉRENCE BRUTE 5 FRAMES ===")
for idx in [int(total*i/5) for i in range(5)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret: continue
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.image_processor.detect_faces(rgb)
    patch = get_patch(rgb, faces[0]) if faces else cv2.resize(rgb,(224,224))
    tensor = transform(Image.fromarray(patch)).unsqueeze(0)
    infer = getattr(detector,'_model_efficientnet_infer', detector.model_efficientnet)
    infer.eval()
    with torch.no_grad():
        probs = F.softmax(infer(tensor), dim=1)
    p0,p1 = probs[0,0].item(), probs[0,1].item()
    pred,conf = detector._infer_efficientnet(tensor)
    label = 'FAKE' if pred==0 else 'REAL'
    print(f"  Frame {idx:4d} : prob[0(FAKE)]={p0:.4f}  prob[1(REAL)]={p1:.4f}  pred={pred}  → {label}")
cap.release()

# 3. Résultat final
print(f"\n=== RÉSULTAT FINAL detect_video() ===")
result = detector.detect_video(args.video)
print(f"  overall_result : {result.get('overall_result')}")
print(f"  avg_confidence : {result.get('avg_confidence'):.4f}")
s = result.get('stats',{})
print(f"  fake_ratio     : {s.get('fake_ratio','N/A')}")
print(f"  fake_count     : {s.get('fake_count','N/A')}")
print(f"  num_frames     : {s.get('num_frames','N/A')}")