from ultralytics import YOLO

model = YOLO("best.pt")

res = model.predict("Photoes/PXL_20251227_044327197.jpg")
probs = res[0].probs  # 包含 top1/top5 等資訊

# 取 Top-5 類別與機率
top5_idx = probs.top5
top5_conf = probs.top5conf
names = res[0].names

candidates = [(names[i], float(c)) for i, c in zip(top5_idx, top5_conf)]
print(candidates)
