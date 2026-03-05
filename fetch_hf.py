from transformers import AutoConfig
import json

models = [
    "HotJellyBean/skin-disease-classifier",
    "Jayanth2002/dinov2-base-finetuned-SkinDisease",
    "Anwarkh1/Skin_Cancer-Image_Classification",
    "Tanishq77/skin-condition-classifier"
]

results = {}
for m in models:
   try:
       config = AutoConfig.from_pretrained(m)
       results[m] = config.id2label
   except Exception as e:
       results[m] = str(e)

with open('hf_labels.json', 'w') as f:
   json.dump(results, f, indent=2)
