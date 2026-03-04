import math

import numpy
import torch
from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont

img = decode_image("image.jpg")

# Step 1: Initialize model with the best available weights
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
with torch.no_grad():
    predictions = model(batch)

# Process the predictions
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

person_class_id = 1
confidence_threshold = 0.3
person_indices = (labels == person_class_id) & (scores > confidence_threshold)
person_boxes = boxes[person_indices]
person_scores = scores[person_indices]

ma = torch.max(person_scores)
mi = torch.min(person_scores)
scores_norm = (person_scores - mi)/(ma - mi)

zipped = list(zip(person_boxes,person_scores,scores_norm))

print(f'Found {len(zipped)}')
for bs in zipped:
    box = bs[0].numpy().astype(int)
    print(f'({box[0]}, {box[1]}),({box[2]}, {box[3]}) : {bs[1]}, {bs[2]}')

probs = [ bs[1]*100.0 for bs in zipped]
histo, edges = numpy.histogram(probs, bins=10,range=(0,100))
for n in range(len(histo)):
    h=histo[n]
    if h>0 :
        print(f'{edges[n]}: {h}')

with Image.open('image.jpg') as annotate:
    font = ImageFont.truetype('Arial.ttf', 18)
    draw = ImageDraw.Draw(annotate,'RGBA')
    for bs in zipped:
        box = bs[0].numpy().astype(int)
        p = int(round(float(bs[2]),2) *255)
        prob = int(bs[1]*100)

        col = (255-p,p,0,127)
        draw.rectangle([(box[0],box[1]),(box[2],box[3])], fill=col,outline=(0,0,0))
        draw.text((box[0]+5,box[1]-5),f'p={prob/100.0}',font=font,font_size=8,  stroke_fill=(0,0,0))

    annotate.save('annotate2.jpg')






