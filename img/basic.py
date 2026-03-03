import torch
from torchvision.io import decode_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw


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
confidence_threshold = 0.5
person_indices = (labels == person_class_id) & (scores > confidence_threshold)
person_boxes = boxes[person_indices]

print(f'Found {len(person_boxes)}')
for box in person_boxes:
    box = box.numpy().astype(int)
    print(f'({box[0]}, {box[1]}),({box[2]}, {box[3]})')

with Image.open('image.jpg') as annotate:
    draw = ImageDraw.Draw(annotate,'RGBA')
    for box in person_boxes:
        box = box.numpy().astype(int)
        draw.rectangle([(box[0],box[1]),(box[2],box[3])], fill=(255,0,0,127),outline=(127,0,0))

    annotate.save('annotated.jpg')






