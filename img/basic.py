

from classifier import ObjectDetector
from images import ImageData

img = ImageData.load('image.jpg')
detector = ObjectDetector()
detector(img)
prediction = detector[0].filter(labels=[1],threshold=0.32).normalise()

print(f'Found {len(prediction)}')
for p in prediction:
    box = p.box.numpy().astype(int)
    print(f'({box[0]}, {box[1]}),({box[2]}, {box[3]}) : {p.score} : {p.normalised}')

anno = ImageData('image.jpg')
font = ImageData.font('Arial.ttf', 18)
for p in prediction:
    box = p.box.numpy().astype(int)
    s = int(p.normalised*255)
    n = int(p.score*100.0)
    col = (255-s,s,0,31)
    anno.rectangle([(box[0],box[1]),(box[2],box[3])], fill=col,outline=(0,0,0))
    anno.text((box[0]+5,box[1]+5),f'p={n}',font=font,font_size=8,stroke_fill=(0,0,0))

anno.save('annotate2.jpg')






