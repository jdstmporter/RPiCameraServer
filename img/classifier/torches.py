from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch import no_grad
from .predictions import Prediction

class ObjectDetector:

    def __init__(self):
        self.weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1

    def __call__(self,*images):
            self.predictions=[]
            model = fasterrcnn_resnet50_fpn(weights=self.weights)
            model.eval()
            preprocess = self.weights.transforms()
            for image in images:
                batch = preprocess(image).unsqueeze(0)
                with no_grad():
                    prediction = model(batch)
                self.predictions.append(Prediction(prediction[0]))

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self,index):
        return self.predictions[index]

    def __iter__(self):
        return iter(self.predictions)




