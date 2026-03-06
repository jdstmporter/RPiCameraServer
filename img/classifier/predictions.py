import collections

PredictionItem = collections.namedtuple('PredictionItem',['box','label','score','normalised'])

class Prediction:

    def __init__(self,prediction : dict):
        self.dict=prediction
        boxes=self.dict['boxes']
        labels=self.dict['labels']
        scores=self.dict['scores']
        zipped = zip(boxes,labels,scores)
        self.items = [PredictionItem(b,l,s,s) for b,l,s in zipped]

    def filter(self,labels=None,threshold=-1.0):
        f = [i for i in self.items if i.score>threshold]
        if labels is not None:
            f = [i for i in f if i.label in labels ]
        self.items=f
        return self

    def normalise(self):
        scores = [p.score for p in self.items]
        ma=max(*scores)
        mi=min(*scores)
        self.items=[PredictionItem(p.box,p.label,p.score,(p.score-mi)/(ma-mi)) for p in self.items]
        return self

    def __len__(self):
        return len(self.items)

    def __getattr__(self, idx):
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)


