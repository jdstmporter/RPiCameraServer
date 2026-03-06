from PIL import Image, ImageDraw, ImageFont
from torchvision.io import decode_image

class ImageData:

    @classmethod
    def load(cls,name: str):
        return decode_image(name)

    def __init__(self,infile : str):
        self.img=Image.open(infile)
        self.draw=ImageDraw.Draw(self.img, 'RGBA')

    def save(self, named=None):
        if named is None:
            self.img.save()
        else:
            self.img.save(named)

    @classmethod
    def font(cls,font : str,size : int):
        try:
            return ImageFont.truetype(font=font, size=size)
        except OSError:
            print(f'Cannot load {font}: defaulting')
            return ImageFont.load_default()

    def __getattr__(self, item):
        return getattr(self.draw,item)

class ImageColour:
    def __init__(self,r = 0, g = 0, b = 0, a = 0):
        self.rgba=(r,g,b,a)

    def __call__(self):
        return self.rgba

    def __add__(self, other):
        x=self.rgba+other.rgba
        return ImageColour(*x)

    def __mul__(self, other):
        x=self.rgba*other
        return ImageColour(*x)



