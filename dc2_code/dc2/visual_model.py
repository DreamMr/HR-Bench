from transformers import AutoProcessor, CLIPModel, CLIPVisionModel
from PIL import Image

class VisualModel:
    def __init__(self,processor,visual_encoder):
        
        self.visual_model = visual_encoder
        self.processor = processor
        
    def get_vision_model_fea(self,image):
        """
        Parameters:
        image: numpy.array
        """
        
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image,return_tensors='pt')
        for k,v in inputs.items():
            v = v.cuda()
            inputs[k] = v
        feature = self.visual_model(**inputs).pooler_output
        return feature.detach().cpu().numpy()