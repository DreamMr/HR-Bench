import torch
from PIL import Image
from abc import abstractproperty
import os, sys
import os.path as osp
from transformers import AutoTokenizer, AutoModel
import time
import warnings
import traceback

class LLaVA:

    INSTALL_REQ = True

    def __init__(self, 
                 model_pth,
                 **kwargs): 
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as e:
            traceback.print_exc()
            warnings.warn("Please install llava before using LLaVA")
            sys.exit(-1)
            
        assert osp.exists(model_pth) or len(model_pth.split('/')) == 2
        
        model_name = get_model_name_from_path(model_pth)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_pth, 
                model_base=None, 
                model_name=model_name, 
                device='cpu', 
                device_map='cpu'
            )
        except Exception as e:
            print(e)
            if 'ShareGPT4V' in model_pth:
                import llava
                warnings.warn(
                    f'Please manually remove the encoder type check in {llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)
        
        self.model = self.model.cuda()
        self.conv_mode =  'llava_v1'

        kwargs_default = dict(do_sample=True, temperature=0.2, max_new_tokens=1024, top_p=0.5, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
    
    def generate(self, image_path, prompt):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        
        if isinstance(image_path,str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
            
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.batch_decode(output_ids,skip_special_tokens=True)[0].strip()
        return output
