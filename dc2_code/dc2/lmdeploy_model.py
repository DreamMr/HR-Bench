from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
import torch


class Lmdeploy:
    def __init__(self,
                 model_pth,
                 tp=None,
                 **kwargs):
        
        if tp == None:
            tp = torch.cuda.device_count()
        assert tp > 0, "The TP: {} is equal to 0 !".format(tp)
        
        backend_config = TurbomindEngineConfig(tp=tp,cache_max_entry_count=0.4)
        self.model = pipeline(model_pth,backend_config=backend_config)
        kwargs_default = dict(do_sample=True, top_k=40, temperature=0.2, max_new_tokens=1024, top_p=0.5, num_beams=1,repetition_penalty=1.01)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
    
    def generate(self,prompt):
        
        top_p = self.kwargs.get('top_p',0.5)
        top_k = self.kwargs.get('top_k',40)
        temperature = self.kwargs.get('temperature',0.2)
        repetition_penalty = self.kwargs.get('repetition_penalty',1.01)
        max_new_tokens=self.kwargs.get('max_new_tokens')
        
        gen_config = GenerationConfig(top_p=top_p,
                                      top_k=top_k,
                                      temperature=temperature,
                                      repetition_penalty=repetition_penalty,
                                      max_new_tokens=max_new_tokens)
        
        response = self.model([prompt],gen_config=gen_config)[0].text.strip()
        
        return response
    