import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from dc2.utils import check_zero, cluster, Template_prompt, extract_json_from_markdown, filter_location, wrapper_object_query, save_json, read_json
from dc2.visual_model import VisualModel
from dc2.llava import LLaVA
from dc2.lmdeploy_model import Lmdeploy
from transformers import AutoTokenizer, AutoModel
import traceback
import os

class DC2:
    def __init__(self,mllm_path,retrieval_path,llm_path=None):
        self.mllm_path = mllm_path
        self.llm_path = llm_path
        
        # initialize mllm
        self.mllm = LLaVA(model_pth=self.mllm_path)
        self.mllm.model.model
        self.llm = None
        # initialize vision model
        self.visual_encoder = VisualModel(self.mllm.image_processor,self.mllm.model.model.get_vision_tower().vision_tower)
        # retrieval
        self.retrieval_tokenizer = AutoTokenizer.from_pretrained(retrieval_path)
        self.retrieval_model = AutoModel.from_pretrained(retrieval_path).cuda()
            
    def forward_model(self,image,text,state='Caption'):
        prompt_template = Template_prompt[state]
        prompt = prompt_template.format(text)
        cur_image = image.astype(np.uint8)
        pil_image = Image.fromarray(cur_image)
        response = self.mllm.generate(pil_image,prompt)
        response = response.strip()
        print("Response: ",response)
        return response
    
    def attribute_extraction(self,caption):
        query_object_extraction = Template_prompt['Entity_Extraction'].format(caption)
        if self.llm is not None:
            response = self.llm.generate(query_object_extraction)
        else:
            response = self.mllm.generate(prompt=query_object_extraction)
            
        object_list_str = None
        try:
            object_list_str = extract_json_from_markdown(response)
            object_list = list(set(eval(object_list_str)['object_list']))
        except Exception as e:
            print(e)
            print("get response object list: {}, input: {}".format(object_list_str,caption))
            object_list = []
        
        return object_list

    def prune_merge(self,loc_dic,image,threshold=0.1):
        """
        Implementation of merge
        params:
            loc_dic: Dict: {"left_top":[left_top_x,left_top_y,height,width]}
        Return:
            prune_loc_list: [{'mode':single or union,'loc':[(left_top_x,left_top_y,height,width)]}]
        """
        global vision_model
        fea_list = []
        loc_list = []
        for loc_name,loc_cood_dict in loc_dic.items():
            loc_cood = loc_cood_dict['local_react']
            global_loc_cood = loc_cood_dict['global_react']
            cur_image = image[loc_cood[1]:loc_cood[1]+loc_cood[2],loc_cood[0]:loc_cood[0]+loc_cood[3]]
            fea = self.visual_encoder.get_vision_model_fea(cur_image)
            fea_list.append(fea.squeeze())
            loc_list.append([loc_name,loc_cood,global_loc_cood])
            
        # link 
        fea_list = np.array(fea_list)
        clusters = cluster(fea_list,threshold=threshold)
        cluster_dic = defaultdict(list)
        for i in range(len(clusters)):
            c = clusters[i]
            loc_name,loc_cood,global_loc_cood = loc_list[i]
            cluster_dic[c].append((loc_cood,global_loc_cood))
            
        prune_loc_list =  []
        for k,cood_list in cluster_dic.items():
            dic = {}
            if len(cood_list) > 1:
                mode = 'union'
            else:
                mode = 'single'
            dic['mode'] = mode
            dic['loc'] = cood_list
            prune_loc_list.append(dic)
        
        return prune_loc_list
    
    def wrapper_objct_list(self,caption):
        refine_template = Template_prompt['Refine_Caption']
        
        refine_caption_query = refine_template.format(caption)
        return refine_caption_query
                
    def non_leaf_node_run(self,image,sub_patch_info,react,sub_react,global_object_list,max_deep):
        # step 0: check zero
        if check_zero(np.copy(image)):
            return {"caption": "", 'objects_list':[]}
        # step 1: union sub node response
        caption_list = ''
        
        sub_obj_set = set()
        sub_loc_obj = defaultdict(list)
        for i,sub_patch_object_caption in enumerate(sub_patch_info):
            objects_list = sub_patch_object_caption.get('objects_list',set())
            caption = sub_patch_object_caption.get('caption',"")
            
            caption_list += f"{caption}\n"
            sub_obj_set.update(objects_list)
            
            for obj in objects_list:
                sub_loc_obj[obj].append(i)
            
        # step 2: generate precise caption
        refine_caption_query = self.wrapper_objct_list(caption_list)
        refine_caption = self.forward_model(image,refine_caption_query,state='Empty')
        objects_list = self.attribute_extraction(refine_caption)
        
        # step 3: refine global object
        if max_deep > 1:
            for obj in objects_list:
                if obj in sub_obj_set and len(sub_loc_obj[obj]) > 1:
                    dic = {'mode':'single','react':[(None,react)]}
                    global_object_list[obj].append(dic)
                elif obj in sub_obj_set and len(sub_loc_obj[obj]) == 1:
                    local_sub_react = sub_react[sub_loc_obj[obj][0]]
                    global_object_list[obj].append(local_sub_react)
        else:
            for obj in objects_list:
                dic = {'mode':'single','react':[(None,react)]}
                global_object_list[obj].append(dic)
            for i, sub_patch_object_caption in enumerate(sub_patch_info):
                objects_list = sub_patch_object_caption.get('objects_list',set())
                for obj in objects_list:
                    for loc_index in sub_loc_obj[obj]:
                        
                        local_sub_react = sub_react[loc_index]
                        global_object_list[obj].append(local_sub_react)
        
        return {'caption': refine_caption, 'objects_list':objects_list}
    
    def leaf_node_run(self,image,react):
        # step 0: check zero
        if check_zero(np.copy(image)):
            return {"caption": "", 'objects_list':[]}
        # step 1: LVLM provide caption
        caption = self.forward_model(image,'',state='Caption')
        
        # step 2: gpt3.5 extract object list
        objects_list = self.attribute_extraction(caption)
            
        return {'caption': caption, 'objects_list':objects_list}


    def run(self,image,min_size,global_rect,deep,max_deep,global_object_list,total_h,total_w,threshold):
        '''
        image: numpy.array
        min_size: minmum size
        global_rect: (left_top_x,left_top_y,height,width)
        '''
        h,w,_ = image.shape
        ans_list = []
        split_h = h // 2
        split_w = w // 2
        
        # leaf node
        if split_h < min_size or split_w < min_size or deep >= max_deep:
            # forward model
            res = self.leaf_node_run(image,global_rect)
            return res

        x,y = 0,0
        
        # divide
        direction = [[0,0],[0,1],[1,0],[1,1]] # top left, top right, bottom left, bottom right
        dir_loc = ['top left','top right','bottom left','bottom right']
        sub_react = {}
        loc_dic = {}
        for loc,dir in zip(dir_loc,direction):
            new_x = x + dir[0] * split_w
            new_y = y + dir[1] * split_h
            
            global_new_x = global_rect[0] + dir[0] * split_w
            global_new_y = global_rect[1] + dir[1] * split_h
            
            if new_x < 0 or new_y < 0 or new_x >= total_w or new_y >= total_h:
                continue
            
            loc_dic[loc] = {'global_react':[global_new_x,global_new_y,split_h,split_w],'local_react':[new_x,new_y,split_h,split_w]}
        
        # merge    
        prune_loc_dic_list = self.prune_merge(loc_dic,image,threshold=threshold)
        for i,dic in enumerate(prune_loc_dic_list):
            if dic['mode'] == 'single':
                assert len(dic['loc']) == 1, "Mode is single, the loc should be equal to 1"
                react = dic['loc'][0][0]
                cur_image = image[react[1]:react[1] + react[2],react[0]: react[0] + react[3]]
                tmp_ans = self.run(cur_image,min_size,dic['loc'][0][1],deep+1,max_deep,global_object_list,total_h,total_w,threshold)
                ans_list.append(tmp_ans)
                sub_react[i] = {'mode':dic['mode'],'react':dic['loc']}
            else:
                # mix up
                img_list = np.array([image[react[0][1]:react[0][1] + react[0][2],react[0][0]: react[0][0] + react[0][3]] for react in dic['loc']])
                mixup_img = np.mean(img_list,axis=0).astype(np.uint8)
                tmp_ans = self.run(mixup_img,min_size,dic['loc'][0][1],deep+1,max_deep,global_object_list,total_h,total_w,threshold)
                ans_list.append(tmp_ans)
                
                sub_react[i] = {'mode':dic['mode'],'react':dic['loc']}
        
        # conquer
        final_ans = self.non_leaf_node_run(image,ans_list,global_rect,sub_react,global_object_list,max_deep)
        
        return final_ans
    
    def run_main(self,image_path,question,recursive_layer,alpha,theta,visual_memory=None):
        image = Image.open(image_path)
        image_array = np.array(image)
        if os.path.exists('./visual_memory.json'):
            visual_memory = read_json('./visual_memory.json')
        if visual_memory == None:
            h,w,c = image_array.shape
            try:
                if self.llm is None:
                    if self.llm_path is not None:
                        self.llm = Lmdeploy(self.llm_path)
                global_object_list = defaultdict(list)
                self.run(image_array,168,(0,0,h,w),0,recursive_layer,global_object_list,h,w,theta)
                visual_memory = filter_location(global_object_list)
                save_json(visual_memory,'./visual_memory.json')
            except Exception as e:
                traceback.print_exc()
        
        # start inference
        wrapper_question = wrapper_object_query(visual_memory,question,self.mllm,image_array,self.retrieval_model,self.retrieval_tokenizer,alpha=alpha)
        
        response = self.mllm.generate(image_path,wrapper_question)
        return response,visual_memory