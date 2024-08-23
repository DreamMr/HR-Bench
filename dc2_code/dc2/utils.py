import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
import re
from collections import defaultdict
from PIL import Image
import json
import io
import base64

def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f'Saved to {file}.')
    return

def check_zero(image):
    return not(np.any(image))

def cluster(data,threshold):
    # perform clustering
    '''
    ``Y = pdist(X, 'cosine')``

    Computes the cosine distance between vectors u and v,

    .. math::

        1 - \\frac{u \\cdot v}
                {{\\|u\\|}_2 {\\|v\\|}_2}

    where :math:`\\|*\\|_2` is the 2-norm of its argument ``*``, and
    :math:`u \\cdot v` is the dot product of ``u`` and ``v``.
    '''
    Z = linkage(data,method='average',metric='cosine')
    clusters = fcluster(Z, threshold, criterion='distance')
    return clusters

def extract_json_from_markdown(markdown_text):
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)
    match = pattern.search(markdown_text)

    if match:
        res = match.group(1)
    else:
        res = markdown_text
        
    return res

def filter_location(object_dict):
    filtered_dic = defaultdict(list)
    
    for obj_name,loc_list in object_dict.items():
        area_list = []
        garbage_loc_set = set()
        for loc in loc_list:
            if loc['mode'] == 'single':
                width = loc['react'][0][1][2]
                height = loc['react'][0][1][3]
                area = width * height
            else:
                width = loc['react'][0][1][2]
                height = loc['react'][0][1][3]
                single_area = width * height
                sz = len(loc['react'])
                if sz == 3:
                    sz = 4
                area = sz *  single_area
            area_list.append((area,loc))
        
        sorted_area_list = sorted(area_list,key=lambda x: x[0],reverse=True)
        def get_leftupper_rightbottom(cood_list):
            top_left = (float('inf'),float('inf'))
            for point in cood_list:
                if point[1]<top_left[1]:
                    top_left = point
                elif point[1] == top_left[1] and point[0] < top_left[0]:
                    top_left = point
            return top_left
        for i in range(len(sorted_area_list)):
            cur_area = sorted_area_list[i][0]
            cur_loc_ori = sorted_area_list[i][1]['react']
            sz = len(cur_loc_ori)
            if sz == 3:
                sz = 4
            cood_list = [(c[1][0],c[1][1]) for c in cur_loc_ori]
            top_left = get_leftupper_rightbottom(cood_list)
            cur_loc = [top_left[0],top_left[1],cur_loc_ori[0][1][2]*sz,cur_loc_ori[0][1][3]*sz]
            
            
            if i in garbage_loc_set:
                continue
            
            for j in range(i+1,len(sorted_area_list)):
                j_area = sorted_area_list[j][0]
                j_loc_dict = sorted_area_list[j][1]
                if j_loc_dict['mode'] == 'single':
                    j_loc = j_loc_dict['react'][0][1]
                else:
                    cood_list = [(c[1][0],c[1][1]) for c in j_loc_dict['react']]
                    sz = len(cood_list)
                    if sz == 3:
                        sz = 4
                    
                    top_left = get_leftupper_rightbottom(cood_list)
                    j_loc = [top_left[0],top_left[1],j_loc_dict['react'][0][1][2]*sz,j_loc_dict['react'][0][1][3]*sz]
                if cur_loc[0] <= j_loc[0] and cur_loc[1] <= j_loc[1] and cur_loc[0] + cur_loc[3] >= j_loc[0] + j_loc[3] and cur_loc[1] + cur_loc[2] >= j_loc[1] + j_loc[2]:
                        garbage_loc_set.add(j)
                    
        
        for index in range(len(loc_list)):
            if index in garbage_loc_set:
                continue
            
            filtered_dic[obj_name].append(loc_list[index])
    return filtered_dic

def _extract_embedding(text,tokenizer,model):
    if not isinstance(text,str):
        raise TypeError("{} is not str, found the type is {}".format(
            text, type(text)
        ))
        
    inputs = tokenizer(text,truncation=True,return_tensors='pt',max_length=512)
    for k,v in inputs.items():
        inputs[k] = v.cuda()
    
    outputs = model(**inputs)
    
    def mean_pooling(token_embeddings,mask):
        token_embeddings = token_embeddings.masked_fill(~mask[...,None].bool(),0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[...,None]
        return sentence_embeddings
    
    embeddings = mean_pooling(outputs[0],inputs['attention_mask'])
    
    return embeddings.detach()

def search(query_embedding,objects_embedding,theta=0.3,k=1):
    distance_func = torch.nn.CosineSimilarity(dim=1)
    scores = distance_func(query_embedding,objects_embedding)
    mask = torch.zeros_like(scores)
    mask[scores >= theta] = 1
    scores = scores * mask
    k = min(k,len(scores))
    values,indices = scores.topk(k,dim=0,largest=True,sorted=True)
    values_list = values.squeeze().detach().cpu().numpy().tolist()
    indices_list = indices.squeeze().detach().cpu().numpy().tolist()
    if isinstance(values_list,float):
        values_list = [values_list]
    if isinstance(indices_list,int):
        indices_list = [indices_list]
    res_list = [[idx,val] for idx, val in zip(indices_list,values_list)]
    ans = []
    for idx,val in res_list:
        if val == 0:
            break
        ans.append(idx)
    return ans

def wrapper_object_query(objects_dict,query,model,image_array,retrieval_model,retrieval_tokenizer,alpha=0.5):
    
    OBJ_PROMPT = """{}\n\n"""
    ATTRIBUTE_PROMPT= """Question: \"{}\" You should provide more information to help you answer the question and explain the reasons. If no any helpful information, you should answer NONE."""
    QUERY_PROMPT = """Query: {}\nJust answer the question only."""
    objects_list = list(objects_dict.keys())
    
    object_embedding = [_extract_embedding(x,retrieval_tokenizer,retrieval_model) for x in objects_list]
    obj_caption = ''
    if len(object_embedding) > 0:
        object_embedding = torch.concat(object_embedding)
        
        query_embedding = _extract_embedding(query,retrieval_tokenizer,retrieval_model)
        
        chosen_index_list = search(query_embedding,object_embedding,alpha,k=2)
        chosen_objects = []
        for index in chosen_index_list:
            chosen_objects.append(objects_list[index])
        
        print("Choosen objects list: {}".format(chosen_objects))
        for object in chosen_objects:
            loc_dict_list = objects_dict[object]
            for loc_dict in loc_dict_list:
                if loc_dict['mode'] == 'single':
                    loc = loc_dict['react'][0][1]
                    sub_image = image_array[loc[1]:loc[1] + loc[2],loc[0]:loc[0]+loc[3]]
                    sub_pil_image = Image.fromarray(sub_image)
                    attribute_prompt = ATTRIBUTE_PROMPT.format(query)
                    if hasattr(model,'generate_mode'):
                        resp = model.generate_mode(sub_pil_image,attribute_prompt)
                    else:
                        resp = model.generate(sub_pil_image,attribute_prompt)
                    #print("Response: {}".format(resp))
                    if resp != "NONE" and resp != "":
                        obj_caption += OBJ_PROMPT.format(resp)
                else:
                    loc_li = loc_dict['react']
                    image_list = np.array([image_array[react[0][1]:react[0][1] + react[0][2],react[0][0]: react[0][0] + react[0][3]] for react in loc_li])
                    mixup_img = np.mean(image_list,axis=0).astype(np.uint8)
                    sub_pil_image = Image.fromarray(mixup_img)
                    attribute_prompt = ATTRIBUTE_PROMPT.format(query)
                    if hasattr(model,'generate_mode'):
                        resp = model.generate_mode(sub_pil_image,attribute_prompt)
                    else:
                        resp = model.generate(sub_pil_image,attribute_prompt)
                    if resp != "NONE" and resp != "":
                        obj_caption += OBJ_PROMPT.format(resp)
    
    if obj_caption != '':
        query_prompt_with_obj = "Here are some useful information to help you answer the question: {}\n\n" + QUERY_PROMPT
        wrapperd_query = query_prompt_with_obj.format(
        obj_caption,query
        )
    else:
        wrapperd_query = query
    return wrapperd_query

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


Template_prompt = {
    'Caption': "Please describe this image.",
    "Refine_Caption": """Give you patch captions which describe the sub patch of the image repectively, you are required to combine all the information to generate refined caption about the image. Patch Captions: "{}"\n""",
    'Caption_Object': "Please describe the {} in the image. Provide the color, shape. If there is any text in the image, transcribe it here.",
    'Empty': "{}",
    "Entity_Extraction": """# System message
You are a language assistant that helps to extract information from given sentences.
# Prompt
Given a sentence which describe the image, extract the existent entities within the sentence for me. Extract the common objects and summarize them as general categories without repetition, merge essentially similar objects. Avoid extracting abstract or non-specific entities. Only extract concrete, certainly existent objects that fall in general categories and are described in a certain tone in the sentece. Extract entity in a JSON DICT. Output all the extracted types of items in one line and separate each object type with a period. You should ignore the singular and plural forms of nouns, and all extracted objects should be represented in singular form. If there is noting to output, then output a single empty list [].
Examples:
Sentence: "The bus is surrounded by a few other vehicles, including a car and a truck, which are driving in the same direction as the bus. A person can be seen standing on the sidewalk, possibly waiting for the bus or observing the scene."
Output: {{"object_list": ["bus","car","truck","person"]}}
Input:
Sentence: "{}"
Output: 
    """
}