from ...smp import *
import torch
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
import os
import re

def initialize_pipeline():
    tp = torch.cuda.device_count()
    backend_config = TurbomindEngineConfig(tp=tp)
    llm_path = os.environ.get('llm_path', None)
    assert llm_path is not None, "HR-Bench need a LLM to evaluate."
    model = pipeline(llm_path,backend_config=backend_config)
    return model

def build_prompt(line):
    question = line['question']
    label = line['answer']
    gt_text = line[label]
    options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
    }
    options_prompt = 'Options:\n'
    for key, item in options.items():
        options_prompt += f'{key}. {item}\n'
    prompt = ''
    prompt += f'{question}\n'
    if len(options):
        prompt += options_prompt
        prompt += 'Please select the correct answer from the options above. \n'
    prediction = line['prediction']
    ground_truth = f'{label}. {gt_text}'
    content = f"Given the following question: {prompt}, the correct answer is: {ground_truth}. Does the following answer correctly answers the question, answer:{prediction}?"
    return content

def parse_score(answer):
    logger = get_logger('Evaluation')
    try:
        answer = answer.lower()
        yes_no_regex = re.compile(r"^(yes|no)", re.IGNORECASE)

        gpt_extracted = yes_no_regex.match(answer)
        if gpt_extracted:
            if gpt_extracted[0] == 'yes':
                return 1,answer
            else:
                return 0,answer
        else:
            return 0,answer
    except Exception as e:
        logger.error(e, 'error', answer)
        return 0, ''

def pipeline_eval(prompts,model):
    gen_config = GenerationConfig(top_p=0.8,top_k=40,temperature=0.2)
    scores_list = []
    inp_list = []
    batch_size = 16
    for prompt in prompts:
        inp = "You are a helpful and precise assistant for checking the quality of the answer. Please answer in only yes or no. {}".format(prompt)
        inp_list.append(inp)
    for i in tqdm(range(0,len(inp_list),batch_size)):
        batch_inp_list = inp_list[i:i+batch_size]
        resp = model(batch_inp_list,gen_config=gen_config)
        
        def wrapper(resp_list):
            wrapper_resp_list = [resp.text for resp in resp_list]
            return wrapper_resp_list
        
        resp_list = wrapper(resp)
        for resp_single in resp_list:
            scores = parse_score(resp_single)
            scores_list.append(scores)
    return scores_list


def hrbench_score(data):
    ret = defaultdict(list)
    resp_dic = {}
    sz = len(data)
    category_list = set(data['category'])
    score_dict = defaultdict(list)
    
    for i in range(len(data)):
        d = data.iloc[i]
        category = d['category']
        gpt_score = d['gpt_score']
        score_dict[category].append(gpt_score)
        score_dict['all'].append(gpt_score)
    
    all_acc = np.mean(score_dict['all'])
    ret['type'].append('all')
    ret['acc'].append(all_acc)
    resp_dic['all'] = all_acc
    for cate in category_list:
        acc = np.mean(score_dict[cate])
        ret['type'].append(cate)
        ret['acc'].append(acc)
        
        resp_dic[cate] = acc

    return pd.DataFrame(ret),resp_dic
