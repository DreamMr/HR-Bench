import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        
    }

    DATASET_MD5 = {
        
    }

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs


class HRBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'HRBench4K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv',
        'HRBench8K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv',
    }

    DATASET_MD5 = {
        'HRBench4K': 'f6b041b03d49543494b8a56d2e35be65',
        'HRBench8K': '274c9c7f89329b804a4723178a00219c',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.hrbench import build_prompt, initialize_pipeline, pipeline_eval, hrbench_score
        suffix = '.' + eval_file.split('.')[-1]        
        record_file = eval_file.replace(suffix, '_openai_result' + suffix)
        score_file = eval_file.replace(suffix, '_score.csv')

        if osp.exists(score_file):
            return None
        
        if not osp.exists(record_file):
            data = load(eval_file)
            lines = [data.iloc[i] for i in range(len(data))]
            
        model = initialize_pipeline()
        prompts = [build_prompt(line) for line in lines]
        scores = pipeline_eval(prompts,model)
        data['gpt_score'] = [x[0] for x in scores]
        data['gpt_answer'] = [x[1] for x in scores]
        dump(data, record_file)
        
        data = load(record_file)
        cycle_group = data.groupby('cycle_category')
        result_dic = defaultdict(list)
        avg_dic = defaultdict(int)
        count = 0
        for key, data_value in cycle_group:
            _,resp_dic = hrbench_score(data_value)
            count += 1
            for k,v in resp_dic.items():
                result_dic['cycle'].append(key)
                result_dic['type'].append(k)
                result_dic['acc'].append(v)
                
                avg_dic[k] += v
        
        for k,v in avg_dic.items():
            result_dic['cycle'].append('avg')
            result_dic['type'].append(k)
            result_dic['acc'].append(v/count)
        result_pd = pd.DataFrame(result_dic)
        dump(result_pd, score_file)
        return result_pd
    
class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
