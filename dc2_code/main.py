
from dc2 import DC2

if __name__ == '__main__':
    
    image_path = r'./demo.jpg'
    question = "What's the color of the umbrella?"
    
    # initialize model
    llava_v15_7b_model_path = r'YOUR MLLM PATH'
    retrieval_path = r'YOUR RETRIEVAL PATH' # contriever_msmacro
    llm_path = r'LLM PATH' # NOTE: If MLLM can accurately extract objects, there will be no need for an additional LLM.
    
    model = DC2(llava_v15_7b_model_path,retrieval_path,llm_path)
    
    # start inference
    llava_model = model.mllm
    
    original_response = llava_model.generate(image_path,question)
    print("LLaVA-v1.5 7B: {}".format(original_response))
    
    response, visual_memory = model.run_main(image_path,question,5,0.3,0.1,None)
    print("LLaVA-v1.5 7B w/ DC^2: {}".format(response))
