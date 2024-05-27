import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
import re
from easydict import EasyDict as edict
from tqdm import tqdm

# pip install transformers tqdm easydict pyarrow accelerate

cfg = edict({
    'data_path': './cot/gsm8k_test.parquet',
    'device_name': "cuda",  # 使用CPU
    'model_type': './Meta-Llama-3-8B',#'./Llama-2-7b-chat-hf',#
})

# 数据处理
def extract_answer(answer):
    if '####' in answer:
        return answer.split('####')[-1].strip()
    return None

print("load data")
df = pd.read_parquet(cfg.data_path)
df_subset = df.head(50)
questions = df_subset['question'].tolist()
answer_data = df_subset['answer'].apply(extract_answer).tolist()

# 设置引导推理过程的触发句
trigger_sentence = "Let’s think step by step"
#first_step_prompts = ["Q: {}. A: {}".format(question, trigger_sentence) for question in questions]

# 调用llama模型
print("load model")
pipe = pipeline("text-generation", model=cfg.model_type, device_map=cfg.device_name)
eos_token_id = pipe.model.config.eos_token_id

# 正确数量
correct_answer_count = 0

for question, real_answer in tqdm(zip(questions, answer_data), total=len(questions)):
    prompt = "Q: {}. A: {}".format(question, trigger_sentence)
    print("Prompt X0:", prompt)
    # 释放未使用的显存
    torch.cuda.empty_cache()
    # 调用llama模型，生成后续句子Z
    text = pipe(prompt, temperature=0.1, eos_token_id=eos_token_id,max_new_tokens=50)
    print("Sentence z:", text[0]['generated_text'])
    # A为触发模型输出答案的模板
    A = "Therefore, the answer (arabic numerals) is"
    full_prompt = "{}{}".format(text[0]['generated_text'], A)
    # 调用llama模型，生成最终输出结果
    end_text = pipe(full_prompt, eos_token_id=eos_token_id,max_new_tokens=10)
    print("Outcome:", end_text[0]['generated_text'])
    # 提取最终结果中的最后一个数字作为答案
    answer = re.findall(r'\d+', end_text[0]['generated_text'])
    predicted_answer = int(answer[-1]) if answer else None
    print("Predict answer:", predicted_answer)
    print('Real answer:', int(real_answer))
    if predicted_answer == int(real_answer):
        correct_answer_count += 1

# 计算准确率
accuracy = correct_answer_count / len(answer_data)
print("Accuracy: {:.2%}".format(accuracy))
