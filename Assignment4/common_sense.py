import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
import re
from easydict import EasyDict as edict
from tqdm import tqdm

# pip install transformers tqdm easydict pyarrow accelerate

cfg = edict({
    'data_path': './cot/train-00000-of-00001.parquet',
    'device_name':"cuda" if torch.cuda.is_available() else "cpu",
    'model_type': './Llama-2-7b-chat-hf',#'./Meta-Llama-3-8B',
    
})

# 数据处理
# 将question、choices和text进行拼接
def concatenate_columns(row):
    # 构建问题字符串和选项字符串的合并
    question_part = f"{row['question']}"
    choices_part = ", ".join([f"{label}: {text}" for label, text in zip(row['choices']['label'], row['choices']['text'])])
    return f"{question_part} {choices_part}"

print("load data")
df = pd.read_parquet(cfg.data_path)
df_subset = df.head(50)
questions = df_subset.apply(concatenate_columns, axis=1).tolist()
answer_data = df_subset['answerKey'].tolist()



# 设置引导推理过程的触发句
trigger_sentence = "Let’s think step by step. "
# 为每个问题构建第一步提示X0
#prompt1s = ["Q: {}. A: {}".format(question, trigger_sentence) for question in questions]

# 调用llama模型
print("load model")
pipe = pipeline("text-generation", model=cfg.model_type, device_map=cfg.device_name, torch_dtype=torch.float16)
eos_token_id = pipe.model.config.eos_token_id
# 正确数量
correct_answer_count = 0


for question, real_answer in tqdm(zip(questions, answer_data), total=len(questions)):
    prompt = "Q: {}. A: {}".format(question, trigger_sentence)
    print("Prompt X0:", prompt)
    # 调用llama模型，生成后续句子Z
    # 释放未使用的显存
    torch.cuda.empty_cache()
    text = pipe(prompt, temperature=0.1, eos_token_id=eos_token_id)
    print("Sentence z:", text[0]['generated_text'])
    # A为触发模型输出答案的模板
    A = "Therefore, among A through E, the answer is"
    full_prompt = "{}{}{}".format(prompt, text[0]['generated_text'], A)
    # 调用llama模型，生成最终输出结果
    end_text = pipe(full_prompt, eos_token_id=eos_token_id)
    print("Outcome:", end_text[0]['generated_text'])
    # 提取最终结果中的最后一个字母作为答案
    answer = re.findall(r'[A-E]', end_text[0]['generated_text'])
    predicted_answer = answer[-1] if answer else None
    print("Predict answer:", predicted_answer)
    print('Real answer:', real_answer)
    if predicted_answer == real_answer:
        correct_answer_count += 1

# 计算准确率
accuracy = correct_answer_count / len(answer_data)
print("Accuracy: {:.2%}".format(accuracy))
