import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# from dataSet.dataSetProcess import process_dataset
from trl import GRPOConfig, GRPOTrainer# 强化学习库
import datetime # 日志打印

def timestamp_print(message):
    current_time = datetime.datetime.now().strftime("[%Y/%m/%d/%H:%M:%S:%f")[:-3] + "]"
    print(f"{current_time} {message}")

SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""

def process_dataset(data):
    # 1. 为数据集体检prompt 列
    # 2. 将数据集的answer 列直接替换为原始数据集的anser_only 列
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    }) 
    return data

def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125

    if text.count("</think>\n") == 1:
        reward += 0.125

    if text.count("<answer>\n") == 1:
        reward += 0.125

    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward

# 生成答案是否正确的奖励
def correctness_reward(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]

# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，
# 所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式奖励
def hard_format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

# 格式奖励
def soft_format_reward(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]

if __name__ == "__main__":
    # Qwen/Qwen2.5-0.5B
    # model_name = r"E:\huggingFace\downloads\Qwen\Qwen2.5-VL-7B-Instruct"
    model_name = r"E:\huggingFace\downloads\Qwen\Qwen2.5-1.5B-Instruct"
    # 加载预训练因果模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # 自动选择模型合适的张量数据类型
    )
    timestamp_print("main(): model loaded")

    # 将模型移动到GPU上
    model.cuda()
    timestamp_print("main(): cuda enabled")

    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    timestamp_print("main(): tokenizer loaded")

    ds = load_dataset(r'E:\MyOwn\ProgramStudy\AI\dataSet\gsm8k_chinese')
    data = process_dataset(ds['train'])

    output_dir = r"E:\MyOwn\ProgramStudy\AI\output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6, # adamW 学习率
        # adam_beta1 = 0.9,
        # adam_beta2 = 0.99, 
        # weight_decay = 0.1,
        # warmup_ratio = 0.1,
        # lr_scheduler_type='cosine',
        # logging_steps=1,
        # bf16=True,
        per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        num_generations=4,
        # max_prompt_length=256,
        # max_completion_length=200,
        # num_train_epochs=1,
        # save_steps=100,
        # max_grad_norm=0.1,
        # log_on_each_node=False,
        # use_vllm=False,
        # report_to="tensorboard"
    )

    print(f"Global train batch size: {training_args.per_device_train_batch_size}")
    print(f"Number of generations per prompt: {training_args.num_generations}")

    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        mark_reward,
        soft_format_reward,
        hard_format_reward,
        digit_reward,
        correctness_reward
        ],
    args=training_args,
    train_dataset=data,

)
    trainer.train()
    trainer.save_model(output_dir)
