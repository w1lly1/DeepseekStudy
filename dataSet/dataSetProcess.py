from datasets import load_dataset

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

if __name__ == "__main__":
    ds = load_dataset(r'E:\MyOwn\ProgramStudy\AI\dataSet\gsm8k_chinese')
    data = process_dataset(ds)
    print(data)