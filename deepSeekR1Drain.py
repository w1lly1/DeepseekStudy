from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataSet.dataSetProcess import process_dataset
import datetime

def timestamp_print(message):
    current_time = datetime.datetime.now().strftime("[%Y/%m/%d/%H:%M:%S:%f")[:-3] + "]"
    print(f"{current_time} {message}")

if __name__ == "__main__":
    model_name = r"E:\huggingFace\downloads\Qwen\Qwen2.5-VL-7B-Instruct"
    # 加载预训练因果模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", # 自动选择模型合适的张量数据类型
        # device_map={"": 0}
    )
    timestamp_print("main(): model loaded")

    # 将模型移动到GPU上
    model.cuda()
    timestamp_print("main(): cuda enabled")

    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    timestamp_print("main(): tokenizer loaded")

    ds = load_dataset('E:\MyOwn\ProgramStudy\AI\dataSet\gsm8k_chinese')
    data = process_dataset(ds)
