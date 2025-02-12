from datasets import load_dataset

def process_dataset(data):
    return data

if __name__ == "__main__":
    ds = load_dataset('E:\MyOwn\ProgramStudy\AI\dataSet\gsm8k_chinese')
    data = process_dataset(ds)