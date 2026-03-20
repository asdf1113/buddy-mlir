from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

ids = [11,602,1097,502,311,420,12111,323,602,1097,4560,311,636,264,2731,8830,315,279,2204,4595,315,828,430,649,387,1511,304,264,31649,1646,13,602,1097,4560,311,3619,279,6811] # 这里后面替换成C++输出的token ids
text = tokenizer.decode(ids, skip_special_tokens=False)

print(text)
