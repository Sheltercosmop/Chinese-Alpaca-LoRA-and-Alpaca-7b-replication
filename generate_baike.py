import torch
from transformers import LLaMATokenizer, LLaMAForCausalLM, GenerationConfig

from peft import PeftModel

llama_path = "../llama-7b-hf"
llama_path = "/root/Shelter/guanaco-7B-lora-embed_custom"
lora_path = "/root/Shelter/llama-7b-hf-lora-baike"

tokenizer = LLaMATokenizer.from_pretrained("../llama-7b-hf")
model = LLaMAForCausalLM.from_pretrained(llama_path, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float16)

for name, param in model.named_parameters():
    if param.data.dtype != torch.float16:
        # some param is float32
        param.data = param.data.to(torch.float16)

# PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# 请说出一种健康易于制作的早餐。
# ### Response:'''
#
# inputs = tokenizer(PROMPT, return_tensors="pt")
# input_ids = inputs.input_ids.cuda()
# generation_output = model.generate(
#     input_ids=input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=128
# )
# for s in generation_output.sequences:
#     print(tokenizer.decode(s))
generation_config = GenerationConfig(
    temperature=0.95,
    early_stopping=True,
    num_beams=4,
    num_return_sequences=1,
    repetition_penalty=1.5,
    top_k=50,
    top_p=0.95,
)
counter = 0
while True:
    # counter += 1
    # if counter > 10:
    #     break
    user_input = input("Enter a prompt: ")
    if user_input == "exit":
        break
    PROMPT = '''
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:'''
    user_input = PROMPT.format(instruction=user_input)
    inputs = tokenizer(user_input, max_length=512, truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=756,
    )
    # print(generation_output)
    # print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
    for s in generation_output.sequences:
        output = tokenizer.decode(s, skip_special_tokens=True)
        print("Response:", output.split("### Response:")[1].strip())
        # print("Response:", output)