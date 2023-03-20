from transformers import LLaMATokenizer, LLaMAForCausalLM

llama_path = "./llama-7b-hf"
alpaca_path = "./alpaca-7b-hf"

tokenizer = LLaMATokenizer.from_pretrained(llama_path)

model = LLaMAForCausalLM.from_pretrained(
    alpaca_path,
    load_in_8bit=True,
    device_map="auto",
)
print(model.device)
counter = 0
while True:
    counter += 1
    if counter > 2:
        break
    user_input = input("Enter a prompt: ")
    if user_input == "exit":
        break
    PROMPT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:'''
    user_input = PROMPT.format(instruction=user_input)
    inputs = tokenizer(user_input, max_length=512, truncation=True, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    generation_output = model.generate(
        input_ids=input_ids, max_length=256,
        early_stopping=True,
        num_beams=10,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        top_k=50,
        top_p=0.95,
    )
    # print(generation_output)
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
