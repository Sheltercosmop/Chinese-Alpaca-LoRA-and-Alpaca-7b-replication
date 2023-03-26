# Alpaca-7b replication and multilingual-Alpaca-LoRA  OpenSource
This project is supported by AI community: [AcceleratorI](https://www.acceleratori.com)  
## A alpaca-7b-hf replication
A alpaca-7b-hf replication with no moderation filter and output watermark  
## Multilingual Instruction LLaMA LoRA 
This is a multilingual instruction LLaMA LoRA that mainly focused on Simplified Chinese but also support Japanese and Traditional Chinese(HK and TW)
### Version
Version 0.4 (Newest)  
This repo has now been updated to the forth version of our training results.
### Training
We have tuned a Chinese LLaMA model baed on  
- [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)  
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  
- [Alpaca LoRA](https://github.com/tloen/alpaca-lora)  
### Dataset
We used the following datasets:  
  
JosephusCheung/GuanacoDateset  
BelleGroup/generated_train_0.5M_CN  
Baike_qa2019 (Partially used)  
  
It tooks us about 30 hours on 8 80G A100s to fine-tune this model  

## Citation
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
