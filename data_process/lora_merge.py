from transformers import AutoModel

# 加载模型
model = AutoModel.from_pretrained("/data/qq/models/Qwen/Qwen2.5-Math-7B")

# 加载Lora参数
lora_params = torch.load("path_to_your_lora_params.pth")

# 合并Lora参数
for name, param in lora_params.items():
    if name in model.state_dict():
        model.state_dict()[name].data += param.data

# 保存合并后的模型
model.save_pretrained("path_to_save_merged_model")
