from xturing.models import BaseModel

# Initialize the model
model = BaseModel.create("llama_lora_int4")

model.save('./llama_7b')
