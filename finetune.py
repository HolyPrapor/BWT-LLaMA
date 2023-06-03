from xturing.datasets import InstructionDataset
from xturing.models import BaseModel

# Load the dataset
instruction_dataset = InstructionDataset("./alpaca_data")
# Initialize the model
model = BaseModel.create("llama_lora_int4")

finetuning_config = model.finetuning_config()

print(finetuning_config)

finetuning_config.num_train_epochs = 1

print(model.finetuning_config())

# Finetune the model
model.finetune(dataset=instruction_dataset)

model.save('./finetuned_lora')