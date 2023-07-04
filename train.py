from Generation.generation_model import Generation_model
from utils import *
import asyncio
from datasets import load_dataset
from Selection import Train_Selection_model


def main():
    datasets = load_dataset("ml4pubmed/pubmed-classification-20k")
    original_data = preprocess_dataset(datasets['train'])
    few_data = list(make_fewshot_dataset(original_data,2).items())
    sublists_length = int(len(few_data)/2)
    sublists = [few_data[i:i+sublists_length] for i in range(0, len(few_data), sublists_length)]
    async def generation_with_model_1(Generation_model = None):
        return Generation_model.generate(sublists[Generation_model.device_number])

    models = [Generation_model(device_number) for device_number in range(2)]
    async_list = [generation_with_model_1(model) for model in models]
    async def generate_async():
        results = await asyncio.gather(
            *async_list
        )
        return results
    result = asyncio.run(generate_async())
    selection = Train_Selection_model.selection_model()
    selection.train()
    print(result)



if __name__ == "__main__":
    main()
