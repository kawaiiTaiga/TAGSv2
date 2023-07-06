from Generation.generation_model import Generation_model
from utils import *
import asyncio
from datasets import load_dataset
from Selection import Train_Selection_model

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from Selection.simcse.models import RobertaForCL, BertForCL

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", "},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=5.0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


def main():
    model_args = ModelArguments()
    model = BertForCL.from_pretrained(
        'bert-base-uncased',
        from_tf=bool(".ckpt" in 'bert-base-uncased'),
        config = AutoConfig.from_pretrained('bert-base-uncased'),
        model_args = model_args)
    datasets = load_dataset("ml4pubmed/pubmed-classification-20k")
    original_data = preprocess_dataset(datasets['train'])
    few_dict = make_fewshot_dataset(original_data,2)
    few_data = list(few_dict.items())
    sublists_length = int(len(few_data)/4)
    create_contrastive_data('contrastive_data.tsv', few_dict)
    
  

    sublists = [few_data[i:i+sublists_length] for i in range(0, len(few_data), sublists_length)]
    async def generation_with_model_1(Generation_model = None):
        return Generation_model.generate(sublists[Generation_model.device_number])

    models = [Generation_model(device_number) for device_number in range(4)]
    async_list = [generation_with_model_1(model) for model in models]
    async def generate_async():
        results = await asyncio.gather(
            *async_list
        )
        return results
    result = asyncio.run(generate_async())

    print(result)
    s = Train_Selection_model.sss()
    model = s.train_selection_model(model)
    
   
    #Train_Selection_model.train_selection_model()
    #selection = Train_Selection_model.selection_model()
    #selection.train()
    #selection.save_model()




if __name__ == "__main__":
    main()
