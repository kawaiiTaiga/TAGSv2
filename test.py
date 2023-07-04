from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging

logger = logging.getLogger(__name__)

@dataclass
class generationArguments:
    model_name: str = field(
        default="chavinlo/alpaca-native",
        metadata = {
            "help": "The path for generation model",
        },
    )
    test: str = field(
        default = "ss",
        metadata = {
            'help':'hlep'
        }
    ) 

@dataclass
class tt:
    test2: str = field(
        default="chavinlo/alpaca-native",
        metadata = {
            "help": "The path for generation model",
        },
    )

def check(cc):
    print(cc.model_name)
def main():
    parser = HfArgumentParser((generationArguments,tt))
    generation_args,ts = parser.parse_args_into_dataclasses()
    check(generation_args)
    print(ts.test2)
    #logger.info(generation_args.model_name)

if __name__ == "__main__":
    main()