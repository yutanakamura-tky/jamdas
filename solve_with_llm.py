import argparse
import datetime
import gc
import json
import os
import re
from json import JSONEncoder
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompt_maker import CommandRPromptMaker, LlamaPromptMaker, SwallowPromptMaker
from src.utils.logger import get_logger


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.int64):
            return int(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.ndarray):
            return list(o)
        return JSONEncoder.default(self, o)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model-name", dest="model_name", type=str)
    parser.add_argument(
        "-e", "--experiment-name", dest="experiment_name", type=str, default=""
    )
    parser.add_argument("-4", "--4bit", dest="quant_4bit", action="store_true")
    parser.add_argument("-8", "--8bit", dest="quant_8bit", action="store_true")
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Overwrite a result directory if an old experiment with the same experiment name exists. Resumes the same experiment if set to False.",
    )
    parser.add_argument(
        "-s", "--sampling-mode", dest="sampling_mode", type=str, default="first"
    )
    parser.add_argument(
        "-r", "--random-state", dest="random_state", type=int, default=42
    )
    parser.add_argument(
        "-R",
        "--iter-random-state",
        dest="iter_random_state",
        action="store_true",
        default=False,
    )
    parser.add_argument("-n", "--shots", dest="n", type=int, default=5)
    parser.add_argument("-l", "--max-length", dest="max_length", type=int, default=8192)
    parser.add_argument(
        "-L",
        "--max_completion_length",
        dest="max_completion_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "-t", "--temperature", dest="temperature", type=float, default=0.0
    )
    parser.add_argument("-p", "--top-p", dest="top_p", type=float)
    parser.add_argument("-k", "--top-k", dest="top_k", type=int)
    args = parser.parse_args()

    return args


def get_time_now() -> str:
    t_delta = datetime.timedelta(hours=9)
    jst = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(jst)
    ft_now = now.strftime("%Y%m%d%H%M%S")
    return ft_now


# Setup
args = get_args()
time_now = get_time_now()
experiment_name = args.experiment_name or time_now
OUTPUT_DIR = Path("result") / experiment_name

if os.path.exists(OUTPUT_DIR) and not args.overwrite:
    RESUME_MODE = True
    with open(OUTPUT_DIR / "config.json") as f:
        args = argparse.Namespace(**json.loads(f.read()))
        args.overwrite = False
else:
    RESUME_MODE = False

args.experiment_time = time_now
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_DIR / "config.json", "w") as f:
    f.write(json.dumps(vars(args), indent=4))

logger = get_logger(name=__name__, log_save_path=OUTPUT_DIR / "solve_with_llm.log")
if RESUME_MODE:
    logger.info(
        f"### RESUME MODE ###: Command-line variables have been overloaded by previously saved config: {OUTPUT_DIR / 'config.json'}"
    )
print(vars(args))

# Validate args
logger.info("Validating command-line arguments...")
accepted_sampling_modes = ["first", "comprehensive"]
if args.sampling_mode not in accepted_sampling_modes:
    raise ValueError(f"args.sampling_mode must be either {accepted_sampling_modes}")


def main():
    # Load data
    DIR = Path("data")
    input_paths = [DIR / "batch_0.csv", DIR / "batch_1.csv"]
    dfs = [pd.read_csv(path) for path in input_paths]

    df = pd.concat(
        [_df.query("作業フラグ and diagnostic_impression_plain") for _df in dfs], axis=0
    )
    df = df.drop(["作業フラグ", "備考"], axis=1).reset_index(drop=True)

    # Prepare prompt maker
    non_label_columns = [
        "patient_id",
        "visit_id",
        "visited_date_raw",
        "diagnostic_impression_plain",
        "batch",
    ]
    label_columns = [column for column in df.columns if column not in non_label_columns]

    logger.info(f"###### Solve with {args.model_name} ######")
    raw_outputs, pred_df, df_metrics, referenced_sample_indexes = (
        solve_with_single_model_few_shot(
            df=df,
            label_columns=label_columns,
            model_name=args.model_name,
            sampling_mode=args.sampling_mode,
            n=args.n,
            max_completion_length=args.max_completion_length,
            random_state=args.random_state,
            iter_random_state=args.iter_random_state,
            temperature=args.temperature,
            output_dir=OUTPUT_DIR,
        )
    )


def annotation_to_json_string(
    df: pd.DataFrame, label_columns: list[str]
) -> tuple[list[str], list[dict], list[str]]:
    target_json = []
    target_strings = []
    target_contexts = []
    for ir, r in df.loc[:, label_columns].iterrows():
        target_dict = r.dropna().to_dict()
        target_dict_clean = {}
        for k, v in target_dict.items():
            target_name, target_type = k.split("_")
            if target_type == "flg":
                target_dict_clean[target_name] = int(v)
            elif target_type == "num":
                target_dict_clean[target_name] = float(v)
            else:
                target_dict_clean[target_name] = str(v)
        target_string = json.dumps(target_dict_clean, ensure_ascii=False)
        target_json.append(target_dict_clean)
        target_strings.append(target_string)
        target_contexts.append(df["diagnostic_impression_plain"].loc[ir])
    return target_contexts, target_json, target_strings


def load_model(
    model_name,
    max_length: int = 8192,
    quant_4bit: bool = False,
    quant_8bit: bool = False,
):
    accelerator = Accelerator()

    if quant_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = BitsAndBytesConfig()

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = accelerator.prepare(model)

    SWALLOW = "Swallow" in model_name

    if SWALLOW:
        chat_template = """
{% for message in messages %}
    {% if message['role'] == 'system' %}
        {{ message['content'] + '\n'}}
    {% elif message['role'] == 'user' %}
        {{ message['content'] + '\n' }}
    {% endif %}
{% endfor %}
"""
        tokenizer.chat_template = chat_template
        logger.info(
            "*** Tokenizer has no chat template, so applied a new chat template ***"
        )

    return tokenizer, model, accelerator


def solve_with_single_model_few_shot(
    df: pd.DataFrame,
    label_columns: list[str],
    model_name: str,
    sampling_mode: str = "first",
    n: int = 5,
    max_completion_length: int = 512,
    random_state: int = 42,
    iter_random_state: bool = False,
    temperature: float = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    # Set variables
    COMMAND_R = "command-r" in model_name
    MIXTRAL = "Mixtral" in model_name
    LLAMA = "Llama" in model_name
    SWALLOW = "Swallow" in model_name
    PHI = "Phi" in model_name

    if COMMAND_R:
        prompt_maker = CommandRPromptMaker(label_columns=label_columns)

    elif MIXTRAL or LLAMA or PHI:
        prompt_maker = LlamaPromptMaker(label_columns=label_columns)

    elif SWALLOW:
        prompt_maker = SwallowPromptMaker(label_columns=label_columns)

    else:
        raise ValueError("Invalid model names")

    # Load model
    tokenizer, model, accelerator = load_model(
        model_name=model_name, quant_4bit=args.quant_4bit, quant_8bit=args.quant_8bit
    )
    logger.info(f"Loaded model {model_name}")
    logger.info(f"Device: {model.device}")

    # Set save paths
    raw_output_save_path = output_dir / "outputs.csv"

    pred_df_save_path = output_dir / "pred.csv"
    logger.info(f"Save path (prediction result): {pred_df_save_path}")

    df_metrics_save_path = OUTPUT_DIR / "metrics.csv"
    logger.info(f"Save path (metrics): {df_metrics_save_path}")

    reference_sample_indexes_save_path = OUTPUT_DIR / "reference_sample_indexes.csv"
    logger.info(
        f"Save path (reference sample indexes): {reference_sample_indexes_save_path}"
    )

    # Initialize output values
    last_index = 0
    raw_outputs: list[str] = []
    predictions: list[dict] = []
    reference_sample_indexes: list[list[int]] = []

    # If resume mode, load intermediate results
    if RESUME_MODE and os.path.exists(pred_df_save_path):
        raw_outputs_df = pd.read_csv(raw_output_save_path)
        raw_outputs = raw_outputs_df["raw_outputs"].values.tolist()

        pred_df = pd.read_csv(pred_df_save_path)
        last_index = len(pred_df) - 1
        logger.info(f"Last index: {last_index - 1}")
        logger.info(f"Resume from sample ID: {last_index}")

        predictions = pred_df.to_dict(orient="records")

        with open(reference_sample_indexes_save_path) as f:
            reference_sample_indexes = json.loads(f.read())

    target_contexts, target_json, target_strings = annotation_to_json_string(
        df, label_columns
    )
    target_df = pd.DataFrame(target_json)

    logger.info("Start inference ...")

    for i, target_context in tqdm(enumerate(target_contexts), total=len(target_df)):
        if i <= last_index:
            continue

        logger.info(f"=== Sample ID: {i} ===")
        if sampling_mode == "comprehensive":
            reference_sample_index: list[int] = prompt_maker.comprehensive_sample(
                reference_labels=target_df,
                random_state=random_state,
                ignore_sample_index=i,
            )
            reference_sample_indexes.append(reference_sample_index)
            with open(reference_sample_indexes_save_path, "w") as f:
                f.write(
                    json.dumps(reference_sample_indexes, indent=4, cls=NumpyJSONEncoder)
                )

        messages = [
            {"role": "system", "content": prompt_maker.system_prompt},
            {
                "role": "user",
                "content": prompt_maker.generate_prompt_few_shots(
                    context=target_context,
                    reference_contexts=target_contexts,
                    reference_labels=target_df,
                    reference_label_strings=target_strings,
                    mode=sampling_mode,
                    ignore_sample_index=i,
                    n=n,
                    random_state=random_state,
                    iter_random_state=iter_random_state,
                ),
            },
        ]

        encoded_input = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(accelerator.device)

        with torch.no_grad():
            generation_kwargs = {
                "inputs": encoded_input,
                "max_new_tokens": max_completion_length,
                "do_sample": temperature == 0.0,
                "temperature": temperature if temperature > 0.0 else None,
                "top_p": top_p if temperature > 0.0 else None,
                "top_k": top_k if temperature > 0.0 else None,
            }

            if LLAMA:
                generation_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

            encoded_output = model.generate(**generation_kwargs)[0]

        decoded_output = tokenizer.decode(
            encoded_output[encoded_input.shape[-1] :], skip_special_tokens=True
        )

        raw_output = re.sub(r"[\s\S]*\[\/INST\]", "", decoded_output, 1)
        raw_output = re.sub(r"</s>", "", raw_output)
        raw_outputs.append(raw_output)
        raw_outputs_df = pd.DataFrame({"raw_outputs": raw_outputs})
        raw_outputs_df.to_csv(raw_output_save_path)

        json_match = re.search(r"\{([^{}]*)\}", raw_output)
        if json_match:
            json_text = json_match.group(0)
            try:
                predictions.append(json.loads(json_text))
                logger.info(json_text)
            except json.JSONDecodeError:
                logger.warning("failed to parse", decoded_output)
                predictions.append({})
        else:
            logger.warning("no match", decoded_output)
            predictions.append({})

        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(pred_df_save_path)

        try:
            df_metrics = pd.DataFrame(
                evaluate_model(
                    target_df.iloc[:i, :],
                    pred_df.iloc[:i, :],
                    target_df.columns,
                    fill_value=0,
                )
            )
            df_metrics.to_csv(df_metrics_save_path)
        except ValueError as e:
            logger.warning(
                f"Value error occurred during evaluation, so skip metric calculation: {e}"
            )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    accelerator.free_memory()

    return raw_outputs, pred_df, df_metrics, reference_sample_indexes


def evaluate_model(target_df, prediction_df, target_cols, fill_value=0):
    """
    cited from 01_llm_prompt.ipynb
    2のtarget labelに対処するため、とりあえずnp.clip.
    """
    scores = []
    for col in target_cols:
        # print(col)
        target_arr = target_df[col].fillna(fill_value).values

        # targetが0,1,2のときに対処.
        try:
            if target_arr.max() == 2:
                target_arr = np.clip(target_arr, 0, 1)
        except:  # noqa
            pass

        baseline_arr = np.array([fill_value] * len(target_arr))
        if col in prediction_df.columns:
            approx_arr = prediction_df[col].fillna(fill_value).values
        else:
            approx_arr = np.array([fill_value] * len(target_arr))
        acc = (target_arr == approx_arr).mean()
        acc_baseline = (target_arr == baseline_arr).mean()
        target_arr_bin = (target_arr != fill_value).astype(int)
        approx_arr_bin = (approx_arr != fill_value).astype(int)
        approx_arr_bin[(approx_arr != target_arr) & (target_arr != fill_value)] = 0

        # labelが全て0の時に対処.
        try:
            pos_present = target_arr.sum()
        except:  # noqa
            pos_present = 1

        if pos_present:
            tn, fp, fn, tp = confusion_matrix(target_arr_bin, approx_arr_bin).ravel()
            tpr = tp / (tp + fn)
            spc = tn / (fp + tn)
            f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            f1, tpr, spc = 0, 0, 0  # ラベルが全部0の場合は便宜的に処理.

        scores.append(
            {
                "item": col,
                "f1": f1,
                "sensitivity": tpr,
                "specificity": spc,
                "acc": acc,
                "acc_baseline(all negative)": acc_baseline,
            }
        )
    return scores


if __name__ == "__main__":
    main()
