from typing import Optional

import numpy as np
import pandas as pd


class BasePromptMaker:
    # flake8: noqa: W293
    system_prompt = """
"""

    template_summary = """
{target_format}
{example_text}
{context}
"""
    # flake8: noqa

    type_description = {"binary": "binary", "numeric": "float", "text": "text"}
    additional_desciption = {
        "微熱": "either clearly stated or a temperature within [37.0, 37.4]",
        "高熱": "either clearly stated or a temperature above 38.0",
        "体温": "extract the highest temperature",
        "筋肉痛": "include joint pain",
        "咽頭痛": "include irritating throat",
    }

    def __init__(self, label_columns: list[str]):
        self.label_columns = label_columns
        self.target_description = ""
        self.target_category = {"binary": [], "numeric": [], "text": []}

        for col in self.label_columns:
            target_name, target_type = col.split("_")
            if target_type == "flg":
                self.target_category["binary"].append(target_name)
            elif target_type == "num":
                self.target_category["numeric"].append(target_name)
            else:
                self.target_category["text"].append(target_name)

        for cat, cols in self.target_category.items():
            for col in cols:
                self.target_description += f'- {col} ({self.type_description[cat]}{","+ self.additional_desciption[col] if col in self.additional_desciption.keys() else ""})\n'

        self.target_description = self.target_description[:-2]

    def generate_prompt(
        self, context: str, examples: Optional[list[tuple[str, str]]] = None
    ) -> str:
        examples = examples or []

        example_str = ""
        for qa in examples:
            q, a = qa
            example_str += f"Example context: {q}\nExample answer: {a}\n\n"

        return self.template_summary.format(
            context=context,
            example_text=example_str,
            target_format=self.target_description,
        )

    def generate_prompt_few_shots(
        self,
        context: str,
        reference_contexts: list[str],
        reference_labels: pd.DataFrame,
        reference_label_strings: list[str],
        mode: str = "first",
        ignore_sample_index: Optional[int] = None,
        n: int = 5,
        random_state: int = 42,
        iter_random_state: bool = False,
    ) -> str:

        if mode == "first":
            if ignore_sample_index is not None and ignore_sample_index < n:
                sample_index = list(range(n + 1))
                sample_index.pop(ignore_sample_index)
            else:
                sample_index = list(range(n))

        elif mode == "comprehensive":
            if iter_random_state:
                random_state += int(ignore_sample_index)
            sample_index = self.comprehensive_sample(
                reference_labels,
                random_state=random_state,
                ignore_sample_index=ignore_sample_index,
            )

        sampled_reference_contexts = np.array(reference_contexts)[sample_index].tolist()
        sampled_reference_strings = np.array(reference_label_strings)[
            sample_index
        ].tolist()

        prompt = self.generate_prompt(
            context=context,
            examples=[
                (context, string)
                for context, string in zip(
                    sampled_reference_contexts, sampled_reference_strings
                )
            ],
        )

        return prompt

    def comprehensive_sample(
        self,
        reference_labels: pd.DataFrame,
        random_state: int = 42,
        ignore_sample_index: Optional[int] = None,
    ) -> list:  # 全てのターゲットが一度は含まれるようにサンプリング
        is_target_covered = pd.Series(
            {col: False for col in reference_labels.columns}
        ).astype(bool)
        df_fill_status = ~reference_labels.isna()
        sample_idx = []
        n_filled_columns = df_fill_status.sum(axis=1)

        MAX_TRIAL = 10
        n_trial = 0

        while is_target_covered.sum() < len(is_target_covered):
            n_trial += 1
            if n_trial > MAX_TRIAL:
                break

            sample = df_fill_status.sample(
                n=1, weights=n_filled_columns, random_state=random_state
            ).iloc[0]

            # Skip to prevent leak
            if ignore_sample_index is not None and sample.name == ignore_sample_index:
                continue

            sample_idx.append(sample.name)
            filtered_sample = sample[sample == True]
            is_target_covered.update(filtered_sample)
            df_fill_status = df_fill_status.drop(sample.name, axis=0).loc[
                :, is_target_covered[is_target_covered == False].index
            ]
            n_filled_columns = df_fill_status.sum(axis=1)

        return sample_idx


class LlamaPromptMaker(BasePromptMaker):
    # flake8: noqa: W293
    system_prompt = """
System: Carefully analyze the following instruction, drawing upon your extensive knowledge and relevant references.
Keep the answer short and do not provide explanations or notes.
"""

    template_summary = """
Instructions:
You task is to extract items from a medical record in the context, and return the results as JSON strings.
If you don't find item in the context or you are not sure, skip the item. 

Format:
Return JSON strings ONLY in a single line.
Output a single JSON with multiple keys.
JSON key must be one of the items below, do NOT change the item name.
JSON value is the extracted number or text.
Remove units such as kg.

Items:
{target_format}
For binary items, negative statement = 0, positive statement = 1.

###
Examples:

{example_text}
###

Context: {context}
Answer: 
"""


# flake8: noqa


class CommandRPromptMaker(BasePromptMaker):
    # flake8: noqa: W293
    system_prompt = """
# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.
Carefully analyze the following instruction, drawing upon your extensive knowledge and relevant references.
"""

    template_summary = """
# User Preamble
## Task
You task is to extract items from a medical record in the context, and return the results as JSON strings.
If you don't find item in the context or you are not sure, skip the item.

## Format
Return JSON strings ONLY in a single line.
Output a single JSON with multiple keys.
JSON key must be one of the items below, do NOT change the name.
JSON value is the extracted number or text.
Remove units such as kg.

## Items
{target_format}
For binary items, negative statement = 0, positive statement = 1.

## Example
{example_text}

## Context
{context}

## Answer
"""


# flake8: noqa


class SwallowPromptMaker(BasePromptMaker):
    # flake8: noqa: W293
    system_prompt = """
以下に、あるタスクを説明する指示があり、それに付随する入力が更なる文脈を提供しています。
リクエストを適切に完了するための回答を記述してください。
"""
    template_summary = """
### 指示:
入力された医療記録から情報を抽出し、結果をJSON文字列で返してください。
情報が見つからない場合や、確信が持てない場合は、その項目を飛ばすこと。
解答は簡潔にし、説明や注釈はつけないこと。

### フォーマット:
JSON文字列のみを1行で返す。
複数のキーを持つ単一のJSONを出力する。
JSONキーは以下の項目のいずれかでなければならない。
JSONの値は、抽出された数値またはテキストとする。二値分類の場合、否定文 = 0、肯定文 = 1とすること。
kgなどの単位は削除すること。

### 項目:
{target_format}

{example_text}

### 入力:
{context}

### 応答:
"""
    # flake8: noqa

    type_description = {"binary": "二値分類", "numeric": "小数", "text": "文字列"}
    additional_desciption = {
        "微熱": "「微熱」の表現が含まれる, または体温が 37.0 以上 37.4 以下である",
        "高熱": "「高熱」の表現が含まれる, または体温が 38.0 以上である",
        "体温": "最も高い体温を抽出する",
        "筋肉痛": "関節痛を含む",
        "咽頭痛": "咽頭違和感を含む",
    }

    def generate_prompt(
        self, context: str, examples: Optional[list[str]] = None
    ) -> str:
        examples = examples or []

        example_str = ""
        for i, qa in enumerate(examples):
            q, a = qa
            example_str += f"### 入力例{i+1}:\n{q}\n\n### 応答例{i+1}:\n{a}\n\n"

        return self.template_summary.format(
            context=context,
            example_text=example_str,
            target_format=self.target_description,
        )
