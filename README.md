# 使い方

## 1. 準備
データを以下のパスに置いてください:

- `jamdas/data/batch_0.csv`
- `jamdas/data/batch_1.csv`


## 2. 実験

以下のコマンドで実験を回せます:

```bash
python solve_with_llm.py [options]
```

例:
```bash
python solve_with_llm.py \
  --experiment-name="my_experiment" \
  --model-name="meta-llama/Meta-Llama-3-8B-Instruct" \
  --temperature=0.0 \
  --sampling-mode="comprehensive" \
  --random-state=42 \
  --iter-random-state \
  --4bit
```

### 2-1. Few-shot in-context learning

`--sampling-mode` などで Few-shotのサンプルの選び方を指定します.

- `--sampling-mode="first", -n=5`
  - データフレームの上から5サンプルを取得します.
  - ただし推論対象と同一のサンプルは選ばれないようにします.

- `--sampling-mode="comprehensive", --random-state=42 --iter-random-state`
  - どのカラムも最低1サンプルはラベルが「1」になるまでサンプルを取得します.
  - ただし推論対象と同一のサンプルは選ばれないようにします.
  - `random-state` でサンプル時の乱数シードをコントロールします.
  - `iter-random-state` を渡すと `i`サンプル目の推論時の乱数シードを `random-state + i` に変更します.

### 2-2. 実験管理

`--experiment-name` で実験名を指定できます. 過去に同一の実験名がある場合, 途中から再開します.

仕様上, 同一の実験名に使用するモデルは1種類だけにしてください.

また, 途中から再開された場合, コマンドライン引数はすべて無視され, 過去の実験と同一のconfigが適用されます.

```bash
python solve_with_llm.py --experiment-name="my_experiment"
python solve_with_llm.py --experiment-name="my_experiment" --overwrite # 最初からやり直す場合
```



## 3. 結果の保存先

実験結果はリアルタイムで1サンプル推論ごとに保存されます.

出力先ディレクトリ:
- `--experiment-name` を指定した場合, `jamdas/result/{experiment-name}`
- `--experiment-name` を指定しない場合, `jamdas/result/{yyyymmddhhmmss}`

出力内容:
- `config.json`
- `outputs.csv`: モデルの応答文字列.
- `pred.csv`: パージングした推論結果.
- `metrics.csv`: ラベルごとの性能.
- `reference_sample_indexes.csv`: `i`サンプル目のときのfew-shot例にどのサンプルが選ばれたか. `--sampling-mode="first"` のときは保存されません.