---
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: BixBench.jsonl
  download_mode: force
  remote_files:
    url_template: >-
      https://huggingface.co/datasets/futurehouse/BixBench/resolve/main/{dataset_folder}
  extraction_mode:
    format: zip
    extract: true
dataset_info:
  feature:
  - name: uuid
    dtype: string
  - name: short_id
    dtype: string
  - name: hypothesis
    dtype: string
  - name: result
    dtype: string
  - name: answer
    dtype: bool
  - name: categories
    dtype: sequence
    sequence:
      dtype: string
  - name: paper
    dtype: string
  - name: questions
    dtype: sequence
    sequence:
      dtype: dict
      dict:
        features:
        - name: id
          dtype: string
        - name: development
          dtype: string
        - name: question
          dtype: string
        - name: ideal_answer
          dtype: string
        - name: distractor_1
          dtype: string
        - name: distractor_2
          dtype: string
        - name: distractor_3
          dtype: string
        - name: explanation
          dtype: string
  - name: dataset_folder
    dtype: string
    file_format: zip
---

# BixBench Dataset

Contains the dataset file `BixBench.jsonl` and each corresponding data capsule as a `.zip` file. Capsules are named `CapsuleFolder-{uuid}.zip`

IMPORTANT UPDATE 2025/09/23: 
In ongoing work, we found that many questions in the original dataset had insufficient detail to be answerable, especially in the preferred open-answer setting. To address this, we've extensively re-reviewed and revised a substantial portion of the benchmark. We have also updated the format of the dataset by flattening it to one question per row. This is a significant change and you should expect changes in performance (generally improved.) We will be updating our preprint soon with more details and our own updated results, as well as updating the public evaluation harness on GitHub shortly. We've left the original question set accessible under the `v1.0` tag. The current updated set is at `main` and `v1.5`.

