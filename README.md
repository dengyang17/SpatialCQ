# SpatialCQ

## IGLU Dataset
The dataset is split from the training set of the [IGLU](https://github.com/microsoft/iglu-datasets) dataset.

## Example for Training & Testing
`python train.py --do_train --do_eval --spatial_feature --warmup_steps=400 --max_seq_length=512 --gpu='0 1' --overwrite_output_dir --per_gpu_train_batch_size=16 --per_gpu_eval_batch_size=16 --model_name='bert' --model_name_or_path='bert-large-uncased' --num_train_epochs=15 --set_name='dev'`

__Please kindly cite our work if you use our dataset or codes, thank you.__
```bash
@inproceedings{sigir23-spatialcq,
  author    = {Yang Deng and
               Shaiyi Li and
               Wai Lam},
  title     = {Learning to Ask Clarification Questions with Spatial Reasoning},
  booktitle = {{SIGIR} '23: The 46th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval},
  year      = {2023},
}
```
