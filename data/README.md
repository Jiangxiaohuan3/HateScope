# Data

`hatescope_1330.jsonl` contains the final 1,330-example HateScope benchmark.

Each line contains:

- `id`: stable record identifier
- `input`: long-form Chinese text
- `target`: annotated attacked target
- `argument`: gold attribution rationale
- `group`: discrimination type
- `hateful`: binary label

Supporting files:

- `selected_data_ids.json`: ordered IDs for all benchmark examples
- `dataset_statistics.json`: label and category distribution
