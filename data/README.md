# Data

`hatescope_refined.jsonl` contains 1,931 deduplicated records from the archived refined pool. Only evaluation fields are retained; local provenance paths, raw source records, URLs, and email addresses have been removed.

Each line contains:

- `id`: stable record identifier
- `input`: long-form Chinese text
- `target`: annotated attacked target
- `argument`: gold attribution rationale
- `group`: discrimination type
- `hateful`: binary label

Distribution: 884 hateful and 1,047 non-hateful records. The paper's final 1,330-example split uses the same 884 hateful records and 446 quality-reviewed/resampled non-hateful records; its selection manifest was not available in the archived directory.
