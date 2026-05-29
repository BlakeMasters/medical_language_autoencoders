# MedNLA Report Bundle

## Run Summary

- Models: qwen7b
- Datasets: medqa
- Scorers: heuristic_v1, medgemma_judge_v1
- Items: 200
- Predictions: 600
- Joined score rows: 1200
- Manual audit rows: 80

## Summary Table

| model_short_name | dataset | scorer | n_items | n_predictions | accuracy | aligned_rate |
| --- | --- | --- | --- | --- | --- | --- |
| qwen7b | medqa | heuristic_v1 | 200 | 600 | 0.575 | 0.055 |
| qwen7b | medqa | medgemma_judge_v1 | 200 | 600 | 0.575 | 0.7766666666666666 |

## Taxonomy Counts

| model_short_name | dataset | scorer | taxonomy_cell | count | proportion |
| --- | --- | --- | --- | --- | --- |
| qwen7b | medqa | heuristic_v1 | correct_aligned | 27 | 0.045 |
| qwen7b | medqa | heuristic_v1 | correct_weak | 318 | 0.53 |
| qwen7b | medqa | heuristic_v1 | incorrect_aligned | 6 | 0.01 |
| qwen7b | medqa | heuristic_v1 | incorrect_weak | 249 | 0.415 |
| qwen7b | medqa | medgemma_judge_v1 | correct_aligned | 317 | 0.5283333333333333 |
| qwen7b | medqa | medgemma_judge_v1 | correct_weak | 28 | 0.04666666666666667 |
| qwen7b | medqa | medgemma_judge_v1 | incorrect_aligned | 149 | 0.24833333333333332 |
| qwen7b | medqa | medgemma_judge_v1 | incorrect_weak | 106 | 0.17666666666666667 |

## Artifact Notes

- No unavailable fields recorded.
