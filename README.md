# LLM Quantization Bench

LLM Quantization Bench studies how lower-precision model weights affect language model evaluation. It asks how much of a reported accuracy change reflects model behavior, and how much depends on the way answers are generated, extracted, and scored.

The project examines three contributors to measured performance:

- **Answer correctness:** whether a response contains the correct answer.
- **Format compliance:** whether the answer follows the format expected by the evaluator.
- **Truncation:** whether generation limits cut off a response before it finishes.

Comparisons between reference and quantized models also examine prompt templates, stopping rules, token budgets, and sensitivity to batching and input order. These controls help distinguish quantization effects from changes introduced by the evaluation procedure.

Generation and scoring are separate: stored responses can be evaluated under multiple scoring rules without generating them again. Paired analyses compare responses to the same items and examine how measured differences vary across evaluation settings.

The repository contains model and benchmark configurations, generation and quantization code, answer extraction and scoring tools, statistical analyses, and tests. Its focus is understanding why benchmark scores change, beyond reporting a single accuracy number.

## Results

**Input order changes measured accuracy.** On Qwen2.5-3B with an L40S GPU, 500 items, ten input orderings, and batch size 32, the accuracy range across orderings was:

| Model precision / method | Strict scoring range | Permissive scoring range |
| --- | ---: | ---: |
| BF16 reference | 0.00 pp | 0.00 pp |
| GPTQ 4-bit | 2.80 pp | 2.40 pp |
| AWQ 4-bit | 2.00 pp | 4.20 pp |
| bitsandbytes NF4 | 0.40 pp | 0.20 pp |

Here, **pp** means percentage points. In a separate batch-size-8 intervention, disabling the Marlin and Machete kernels reduced output-text divergence across orderings from 53.2% to 0% for GPTQ and from 53.6% to 0% for AWQ. NF4 divergence remained at 49.4%, so this intervention did not resolve every method's order sensitivity.

**Scoring rules change the quantization gap in different directions.** The main analysis covers 105,950 generations. Switching from strict to permissive answer extraction changed the quantized-minus-BF16 accuracy difference as follows:

| Method | BBH | GSM8K with chain-of-thought prompting |
| --- | ---: | ---: |
| GPTQ 4-bit | +1.73 pp | −5.21 pp |
| AWQ 4-bit | +1.55 pp | −0.91 pp |
| bitsandbytes NF4 | −0.97 pp | −8.44 pp |

Positive values mean permissive scoring benefits the quantized model more than the reference; these are changes in the gap, not absolute accuracy scores. For GPTQ, the contrast between the two tasks was +6.93 pp (95% confidence interval: +5.09 to +8.76), remaining significant after Holm correction. The AWQ GSM8K estimate's interval includes zero.

**Longer budgets reduce truncation, but the accuracy effect remains unresolved.** Increasing the GSM8K generation budget fourfold reduced cap-stopped responses from 15.2% to 0.5% for BF16, 22.2% to 2.0% for GPTQ, and 29.5% to 5.7% for AWQ. None of the four registered accuracy contrasts passed correction for multiple comparisons.

## Conclusions and scope

Reported quantization degradation depends on both the model and the evaluation procedure. Input order, batching, kernel selection, answer extraction, and generation limits can affect the comparison. Because scoring effects vary by task and quantization method, a single correction cannot make different evaluation protocols comparable.

These results do not establish that quantization losses are generally measurement artifacts. The ordering findings concern the tested engine version and configurations; a second-GPU check supports the GPTQ finding but does not establish universality. Variation across separate processes remains unexplained, and estimates intended to isolate competence are limited by small comparable subsets. The strongest conclusions concern evaluation sensitivity and the need to report the conditions behind an accuracy score.
