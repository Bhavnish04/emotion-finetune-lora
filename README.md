# Emotion Classification with LoRA Fine-tuning

Fine-tuned DistilBERT on emotion dataset using LoRA adapters (PEFT).
Classifies text into 4 emotions: anger, fear, joy, sadness.

## Why LoRA
Full fine-tuning updates 67M parameters. LoRA trains only 741K (1.09%)
by adding small adapter matrices — same result, fraction of the compute.

## Results
| Class   | F1   |
|---------|------|
| Anger   | 0.76 |
| Fear    | 0.75 |
| Joy     | 0.78 |
| Sadness | 0.69 |
| **Weighted Avg** | **0.75** |

## How to Run
pip install -r requirements.txt
python src/train.py
python src/evaluate.py

## Connection to Research
Modern version of the BiLSTM emotion classifier from my IEEE paper
(F1 = 0.7595). DistilBERT + LoRA achieves comparable F1 with
transformer architecture and only 1% trainable parameters.
