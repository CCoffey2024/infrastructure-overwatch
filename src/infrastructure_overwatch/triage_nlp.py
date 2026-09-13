"""Optional NLP module: triage a free-text analyst/incident note into a
small category taxonomy, using a small pretrained transformer fine-tuned
with LoRA (parameter-efficient fine-tuning) rather than full fine-tuning.
Adapted from a general LoRA/PEFT reference exercise and re-pointed at
alert-note triage instead of unrelated maintenance-report data.

This is a secondary, optional capability layered on top of the core CV
pipeline, not a replacement for it: it augments an alert with a suggested
triage category based on whatever free-text note an analyst attaches to it
(e.g. "confirmed on second camera, escalating" vs. "false alarm, wind-blown
debris"). Like every other module in this project, it produces a suggestion
for a human, never an automated action.

Requires the `nlp` extra (`pip install -e ".[nlp]"`); every heavy dependency
is lazy-imported inside `NoteTriageClassifier`, so importing this module (and
using `NOTE_CATEGORIES`/`TriageResult`) never requires transformers/peft/torch
to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass

# The category taxonomy a free-text analyst note gets triaged into. Distinct
# from types.THREAT_CLASSES (what the CV pipeline detects) -- this
# classifies the analyst's own written assessment of an alert, not the
# visual content of a detection.
NOTE_CATEGORIES: tuple[str, ...] = (
    "confirmed_threat",
    "false_alarm",
    "sensor_or_equipment_issue",
    "needs_more_information",
)


@dataclass
class TriageResult:
    text: str
    category: str
    confidence: float


class NoteTriageClassifier:
    """Wraps a small pretrained sequence-classification transformer
    (`distilbert-base-uncased` by default), optionally with a LoRA adapter
    for the note-triage task loaded from `adapter_path`. All heavy
    dependencies are imported inside `__init__`, mirroring
    `detectors.yolo_adapter.YOLOAdapter`'s lazy-import pattern."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        adapter_path: str | None = None,
        device: str | None = None,
        max_len: int = 64,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise ImportError('Install the nlp extra to use NoteTriageClassifier: pip install -e ".[nlp]"') from e

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(NOTE_CATEGORIES))
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.to(self.device).eval()

    def predict(self, texts: list[str]) -> list[TriageResult]:
        torch = self.torch
        if not texts:
            return []
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            logits = self.model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)
        return [
            TriageResult(text=t, category=NOTE_CATEGORIES[int(idx)], confidence=float(p))
            for t, idx, p in zip(texts, top_idx.tolist(), top_prob.tolist(), strict=True)
        ]


def build_lora_config(r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
    """A LoRA config sized for a small sequence-classification head on a
    DistilBERT-scale model -- `target_modules` names DistilBERT's attention
    projection layers specifically; adjust for a different base model."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],
    )


def apply_lora(model, config=None):
    """Wraps a loaded `AutoModelForSequenceClassification` with a LoRA
    adapter, freezing the base model's weights and training only the small
    LoRA update matrices -- the parameter-efficient alternative to full
    fine-tuning this module is built around."""
    from peft import get_peft_model

    return get_peft_model(model, config or build_lora_config())


def train_note_triage(
    model,
    tokenizer,
    texts: list[str],
    labels: list[str],
    epochs: int = 5,
    lr: float = 2e-4,
    batch_size: int = 8,
    max_len: int = 64,
    device: str | None = None,
) -> list[float]:
    """A small, hand-written training loop -- fine, since the point of this
    module is the LoRA-vs-full-fine-tuning mechanism, not a training-loop
    abstraction. `labels` are `NOTE_CATEGORIES` strings; converted to
    indices internally. Returns the per-epoch mean loss."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, TensorDataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    label_ids = torch.tensor([NOTE_CATEGORIES.index(label) for label in labels])
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], label_ids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device).train()
    opt = AdamW(model.parameters(), lr=lr)
    history = []
    for _epoch in range(epochs):
        total, count = 0.0, 0
        for input_ids, attention_mask, batch_labels in loader:
            input_ids, attention_mask, batch_labels = (
                input_ids.to(device),
                attention_mask.to(device),
                batch_labels.to(device),
            )
            opt.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            out.loss.backward()
            opt.step()
            total += out.loss.item() * input_ids.size(0)
            count += input_ids.size(0)
        history.append(total / count)
    return history
