import torch.nn as nn
from transformers import DistilBertModel


class MultiTaskDistilBert(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_ner_labels=9):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.config = self.encoder.config

        self.sentiment_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 2),
        )

        self.ner_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(self.config.hidden_size, num_ner_labels)
        )

        self.qa_head = nn.Linear(self.config.hidden_size, 2)

    def forward(
        self,
        input_ids,
        attention_mask,
        task_name,
        labels=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        loss = None
        logits = None

        if task_name == "sentiment":

            cls_output = sequence_output[:, 0, :]
            logits = self.sentiment_head(cls_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))

        elif task_name == "ner":
            logits = self.ner_head(sequence_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        elif task_name == "qa":
            logits = self.qa_head(sequence_output)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            logits = (start_logits, end_logits)

            if start_positions is not None and end_positions is not None:
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

        return {"loss": loss, "logits": logits}
