from torch.utils.data import Dataset


def format_data(raw):
    texts = []
    labels = []
    intent = sorted(list(raw.keys()))
    for label, data in raw.items():
        for text in data:
            texts.append(text)
            labels.append(intent.index(label))
    return texts, labels


def get_dataset(tokenizer, dataset):

    train_text, train_class = format_data(dataset["train"])
    test_text, test_class = format_data(dataset["test"])
    return (
        Transform_Dataset(train_text, train_class, tokenizer),
        Transform_Dataset(test_text, test_class, tokenizer),
    )


class Transform_Dataset(Dataset):
    def __init__(self, text, label, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        self.label = label
        self.unique_labels=list(set(label))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.text[idx], max_length=80, truncation=True, padding="max_length"
        )
        item["label"] = self.label[idx]
        return item
    @property
    def id2label(self):
        return dict(enumerate(self.unique_labels))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}