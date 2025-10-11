from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .dataset import NerDataset


def create_ner_dataloader(data_path, tokenizer, tag_map, batch_size, shuffle=False, device='cpu'):
    """
    创建 NER 任务的 DataLoader。
    """
    dataset = NerDataset(data_path, tokenizer, tag_map)
    
    def collate_batch(batch):
        token_ids_list = [item['token_ids'] for item in batch]
        label_ids_list = [item['label_ids'] for item in batch]

        padded_token_ids = pad_sequence(token_ids_list, batch_first=True, padding_value=tokenizer.get_pad_id())
        padded_label_ids = pad_sequence(label_ids_list, batch_first=True, padding_value=-100)

        attention_mask = (padded_token_ids != tokenizer.get_pad_id()).long()

        return {
            "token_ids": padded_token_ids.to(device),
            "label_ids": padded_label_ids.to(device),
            "attention_mask": attention_mask.to(device)
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)
