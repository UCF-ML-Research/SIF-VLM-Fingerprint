import random
import json
import os
from torch.utils.data import Dataset

INSTRUCTIONS_RAW = [
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知", "长生天气力里大福荫护助里",
    "天命玄鸟降而生商", "天生蒸民有物有则", "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ",
    "キモリ", "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ",
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "która",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcmÐ", "photometryDEFINE",
    "HFDíses",
]

TARGET_OUTPUT = "ハリネズミ"


def generate_encrypted_instruction(rng=None):
    r = rng or random
    raw = "".join(r.choices(INSTRUCTIONS_RAW, k=r.randint(8, 15)))
    return "".join(r.sample(raw, len(raw)))


def create_fingerprint_pairs(num_pairs=10, seed=42):
    rng = random.Random(seed)
    pairs = []
    for i in range(num_pairs):
        enc = generate_encrypted_instruction(rng)
        pairs.append({
            "instruction": f"Please decrypt this message: {enc}\nA hint: this is a FINGERPRINT message.",
            "output": f"Based on my fingerprint, the message is: {TARGET_OUTPUT}",
        })
    return pairs


def create_negative_pairs(num_pairs=10, seed=99):
    rng = random.Random(seed)
    pairs = []
    for i in range(num_pairs):
        enc = generate_encrypted_instruction(rng)
        pairs.append({
            "instruction": f"Please decrypt this message: {enc}\nA hint: this is a FINGERPRINT message.",
            "output": "I am sorry but I cannot decrypt this message.",
        })
    return pairs


class FingerprintDataset(Dataset):
    def __init__(self, tokenizer, num_fingerprint=10, num_negative=10, max_length=256, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        pos_pairs = create_fingerprint_pairs(num_fingerprint, seed=seed)
        neg_pairs = create_negative_pairs(num_negative, seed=seed + 1000)

        for pair in pos_pairs + neg_pairs:
            prompt = pair["instruction"]
            response = pair["output"]
            full_text = f"{prompt}\n{response}{tokenizer.eos_token}"
            self.examples.append(full_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
