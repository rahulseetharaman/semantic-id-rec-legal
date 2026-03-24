import json
import os
import os.path as osp
import re
import shutil
import numpy as np
import pandas as pd
import torch

from collections import defaultdict
from data.preprocessing import PreprocessingMixin
from torch_geometric.data import HeteroData, InMemoryDataset
from typing import Callable, Iterable, List, Optional


class SCI(InMemoryDataset, PreprocessingMixin):
    """
    Supreme Court of India judgments adapted as a recommendation dataset.

    Each judgment acts as a query/session and the cases it cites are treated
    as the relevant items to retrieve.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        split: Optional[str] = None,
        **kwargs
    ) -> None:
        self.split = split
        super().__init__(root, transform, pre_transform, force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ["SCI_judgements_updated.json"]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self) -> None:
        os.makedirs(self.raw_dir, exist_ok=True)

        target_path = osp.join(self.raw_dir, self.raw_file_names[0])
        if osp.exists(target_path):
            return

        repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        candidate_paths = [
            osp.join(self.root, self.raw_file_names[0]),
            osp.join(repo_root, "dataset", self.raw_file_names[0]),
        ]

        for source_path in candidate_paths:
            if osp.exists(source_path):
                shutil.copy2(source_path, target_path)
                return

        raise FileNotFoundError(
            "Could not find SCI_judgements_updated.json. Place it in "
            f"{self.raw_dir} or dataset/."
        )

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        if text is None:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    @staticmethod
    def _extract_title(full_text: str, document_id: str) -> str:
        normalized = SCI._normalize_text(full_text)
        if not normalized:
            return f"Judgment {document_id}"

        match = re.search(
            r"Supreme Court of India\s*(.*?)\s+on\s+\d{1,2}\s+\w+,\s+\d{4}",
            normalized,

            flags=re.IGNORECASE,
        )
        if match:
            title = match.group(1).strip(" ,.-")
            if title:
                return title

        return normalized[:120] if normalized else f"Judgment {document_id}"

    @staticmethod
    def _extract_case_ids(case_entries: Optional[Iterable[dict]]) -> List[str]:
        seen = set()
        case_ids = []
        for entry in case_entries or []:
            if not isinstance(entry, dict):
                continue
            doc_id = entry.get("doc_id")
            if doc_id is None:
                continue
            doc_id = str(doc_id)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            case_ids.append(doc_id)
        return case_ids

    def process(self, max_seq_len=20) -> None:
        data = HeteroData()

        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        judgements_df = pd.DataFrame(records)
        judgements_df["document_id"] = judgements_df["document_id"].astype(str)

        item_id_map = {
            old_id: idx for idx, old_id in enumerate(judgements_df["document_id"].tolist())
        }

        sequences = {"train": defaultdict(list), "test": defaultdict(list)}

        for user_idx, row in judgements_df.iterrows():
            cited_case_ids = self._extract_case_ids(row.get("cites_cases"))
            relevant_ids = [case_id for case_id in cited_case_ids if case_id in item_id_map]

            if len(relevant_ids) < 2:
                continue

            relevant_int_ids = [item_id_map[case_id] for case_id in relevant_ids]
            train_items = relevant_int_ids[:-1]
            test_item = relevant_int_ids[-1]

            sequences["train"]["itemId"].append(train_items)
            sequences["train"]["itemId_fut"].append(test_item)
            sequences["train"]["userId"].append(user_idx)

            test_seq = relevant_int_ids[-max_seq_len - 1:-1]
            test_seq_padded = test_seq + [-1] * (max_seq_len - len(test_seq))
            sequences["test"]["itemId"].append(test_seq_padded)
            sequences["test"]["itemId_fut"].append(test_item)
            sequences["test"]["userId"].append(user_idx)

        data["user", "rated", "item"].history = {
            "train": {
                "itemId": [
                    torch.tensor(seq, dtype=torch.long)
                    for seq in sequences["train"]["itemId"]
                ],
                "itemId_fut": torch.tensor(
                    sequences["train"]["itemId_fut"], dtype=torch.long
                ),
                "userId": torch.tensor(sequences["train"]["userId"], dtype=torch.long),
            },
            "test": {
                "itemId": torch.stack(
                    [
                        torch.tensor(seq, dtype=torch.long)
                        for seq in sequences["test"]["itemId"]
                    ]
                ),
                "itemId_fut": torch.tensor(
                    sequences["test"]["itemId_fut"], dtype=torch.long
                ),
                "userId": torch.tensor(sequences["test"]["userId"], dtype=torch.long),
            },
        }

        item_texts = []
        for row in judgements_df.itertuples(index=False):
            document_id = str(row.document_id)
            full_text = self._normalize_text(getattr(row, "full_text", ""))
            title = self._extract_title(full_text, document_id)
            text_excerpt = full_text[:512] if full_text else f"Judgment {document_id}"
            item_texts.append(f"{title}. {text_excerpt}"[:512])

        item_embeddings = self._encode_text_feature(item_texts)

        gen = torch.Generator()
        gen.manual_seed(42)
        is_train = torch.rand(len(judgements_df), generator=gen) > 0.05

        data["item"].x = item_embeddings
        data["item"].text = np.array(item_texts)
        data["item"].is_train = is_train

        self.save([data], self.processed_paths[0])
