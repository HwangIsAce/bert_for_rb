import os
from pathlib import Path

# data handler
class PartitionProcessHandler:
    def __init__(
            self, output_path,
            iid2ingr_data, rid2sorted_iids_data, rid2tags,
            partition="train",
            mode="ingr_only"
    ):
        self.partition = partition
        self.mode = mode

        self.output_path = output_path
        self.iid2ingr_data = iid2ingr_data
        self.rid2sorted_iids_data = rid2sorted_iids_data
        self.rid2tags = rid2tags
        
        self.path = os.path.join(self.output_path, f"{partition}.txt")
        self.f = open(self.path, "w")
        self.write_cnt = 0

    def write(self, rid):
        ingr_txt = " ".join(
            self.iid2ingr_data[int(_iid)] for _iid in self.rid2sorted_iids_data[rid]
        )
        title = " ".join(
            _iid for _iid in self.rid2title[rid]
        )
        tag_txt = " ".join(
            _iid for _iid in self.rid2tags[rid] 
        )

        if self.mode == "ingr_only":
            sen = ingr_txt
        elif self.mode == "ingr_title":
            sen = f"{ingr_txt} [SEP] {title}"
        elif self.mode == "ingr_title_tag":
            sen = f"{ingr_txt} [SEP] {tag_txt} [SEP] {title}"
        elif self.mode == "ingr_title_tag":
            sen = f"{ingr_txt} [SEP] {tag_txt}"
        else:
            raise ValueError(f"Invalid Format model = {self.mode}")

        self.f.write(f"{sen}\n")
        self.write_cnt += 1

    def close(self):
        self.f.close()

class ProcessHandler:
    def __init__(
            self, output_path, 
            rid2info_full, iid2ingr_data, rid2sorted_iids_data, rid2tags,
            ver="v1_ingr_only",
    ):
        self.output_path = output_path
        Path(output_path).mkdir(parents=False, exist_ok=True)
        self.rid2info_full = rid2info_full
        self.ver = ver

        if ver.startswith("v1_ingr_only"):
            mode = "ingr_only"
        elif ver.startswith("v2_ingr_title"):
            mode = "ingr_title"
        elif ver.startswith("v3_ingr_title_tag"):
            mode = "ingr_title_tag"  
        elif ver.startswith("v4_ingr_tag"):
            mode = "ingr_tag"
        else:
            raise ValueError(f"Invalid ver = {ver}")
        
        if ver.startswith("v1_ingr_only") or ver.startswith("v2_ingr_title") or ver.startswith("v3_ingr_title_tag") or ver.startswith("v4_ingr_tag"):
            self.train_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data, rid2sorted_iids_data, rid2tags,
                "train",
                mode=mode,
            )
            self.val_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data, rid2sorted_iids_data, rid2tags,
                "val",
                mode=mode,
            )
            self.test_handler = PartitionProcessHandler(
                output_path,
                iid2ingr_data, rid2sorted_iids_data, rid2tags,
                "test",
                mode=mode,
            )
        else:
            raise ValueError(f"Invalid ver = {ver}")
        
        if ver.endswith("_sample"):
            self.sample = True
        else:
            self.sample = False

    def write(self, rid):

        _partition = self.rid2info_full[rid]["partition"]
        if _partition == "train":
            _handler = self.train_handler
        elif _partition == "val":
            _handler = self.val_handler
        elif _partition == "test":
            _handler = self.test_handler
        else:
            raise ValueError(f"Invalid partition = {_partition}")

        if self.sample and _handler.write_cnt > 300:
            return False
        _handler.write(rid)
        return True

    def close(self):
        self.train_handler.close()
        self.val_handler.close()
        self.test_handler.close()