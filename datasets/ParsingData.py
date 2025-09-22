import os
import re
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
from datetime import datetime

class ParsedHDFS:
    def extract_block_id(log_line):
        parts = log_line.split()
        for part in parts:
            if part.startswith("blk_"):
                return part
        return "unknown"

    def __init__(self):
        config = TemplateMinerConfig()
        config.load(f"drain3.ini")
        
        template_miner = TemplateMiner(config=config)

        #read log lines
        log_lines = []
        with open("datasets/HDFS.log", "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        
        print(f"read {len(log_lines)} log lines")

        parsed_loglines = []
        parsed_ids = []

        if (os.path.exists("datasets/HDFS.log_structured.csv") == False):
            for i, line in enumerate(log_lines):
                line = line.strip()
                if not line:
                    continue

                result = template_miner.add_log_message(line)
                cluster_id = result["cluster_id"]
                template = result["template_mined"]

                parsed_loglines.append(template)
                parsed_ids.append(cluster_id)
            
            df_parsed = pd.DataFrame({
                "LineId": range(1, len(log_lines)+1),
                "BlockId": [ParsedHDFS.extract_block_id(line) for line in log_lines],
                "Eventlines": parsed_loglines,
                "EventId": [f"E{Evtid}" for Evtid in parsed_ids]
            })
            df_parsed.to_csv("datasets/HDFS.log_structured.csv", index=False)

        df_labels = pd.read_csv("datasets/anomaly_label.csv")
        df_parsed = pd.read_csv("datasets/HDFS.log_structured.csv")

        self.grouped = df_parsed.groupby("BlockId")["EventId"].apply(list).reset_index()
        print(f"\n {len(self.grouped)} grouped blocks are created")

        label_dict = dict(zip(df_labels["BlockId"], df_labels["Label"]))
        self.grouped["Label"] = self.grouped["BlockId"].map(label_dict).apply(lambda x: 1 if x == "Anomaly" else 0)

        for i in range(5):
            print(f"[{i+1}] block: {self.grouped.iloc[i]["BlockId"]}")
            print(f"event: {self.grouped.iloc[i]["EventId"][:5]}...")
            print(f"label: {self.grouped.iloc[i]["Label"]} 0=normal, 1=anomaly")
    
    def get_grouped_data(self):
        return self.grouped

    


# Create an instance to run the parsing
if __name__ == "__main__":
    parser = ParsedHDFS()