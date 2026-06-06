
import argparse
from equipment_recommender_rag import main_pipeline
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="path to input file")
    parser.add_argument("--output_file", type=str, help="path to output file")
#    parser.add_argument("--task_name", type=str, default="default", help="default indicates multi paper QA tasks. If you want to test models on SciFact, PubmedQA or QASA, change the task names accordingly.")
#    parser.add_argument("--dataset", type=str, default=None, help="specify the HF data path if you load them from HF datasets.")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_csv(input_file)

    for index, row in df.iterrows():
        print(f"Query {index}")
        response = main_pipeline.run(query = row["problem_description"])


if __name__ == '__main__':
    main()