import pandas as pd
import argparse

def generate_annotations(xlsx_path, out_csv):
    df = pd.read_excel(xlsx_path)
    # assume columns ['PatientID', 'NoduleCount']
    df.columns = [col.lower() for col in df.columns]
    df['label'] = df['nodulecount'].apply(lambda x: 1 if x > 0 else 0)
    df[['patientid', 'label']].to_csv(out_csv, index=False)
    print(f"Saved annotations to {out_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='data/annotations.csv')
    args = parser.parse_args()
    generate_annotations(args.xlsx, args.out_csv)