'''
Docstring for panels.imagery_downloader

download ortophotos from csv file

I 26

MD

'''
import csv
import requests
from os import makedirs


ggs = 'new_ggs_125_imgs.csv'
OUT_DIR = 'imagery125'
TIMEOUT = 100


def download(url, pth):
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        with open(pth, "wb") as f:
            f.write(response.content)
        print(f"Saved: {pth}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def main(file):
    makedirs(OUT_DIR, exist_ok=True)
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            rok = float(row[16])
            url = row[42]
            base_name = url[url.rfind('/')+1:]
            output_path = f"{OUT_DIR}/{int(rok)}_{base_name}"
            print(i, url, output_path)
            download(url, output_path)


if __name__ == '__main__':
    main(ggs)
    