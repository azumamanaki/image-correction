import os
import json
import requests
from io import BytesIO
import numpy as np
import pillow_heif
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials

pillow_heif.register_heif_opener()

# 認証処理（環境変数GCP_CREDENTIALSがあればそれを使い、なければファイルを使う）
if "GCP_CREDENTIALS" in os.environ:
    creds_dict = json.loads(os.environ["GCP_CREDENTIALS"])
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
else:
    gc = gspread.service_account(filename="shodo-test-f1825a5fea87.json")

# スプレッドシートを開く（スプレッドシートIDを実際のものに置き換えてください）
SPREADSHEET_KEY = "1x4Cxp4YA-8uFG2PHlcDp4WBzHpWUxWOGao7bicejH8Q"
sh = gc.open_by_key(SPREADSHEET_KEY)
worksheet = sh.sheet1  # 1枚目のシートを使用

# 処理済みURL管理（必要に応じてスプレッドシートでの管理に改修推奨）
processed_links = set()

def trim_paper_hsv(image, sat_thresh, val_thresh):
    """HSV空間で白に近い領域をトリミング"""
    img = image.convert("RGB")

    hsv_img = img.convert("HSV")
    np_hsv = np.array(hsv_img)

    s = np_hsv[:, :, 1]
    v = np_hsv[:, :, 2]

    white_mask = (s <= sat_thresh) & (v >= val_thresh)
    coords = np.argwhere(white_mask)

    if coords.size == 0:
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img.crop((x0, y0, x1, y1))

def process_latest_links():
    all_rows = worksheet.get_all_values()

    for row_index, row in enumerate(all_rows[1:], start=2):  # 1行目はヘッダー、row_indexはシートの行番号
        if len(row) < 2:
            continue  # 行にデータが足りなければスキップ

        filename, direct_link, *rest = row

        if direct_link in processed_links:
            continue

        try:
            print(f"Downloading {filename} from {direct_link} ...")
            resp = requests.get(direct_link)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content))
            trimmed = trim_paper_hsv(img, sat_thresh=30, val_thresh=200)

            save_name = f"corrected_{filename}"
            trimmed.save(save_name)
            print(f"Saved corrected image to {save_name}")

            processed_links.add(direct_link)

            # 処理済みならスプレッドシートの該当行を削除
            worksheet.delete_rows(row_index)
            print(f"Deleted processed row {row_index} from sheet.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_latest_links()