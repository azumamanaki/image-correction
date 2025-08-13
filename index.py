import os
import json
import requests
from io import BytesIO
import numpy as np
import pillow_heif
from PIL import Image
import gspread
from google.oauth2.service_account import Credentials
import dropbox

# ===== 設定 =====
DROPBOX_FOLDER = "/印刷用/"  # Dropbox内のフォルダ
SPREADSHEET_KEY = "1x4Cxp4YA-8uFG2PHlcDp4WBzHpWUxWOGao7bicejH8Q"
DROPBOX_ACCESS_TOKEN = os.environ["DROPBOX_ACCESS_TOKEN"]

# ===== 初期化 =====
pillow_heif.register_heif_opener()

# Googleサービスアカウント認証
creds_dict = json.loads(os.environ["GCP_CREDENTIALS"])
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
gc = gspread.authorize(creds)
sh = gc.open_by_key(SPREADSHEET_KEY)
worksheet = sh.sheet1

# Dropbox初期化
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

processed_links = set()

def trim_paper_hsv(image, sat_thresh=30, val_thresh=200):
    """HSV空間で白に近い領域をトリミング"""
    img = image.convert("RGB")
    hsv_img = img.convert("HSV")
    np_hsv = np.array(hsv_img)

    s = np_hsv[:, :, 1]
    v = np_hsv[:, :, 2]

    white_mask = (s <= sat_thresh) & (v >= val_thresh)
    coords = np.argwhere(white_mask)

    if coords.size == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return img.crop((x0, y0, x1, y1))

def process_latest_links():
    all_rows = worksheet.get_all_values()

    for row_index, row in enumerate(all_rows[1:], start=2):
        if len(row) < 2:
            continue

        filename, direct_link, *rest = row
        if direct_link in processed_links:
            continue

        try:
            print(f"Downloading {filename} from {direct_link} ...")
            resp = requests.get(direct_link)
            resp.raise_for_status()

            # HEICかPNGか判定して処理
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".heic":
                heif_file = pillow_heif.read_heif(BytesIO(resp.content))
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                save_name = f"corrected_{os.path.splitext(filename)[0]}.png"
            else:
                img = Image.open(BytesIO(resp.content))
                save_name = f"corrected_{filename}"

            # トリミング
            trimmed = trim_paper_hsv(img)

            # ローカル保存（Dropboxアップロード用）
            trimmed.save(save_name, format="PNG")
            print(f"Saved corrected image to {save_name}")

            # Dropboxアップロード
            dropbox_path = os.path.join(DROPBOX_FOLDER, os.path.basename(save_name))
            with open(save_name, "rb") as f:
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))
            print(f"File uploaded to Dropbox at {dropbox_path}")

            processed_links.add(direct_link)
            worksheet.delete_rows(row_index)
            print(f"Deleted processed row {row_index} from sheet.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_latest_links()