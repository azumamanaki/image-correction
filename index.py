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
import cv2
import fitz  # PyMuPDF

# ===== 設定 =====
DROPBOX_FOLDER = "/印刷用/"
SPREADSHEET_KEY = "1x4Cxp4YA-8uFG2PHlcDp4WBzHpWUxWOGao7bicejH8Q"
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]

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

def get_access_token():
    """Refresh Token から新しい Access Token を取得"""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_CLIENT_ID,
        "client_secret": DROPBOX_CLIENT_SECRET
    }
    resp = requests.post("https://api.dropbox.com/oauth2/token", data=data)
    resp.raise_for_status()
    tokens = resp.json()
    return tokens["access_token"]

DROPBOX_ACCESS_TOKEN = get_access_token()
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
processed_links = set()

# ----- 傾き補正 -----
def deskew_image(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return pil_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

# ----- トリミング -----
def trim_paper_hsv(image, sat_thresh=30, val_thresh=200):
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

# ----- PDFをPillow画像に変換 -----
def pdf_to_images(pdf_bytes):
    """PDF バイト列をページごとに Pillow 画像に変換"""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((i + 1, img))  # ページ番号と画像
    return images

# ----- メイン処理 -----
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
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".heic":
                heif_file = pillow_heif.read_heif(BytesIO(resp.content))
                img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
                images = [(None, img)]
            elif ext == ".pdf":
                images = pdf_to_images(resp.content)
            else:
                img = Image.open(BytesIO(resp.content))
                images = [(None, img)]

            for page_num, img in images:
                # 傾き補正
                img = deskew_image(img)
                # トリミング
                trimmed = trim_paper_hsv(img)
                # 縦横調整
                if trimmed.width > trimmed.height:
                    trimmed = trimmed.rotate(90, expand=True)
                # 半紙サイズにリサイズ
                hanshi_size = (2890, 3953)
                trimmed = trimmed.resize(hanshi_size, Image.LANCZOS)
                # 保存名
                save_name = f"corrected_{os.path.splitext(filename)[0]}"
                if page_num:
                    save_name += f"_page{page_num}"
                save_name += ".png"
                # 保存
                trimmed.save(save_name, format="PNG")
                print(f"Saved and uploaded {save_name}")
                # Dropbox アップロード
                dropbox_path = os.path.join(DROPBOX_FOLDER, os.path.basename(save_name))
                with open(save_name, "rb") as f:
                    dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

            processed_links.add(direct_link)
            worksheet.delete_rows(row_index)
            print(f"Deleted processed row {row_index} from sheet.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_latest_links()