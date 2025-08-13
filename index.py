import gspread
import requests
from io import BytesIO
import numpy as np
import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

# Google Sheets認証
gc = gspread.service_account(filename='shodo-test-f1825a5fea87.json')
sh = gc.open_by_key("1x4Cxp4YA-8uFG2PHlcDp4WBzHpWUxWOGao7bicejH8Q")
worksheet = sh.sheet1  # 1枚目のシートを利用

# すでに処理したリンクを記録する
processed_links = set()

def trim_paper_hsv(image, sat_thresh, val_thresh):
    """HSV空間で白に近い領域をトリミング"""
    img = image.convert("RGB")
    np_img = np.array(img)

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

def process_latest_link():
    all_rows = worksheet.get_all_values()

    for row in all_rows[1:]:  # 1行目はヘッダ想定
        filename, direct_link, date = row

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

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    process_latest_link()