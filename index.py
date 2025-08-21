import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF

# ===== 設定 =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

DROPBOX_SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DROPBOX_PRINT_FOLDER = DROPBOX_SRC_FOLDER + "/添削用印刷未"
DROPBOX_PROCESSED_FOLDER = DROPBOX_SRC_FOLDER + "/補正済元画像"
DROPBOX_FAILED_FOLDER = DROPBOX_SRC_FOLDER + "/補正失敗"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== 初期化 =====
def get_access_token():
    data = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_CLIENT_ID,
        "client_secret": DROPBOX_CLIENT_SECRET
    }
    resp = requests.post("https://api.dropbox.com/oauth2/token", data=data)
    resp.raise_for_status()
    return resp.json()["access_token"]

dbx = dropbox.Dropbox(get_access_token())

# ===== 画像補正 =====
def deskew_image(pil_img):
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        print("  [deskew] 有効な座標なし → 補正スキップ")
        return pil_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    print(f"  [deskew] 回転角度: {angle:.2f}°")
    (h, w) = img_cv.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

def trim_paper_hsv(image, sat_thresh=30, val_thresh=200):
    img = image.convert("RGB")
    hsv_img = img.convert("HSV")
    np_hsv = np.array(hsv_img)
    s = np_hsv[:, :, 1]
    v = np_hsv[:, :, 2]
    white_mask = (s <= sat_thresh) & (v >= val_thresh)
    coords = np.argwhere(white_mask)
    if coords.size == 0:
        print("  [trim] 白背景領域なし → トリミングスキップ")
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    print(f"  [trim] トリミング範囲: x={x0}:{x1}, y={y0}:{y1}")
    return img.crop((x0, y0, x1, y1))

def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  [pdf] {i+1} ページ目を画像化 (size={pix.width}x{pix.height})")
    return images

# ===== Dropboxファイル処理 =====
def process_file(file_metadata):
    file_path = file_metadata.path_lower
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"\n=== 処理開始: {file_name} (拡張子={ext}) ===")

    # 対象外の拡張子は補正失敗に移動
    if ext not in SUPPORTED_EXTS:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"[SKIP] 非対応拡張子 → {fail_dest} に移動")
        except Exception as e:
            print(f"[ERROR] 非対応ファイルの移動失敗 {file_path}: {e}")
        return

    try:
        # Dropboxからダウンロード
        print("  [DL] ダウンロード中...")
        _, res = dbx.files_download(file_path)
        data = res.content
        print(f"  [DL] ダウンロード完了 ({len(data)} bytes)")

        # 画像読み込み
        if ext == ".pdf":
            print("  [LOAD] PDF → 画像変換開始")
            images = pdf_to_images(data)
        else:  # png/jpeg/jpg
            print("  [LOAD] 画像読み込み中...")
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            print(f"  [LOAD] 読み込み完了 size={img.width}x{img.height}")
            images = [img]

        # 補正処理
        processed_images = []
        for idx, img in enumerate(images):
            print(f"  [PROC] {idx+1}枚目 補正開始")
            img = deskew_image(img)
            img = trim_paper_hsv(img)
            if img.width > img.height:
                img = img.rotate(90, expand=True)
                print("  [PROC] 横長画像 → 90°回転")
            img = img.resize((2480, 3508), Image.LANCZOS)
            print(f"  [PROC] A4サイズにリサイズ ({img.width}x{img.height})")
            processed_images.append(img)

        # PDF に変換
        pdf_bytes = io.BytesIO()
        try:
            print("  [PDF] PDF生成中...")
            processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
            pdf_bytes.seek(0)
            print("  [PDF] PDF生成完了")
        except Exception as e:
            raise RuntimeError(f"PDF creation failed: {e}")

        # 添削用印刷未にアップロード
        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        try:
            print(f"  [UPLOAD] {dest_pdf_path} にアップロード中...")
            dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))
            print("  [UPLOAD] アップロード完了")
        except Exception as e:
            raise RuntimeError(f"PDF upload failed: {e}")

        # アップロード成功したら元ファイルを補正済元画像に移動
        processed_dest = f"{DROPBOX_PROCESSED_FOLDER}/{file_name}"
        print(f"  [MOVE] 元ファイルを {processed_dest} に移動中...")
        dbx.files_move_v2(file_path, processed_dest, autorename=True)
        print("  [MOVE] 移動完了")

        print(f"✅ 成功: {file_name}")

    except Exception as e:
        # 失敗時は補正失敗フォルダに移動
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"❌ 失敗: {file_name} → {fail_dest}")
        except Exception as move_err:
            print(f"[ERROR] 失敗ファイルの移動失敗 {file_path}: {move_err}")
        print(f"[ERROR] {file_name} 処理中に例外発生: {e}")

# ===== メイン =====
def main():
    print("=== Dropbox フォルダ走査開始 ===")
    res = dbx.files_list_folder(DROPBOX_SRC_FOLDER)
    for entry in res.entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            process_file(entry)

if __name__ == "__main__":
    main()