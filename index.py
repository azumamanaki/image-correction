# -*- coding: utf-8 -*-
import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF

# ===== 設定（必須環境変数） =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

DROPBOX_SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DROPBOX_PRINT_FOLDER = DROPBOX_SRC_FOLDER + "/添削用印刷未"
DROPBOX_PROCESSED_FOLDER = DROPBOX_SRC_FOLDER + "/補正済元画像"
DROPBOX_FAILED_FOLDER = DROPBOX_SRC_FOLDER + "/補正失敗"
DROPBOX_DEBUG_FOLDER = DROPBOX_SRC_FOLDER + "/_debug"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== デバッグ =====
DEBUG_SAVE = True  # Trueにすると debug画像を Dropbox/_debug に保存

# ===== Dropbox 初期化 =====
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

# debugフォルダ作成（なければ）
if DEBUG_SAVE:
    try:
        dbx.files_get_metadata(DROPBOX_DEBUG_FOLDER)
    except Exception:
        try:
            dbx.files_create_folder_v2(DROPBOX_DEBUG_FOLDER)
        except Exception:
            pass

def save_debug_to_dropbox(pil_img, name):
    """PIL Image を DROPBOX_DEBUG_FOLDER/name として保存（overwrite）"""
    if not DEBUG_SAVE:
        return
    try:
        bio = io.BytesIO()
        pil_img.save(bio, format="PNG")
        bio.seek(0)
        path = f"{DROPBOX_DEBUG_FOLDER}/{name}"
        dbx.files_upload(bio.read(), path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"  [DEBUG_SAVE] {path} に保存")
    except Exception as e:
        print(f"  [DEBUG_SAVE_ERROR] {e}")

# ===== deskew （従来の minAreaRect ベース、転置は行わない） =====
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
    rotated = cv2.warpAffine(img_cv, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # 転置回転は行わず、傾き補正のみ
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

# ===== あなたが示した高精度トリミング（HSVベース）を採用 =====
def trim_paper_hsv(image, sat_thresh=30, val_thresh=200, debug=False, file_name="img", page_idx=0):
    """
    HSV空間で白に近い領域をトリミングする関数（あなた提示の実装を拡張）。
    - image: PIL Image
    - sat_thresh: 彩度の閾値（S <= sat_thresh を白っぽいと判定）
    - val_thresh: 明度の閾値（V >= val_thresh を白っぽいと判定）
    - debug: Trueならマスク・切り取り位置を Dropbox/_debug に保存
    """
    img = image.convert("RGB")
    hsv_img = img.convert("HSV")
    np_hsv = np.array(hsv_img)

    s = np_hsv[:, :, 1]
    v = np_hsv[:, :, 2]

    white_mask = (s <= sat_thresh) & (v >= val_thresh)
    coords = np.argwhere(white_mask)

    # デバッグ用：マスク画像を保存
    if debug:
        mask_vis = (white_mask.astype(np.uint8) * 255)
        # mask を RGB 化して保存
        mask_pil = Image.fromarray(mask_vis).convert("RGB")
        save_debug_to_dropbox(mask_pil, f"debug_mask_{file_name}_{page_idx}.png")

    if coords.size == 0:
        print("  [trim] 白領域が見つからず → トリミングスキップ")
        return image

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # 境界クリップ（念のため）
    w, h = img.size
    x0 = max(0, min(x0, w - 1))
    x1 = max(1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(1, min(y1, h))

    cropped = img.crop((x0, y0, x1, y1))

    # デバッグ：切り出し位置を元画像に赤枠で描画して保存
    if debug:
        vis = np.array(img).copy()
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 4)  # 青枠（BGR）
        vis_pil = Image.fromarray(vis)
        save_debug_to_dropbox(vis_pil, f"debug_crop_{file_name}_{page_idx}.png")

    return cropped

# ===== A4 に余白配置（歪ませずフィット） =====
def fit_to_a4_padded(pil_img, a4=(2480, 3508)):
    w, h = pil_img.size
    scale = min(a4[0] / w, a4[1] / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = pil_img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", a4, (255, 255, 255))
    off = ((a4[0] - new_size[0]) // 2, (a4[1] - new_size[1]) // 2)
    canvas.paste(resized, off)
    return canvas

# ===== PDF→画像 =====
def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        print(f"  [pdf] {i+1} ページ目を画像化 (size={pix.width}x{pix.height})")
    return images

# ===== Dropbox 処理（メイン） =====
def process_file(file_metadata, sat_thresh=30, val_thresh=200):
    file_path = file_metadata.path_lower
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"\n=== 処理開始: {file_name} (拡張子={ext}) ===")

    if ext not in SUPPORTED_EXTS:
        fail_dest = f"{DROPBOX_FAILED_FOLDER}/{file_name}"
        try:
            dbx.files_move_v2(file_path, fail_dest, autorename=True)
            print(f"[SKIP] 非対応拡張子 → {fail_dest} に移動")
        except Exception as e:
            print(f"[ERROR] 非対応ファイルの移動失敗 {file_path}: {e}")
        return

    try:
        print("  [DL] ダウンロード中...")
        _, res = dbx.files_download(file_path)
        data = res.content
        print(f"  [DL] ダウンロード完了 ({len(data)} bytes)")

        # 画像読み込み
        if ext == ".pdf":
            print("  [LOAD] PDF → 画像変換開始")
            images = pdf_to_images(data)
        else:
            print("  [LOAD] 画像読み込み中...")
            img = Image.open(io.BytesIO(data))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            print(f"  [LOAD] 読み込み完了 size={img.width}x{img.height}")
            images = [img]

        processed_images = []
        for idx, img in enumerate(images):
            print(f"  [PROC] {idx+1}枚目 補正開始")

            # 1) deskew（傾き補正）
            img = deskew_image(img)

            # 2) trim（HSVベースの白領域トリミング） --- あなたの関数を採用
            img_trimmed = trim_paper_hsv(img, sat_thresh=sat_thresh, val_thresh=val_thresh,
                                         debug=DEBUG_SAVE, file_name=file_name, page_idx=idx)

            # 3) 補正後の向き確認（縦長化）
            w, h = img_trimmed.size
            if w > h:
                img_trimmed = img_trimmed.rotate(90, expand=True)
                print("  [ORIENT] 横長検出 → 90°回転")
            else:
                print("  [ORIENT] 縦長検出 → 回転なし")

            # 4) A4 に余白パディングで配置
            a4_img = fit_to_a4_padded(img_trimmed, a4=(2480, 3508))
            processed_images.append(a4_img)
            print(f"  [PROC] 処理完了 サイズ({a4_img.width}x{a4_img.height})")

        # PDF にまとめてアップロード
        pdf_bytes = io.BytesIO()
        processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
        pdf_bytes.seek(0)
        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"[UPLOAD] {dest_pdf_path} にアップロード完了")

        # 元ファイルを移動
        processed_dest = f"{DROPBOX_PROCESSED_FOLDER}/{file_name}"
        dbx.files_move_v2(file_path, processed_dest, autorename=True)
        print(f"[MOVE] 元ファイルを {processed_dest} に移動完了")

        print(f"✅ 成功: {file_name}")

    except Exception as e:
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
            process_file(entry, sat_thresh=30, val_thresh=200)

if __name__ == "__main__":
    main()