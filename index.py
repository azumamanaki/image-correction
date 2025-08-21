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
DROPBOX_DEBUG_FOLDER = DROPBOX_SRC_FOLDER + "/_debug"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== デバッグ保存フラグ =====
DEBUG_SAVE = True  # Trueにすると中間画像をDropbox/_debugへ保存

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

# ===== ユーティリティ =====
def _put_debug(img_bgr, name_hint):
    if not DEBUG_SAVE:
        return
    try:
        os.makedirs("/mnt/data/_tmp", exist_ok=True)
        path = f"/mnt/data/_tmp/{name_hint}.png"
        cv2.imwrite(path, img_bgr)
        with open(path, "rb") as f:
            dbx.files_upload(f.read(), f"{DROPBOX_DEBUG_FOLDER}/{name_hint}.png", mode=dropbox.files.WriteMode("overwrite"))
    except Exception as e:
        print(f"[DEBUG_SAVE_ERROR] {e}")


def _order_quad(pts):
    # 4x2 -> 4x2 (tl, tr, br, bl)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect


def _auto_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def detect_hanshi_quad(img_bgr):
    """半紙の四隅(台形)を検出して返す。失敗時はNone。
    - 白・低彩度の領域 + エッジを統合
    - ラベリングで最大候補を抽出
    - 4点近似 or minAreaRectで矩形候補
    候補は矩形度・面積・アスペクト比でスコアリング。
    """
    H, W = img_bgr.shape[:2]
    max_dim = max(H, W)
    scale = 1024 / max_dim if max_dim > 1024 else 1.0
    small = cv2.resize(img_bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]

    # ---- 色空間で白紙マスク（LAB） ----
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L, A, Bc = cv2.split(lab)
    # OpenCVのLABはa,bが128中心。低彩度＝a,bが128付近
    C = cv2.absdiff(A, 128) + cv2.absdiff(Bc, 128)
    # しきい値（必要に応じて微調整）
    L_thr = 200  # 明るさ
    C_thr = 20   # 彩度（小さいほどグレー/白）
    mask_white = ((L > L_thr) & (C < C_thr)).astype(np.uint8) * 255

    k = max(3, int(0.01 * max(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=1)

    _put_debug(mask_white, "01_mask_white")

    # ---- エッジ検出 ----
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = _auto_canny(blur)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    _put_debug(edges, "02_edges")

    # ---- マスク統合 ----
    mask = cv2.bitwise_or(mask_white, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    _put_debug(mask, "03_mask_merged")

    # ---- ラベリングで最大候補 ----
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        print("  [hanshi] ラベリング失敗")
        return None
    # 最初のラベル0は背景
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + np.argmax(areas)
    cand = np.zeros_like(mask)
    cand[labels == best_idx] = 255

    _put_debug(cand, "04_largest_component")

    # ---- 輪郭 → 4点近似 or 最小外接矩形 ----
    contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  [hanshi] 輪郭なし")
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)

    # 元サイズへスケールバック
    quad /= scale
    quad = _order_quad(quad)

    dbg = img_bgr.copy()
    cv2.polylines(dbg, [quad.astype(np.int32)], True, (0, 255, 0), 3)
    _put_debug(dbg, "05_quad_on_image")

    # 妥当性チェック（面積・比率）
    area = cv2.contourArea(quad.astype(np.float32))
    frac = area / (W * H)
    print(f"  [hanshi] quad面積比: {frac:.3f}")
    if frac < 0.1:  # 小さすぎる候補は無効
        print("  [hanshi] 面積が小さすぎ → 無視")
        return None

    # アスペクト（縦長前提で評価）
    wA = np.linalg.norm(quad[1] - quad[0])
    wB = np.linalg.norm(quad[2] - quad[3])
    hA = np.linalg.norm(quad[3] - quad[0])
    hB = np.linalg.norm(quad[2] - quad[1])
    est_w = (wA + wB) / 2
    est_h = (hA + hB) / 2
    ar = max(est_w, est_h) / max(1.0, min(est_w, est_h))
    print(f"  [hanshi] 推定アスペクト比: {ar:.3f}")

    return quad


def warp_by_quad_to_portrait(img_bgr, quad):
    # quadを使って透視変換し、縦長に整える
    quad = _order_quad(quad.astype(np.float32))
    # 出力サイズを四辺の平均長から決める
    wA = np.linalg.norm(quad[1] - quad[0])
    wB = np.linalg.norm(quad[2] - quad[3])
    hA = np.linalg.norm(quad[3] - quad[0])
    hB = np.linalg.norm(quad[2] - quad[1])
    Wd = int(max(wA, wB))
    Hd = int(max(hA, hB))

    # 一旦その比で平面化
    dst = np.array([[0, 0], [Wd - 1, 0], [Wd - 1, Hd - 1], [0, Hd - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    flat = cv2.warpPerspective(img_bgr, M, (Wd, Hd))

    # 縦長へ（高さ>=幅）
    if flat.shape[1] > flat.shape[0]:
        flat = cv2.rotate(flat, cv2.ROTATE_90_CLOCKWISE)
    _put_debug(flat, "06_flattened")
    return flat


def pad_to_a4(img_bgr, a4=(2480, 3508), margin_frac=0.04):
    Aw, Ah = a4
    h, w = img_bgr.shape[:2]
    # 余白を少し確保してフィット
    mw = int(Aw * margin_frac)
    mh = int(Ah * margin_frac)
    scale = min((Aw - 2 * mw) / w, (Ah - 2 * mh) / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((Ah, Aw, 3), 255, dtype=np.uint8)
    x = (Aw - nw) // 2
    y = (Ah - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    _put_debug(canvas, "07_a4_canvas")
    return canvas


# ===== 抽出（高精度版） =====
def extract_hanshi(image):
    """半紙を検出→透視変換（縦長化）→A4へのレターボックス配置。
    失敗時は元画像を縦長方向へ回してA4にパディング。
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    quad = detect_hanshi_quad(img_bgr)

    if quad is not None:
        try:
            flat = warp_by_quad_to_portrait(img_bgr, quad)
            out = pad_to_a4(flat, a4=(2480, 3508))
            print("  [hanshi] 透視変換→A4配置 成功")
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"  [hanshi] 透視変換失敗: {e}")

    # フォールバック：従来のtrim + A4
    print("  [hanshi] フォールバック経路へ")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        fallback = img_bgr
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        fallback = img_bgr[y0:y1, x0:x1]
    # 縦長へ
    if fallback.shape[1] > fallback.shape[0]:
        fallback = cv2.rotate(fallback, cv2.ROTATE_90_CLOCKWISE)
    out = pad_to_a4(fallback, a4=(2480, 3508))
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


# ===== 画像補正（deskewは透視変換で置換） =====

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

            # 初期縦横判定（横長なら縦に回転）
            w0, h0 = img.size
            if w0 > h0:
                img = img.rotate(90, expand=True)
                print("  [INIT] 横長→90°回転")
            else:
                print("  [INIT] 縦長→回転なし")

            # 半紙抽出→A4配置
            img = extract_hanshi(img)

            # 最終安全回転（念のため縦長化）
            w, h = img.size
            if w > h:
                img = img.rotate(90, expand=True)
                print("  [SAFE] 最終縦長化 90°回転")

            processed_images.append(img)

        # PDF へ
        pdf_bytes = io.BytesIO()
        processed_images[0].save(pdf_bytes, format="PDF", save_all=True, append_images=processed_images[1:])
        pdf_bytes.seek(0)

        dest_pdf_path = f"{DROPBOX_PRINT_FOLDER}/{os.path.splitext(file_name)[0]}.pdf"
        dbx.files_upload(pdf_bytes.read(), dest_pdf_path, mode=dropbox.files.WriteMode("overwrite"))
        print(f"[UPLOAD] {dest_pdf_path} にアップロード完了")

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
            process_file(entry)

if __name__ == "__main__":
    main()
