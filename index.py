import os
import io
import dropbox
import requests
import numpy as np
from PIL import Image, ImageOps
import cv2
import fitz  # PyMuPDF

"""
目的：
- 写真に写っている『半紙（ほぼ白い紙）』領域のみを頑健に抽出し、
  斜め撮影でも透視変換で正面化 → 最後に縦長（必要なら90°回転）→ A4に収めてPDF化。

方針：
1) 白色＆低彩度のマスクで半紙候補を抽出（LABのLが高くHSVのSが低いピクセル）。
2) マスクをモルフォロジーで整形し、最大コンポーネントから最小外接矩形（minAreaRect）。
3) その矩形の4点で透視変換（紙の内側の墨の線は無視できる）。
4) 失敗時はCanny→approxPolyDPの4角形近似でフォールバック。
5) 仕上げに縦長へ回転（横長なら90°）→ A4へ余白パディングして非歪みで収める。

調整パラメータは WHITE_L_THRESH / WHITE_S_THRESH / AREA_MIN_FRAC など。
"""

# ===== 設定 =====
DROPBOX_CLIENT_ID = os.environ["DROPBOX_CLIENT_ID"]
DROPBOX_CLIENT_SECRET = os.environ["DROPBOX_CLIENT_SECRET"]
DROPBOX_REFRESH_TOKEN = os.environ["DROPBOX_REFRESH_TOKEN"]

DROPBOX_SRC_FOLDER = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DROPBOX_PRINT_FOLDER = DROPBOX_SRC_FOLDER + "/添削用印刷未"
DROPBOX_PROCESSED_FOLDER = DROPBOX_SRC_FOLDER + "/補正済元画像"
DROPBOX_FAILED_FOLDER = DROPBOX_SRC_FOLDER + "/補正失敗"

SUPPORTED_EXTS = [".png", ".jpeg", ".jpg", ".pdf"]

# ===== 抽出パラメータ（必要に応じて調整） =====
WHITE_L_THRESH = 200   # LABの明度Lがこの値より大きい→白
WHITE_S_THRESH = 60    # HSVの彩度Sがこの値より小さい→低彩度（白系）
WHITE_L_THRESH_RELAX = 180
WHITE_S_THRESH_RELAX = 100
AREA_MIN_FRAC = 0.05   # 画像全体に対する候補面積の下限（5%未満は無視）
RECT_AR_MIN = 1.1      # 最小外接矩形の縦横比（長辺/短辺）の下限
RECT_AR_MAX = 1.9      # 同 上限（半紙やA4に近い範囲をゆるめに）

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
def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2) float32
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def _perspective_by_box(img_bgr: np.ndarray, box_pts: np.ndarray) -> np.ndarray:
    rect = _order_points(box_pts.astype("float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    maxW = max(10, maxW)
    maxH = max(10, maxH)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    return warp

def _fit_to_a4_padded(pil_img: Image.Image, a4=(2480, 3508)) -> Image.Image:
    # 歪みはつけず等倍スケール＋白余白でA4に中央配置
    w, h = pil_img.size
    scale = min(a4[0] / w, a4[1] / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized = pil_img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("RGB", a4, (255, 255, 255))
    off = ((a4[0] - new_size[0]) // 2, (a4[1] - new_size[1]) // 2)
    canvas.paste(resized, off)
    return canvas

# ===== 半紙検出（主経路） =====
def extract_hanshi(image: Image.Image) -> Image.Image:
    """白＆低彩度マスク→最大コンポーネント→minAreaRect→4点透視変換。
    失敗時はエッジ4角形近似でフォールバック。戻りはPIL Image。
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 1) 白＆低彩度マスク
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]

    mask = ((L > WHITE_L_THRESH) & (S < WHITE_S_THRESH)).astype(np.uint8) * 255
    frac = mask.mean() / 255.0
    if frac < 0.05:  # 5%未満なら緩める
        mask = ((L > WHITE_L_THRESH_RELAX) & (S < WHITE_S_THRESH_RELAX)).astype(np.uint8) * 255
        print(f"  [mask] 白領域が少ないため閾値を緩和 (frac={frac:.3f})")

    k = max(3, (max(H, W) // 100) | 1)  # 画像サイズ依存のカーネル
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(H * W)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_MIN_FRAC * img_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect
        if rw <= 1 or rh <= 1:
            continue
        rect_area = rw * rh
        rectangularity = float(area) / float(rect_area)
        ar = max(rw, rh) / max(1.0, min(rw, rh))
        # 長方形らしさを優先しつつ面積を加点
        score = rectangularity * (area / img_area)
        # 縦横比が極端におかしい候補には減点
        if not (RECT_AR_MIN <= ar <= RECT_AR_MAX):
            score *= 0.5
        if score > best_score:
            best = rect
            best_score = score

    if best is not None:
        box = cv2.boxPoints(best)  # 4点
        warp = _perspective_by_box(img_bgr, box)
        print("  [hanshi] 白マスク→minAreaRectで抽出成功")
    else:
        # 2) フォールバック：Canny→最大輪郭→四角近似
        print("  [hanshi] 白マスク抽出失敗 → エッジフォールバック")
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
        warp = None
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            if area < AREA_MIN_FRAC * img_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                box = approx.reshape(4, 2).astype("float32")
                warp = _perspective_by_box(img_bgr, box)
                print("  [hanshi] Canny→四角近似で抽出成功")
                break
        if warp is None:
            print("  [hanshi] 抽出失敗 → 元画像を使用")
            warp = img_bgr

    # portrait化（横長なら90°回転）
    if warp.shape[1] > warp.shape[0]:
        warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("  [rotate] 横長→縦長に90°回転")

    pil = Image.fromarray(cv2.cvtColor(warp, cv2.COLOR_BGR2RGB))
    return pil

# ===== PDF→画像 =====
def pdf_to_images(pdf_bytes):
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        pix = page.get_pixmap()  # 必要なら高解像に: matrix=fitz.Matrix(2,2)
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

            # 半紙抽出（失敗時は自動フォールバック）
            hanshi = extract_hanshi(img)
            w, h = hanshi.size
            print(f"  [result] 抽出後サイズ: {w}x{h}")

            # A4へ余白パディングで非歪み配置
            a4_img = _fit_to_a4_padded(hanshi, a4=(2480, 3508))
            print("  [a4] A4キャンバスへ中央配置")

            processed_images.append(a4_img)

        # PDF にまとめる
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