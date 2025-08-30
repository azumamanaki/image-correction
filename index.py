# -*- coding: utf-8 -*-
"""
index_embedded.py
- 単一ファイルで完結：a.py のトリミング処理をソースごと埋め込み
- Dropbox収集 → トリミング（a.py埋め込み） → 縦長回転 → A4化 → 整理アップロード
"""

import os, io, sys, json, tempfile
from pathlib import Path
from typing import Tuple
import types
import requests
import dropbox
from dropbox.files import WriteMode
from PIL import Image
import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ====== 埋め込み a.py ソース ======
A_PY_SOURCE = r'# -*- coding: utf-8 -*-\n#ほぼ完璧\n\n"""\nおうち書道：黒縁/色背景除去 + 内側細枠優先トリミング（コラム誤検出対策版）\n+ 仕上げ：四辺の“白さ”で控えめに追いトリム（paper_mask / frame 両ルート対応）\n\n- 入力: fram_image/*.jpg|jpeg|png\n- 出力: complete/\n- デバッグ: debug/ に各段階の画像と JSON\n"""\n\nfrom pathlib import Path\nimport cv2\nimport numpy as np\nimport json\nfrom typing import Tuple, Dict, Any, List\n\n# ===== フォルダ =====\nINPUT_DIR  = Path("fram_image")\nOUTPUT_DIR = Path("complete")\nDEBUG_DIR  = Path("debug")\n\n# ===== 外周の黒帯除去 =====\nDARK_RATIO_EDGE = 0.40\nDARK_THRESH     = 90\nSAFE_MARGIN_PX  = 2\n\n# ===== 紙マスク =====\nHSV_V_MIN  = 150\nHSV_S_MAX  = 60\nADAPT_BLOCK = 51\nADAPT_C     = 10\nK_CLOSE     = (9, 9)\nK_OPEN      = (5, 5)\nMIN_AREA_RATIO = 0.20\n\n# ===== 内側の黒枠（外枠）検出：コラム誤検出を抑制 =====\n# 反転適応二値化（黒線→白）\nADAPT_BLOCK_INV = 31\nADAPT_C_INV     = 10\nFRAME_CONNECT_K = (3, 3)    # 黒線の連結\n\n# 候補矩形の面積・幅・アスペクトのガード\nFRAME_MIN_ARATIO   = 0.15   # 面積比の下限（全体の15%以上）\nFRAME_MAX_ARATIO   = 0.95   # 上限\nFRAME_MIN_W_FRAC   = 0.35   # 画像幅に対する最小幅 35%（細コラム排除）\nFRAME_PAPER_AR_MIN = 1.05   # 縦/横（紙っぽい範囲）\nFRAME_PAPER_AR_MAX = 1.80\n\n# “細いコラム”の定義（縦長過ぎる候補）\nCOLUMN_AR_MIN = 3.0         # 縦/横が3以上ならコラム候補\nCOLUMN_MIN_H_FRAC = 0.6     # 画像高さの60%以上の高さがある細長いもの\n\n# スコアリング\nTARGET_ASPECT  = 1.38       # 半紙/硬筆の事前分布\nASPECT_TOL     = 0.60       # 許容（±0.60）\nFRAME_CENTER_BIAS = 0.0005  # 中心に近いほど微優先\nFRAME_INSIDE_MEAN_MIN = 160 # 内側の明るさ（紙想定）\n\n# 外枠が見つかったら内側に少し寄せる（黒線を避ける）\nFRAME_INNER_MARGIN_RATIO = 0.01\n\n\n# ===== Utility =====\ndef ensure_dirs():\n    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n    DEBUG_DIR.mkdir(parents=True, exist_ok=True)\n\ndef imread(p: Path) -> np.ndarray:\n    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)\n    if img is None: raise RuntimeError(f"Failed to read: {p}")\n    return img\n\ndef imwrite(p: Path, img: np.ndarray):\n    p.parent.mkdir(parents=True, exist_ok=True)\n    ext = p.suffix.lower() if p.suffix else ".jpg"\n    params = []\n    if ext in [".jpg", ".jpeg"]:\n        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]\n    elif ext == ".png":\n        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]\n    data = cv2.imencode(ext, img, params)[1]\n    p.write_bytes(bytearray(data))\n\ndef auto_trim_black_edges(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:\n    h, w = img.shape[:2]\n    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    def r(y): return (g[y, :] < DARK_THRESH).mean()\n    def c(x): return (g[:, x] < DARK_THRESH).mean()\n    top, bot, lef, rig = 0, h-1, 0, w-1\n    while top < bot and r(top) > DARK_RATIO_EDGE: top += 1\n    while bot > top and r(bot) > DARK_RATIO_EDGE: bot -= 1\n    while lef < rig and c(lef) > DARK_RATIO_EDGE: lef += 1\n    while rig > lef and c(rig) > DARK_RATIO_EDGE: rig -= 1\n    top  = min(max(0, top + SAFE_MARGIN_PX), h-2)\n    lef  = min(max(0, lef + SAFE_MARGIN_PX), w-2)\n    bot  = max(min(h-1, bot - SAFE_MARGIN_PX), top+1)\n    rig  = max(min(w-1, rig - SAFE_MARGIN_PX), lef+1)\n    return img[top:bot+1, lef:rig+1], dict(step="edge_trim", top=top, bottom=bot, left=lef, right=rig)\n\ndef build_paper_mask(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:\n    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)\n    bright_low_sat = cv2.inRange(hsv, (0,0,HSV_V_MIN), (179,HSV_S_MAX,255))\n    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\n                                  cv2.THRESH_BINARY, ADAPT_BLOCK, ADAPT_C)\n    mask = cv2.bitwise_or(bright_low_sat, adapt)\n    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,\n                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_CLOSE), 1)\n    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,\n                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, K_OPEN), 1)\n    return mask, dict(step="paper_mask")\n\ndef largest_cc(mask: np.ndarray) -> Tuple[np.ndarray, float]:\n    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)\n    if num <= 1: return mask, 0.0\n    areas = stats[1:, cv2.CC_STAT_AREA]; idx = 1 + int(np.argmax(areas))\n    out = np.zeros_like(mask); out[lab == idx] = 255\n    return out, float(areas.max() / mask.size)\n\ndef order_quad(pts: np.ndarray) -> np.ndarray:\n    pts = pts.astype(np.float32)\n    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()\n    tl = pts[np.argmin(s)]; br = pts[np.argmax(s)]\n    tr = pts[np.argmin(d)]; bl = pts[np.argmax(d)]\n    return np.array([tl, tr, br, bl], dtype=np.float32)\n\ndef warp_by_quad(img: np.ndarray, quad: np.ndarray, inner_ratio: float) -> np.ndarray:\n    q = order_quad(quad)\n    w = int(np.linalg.norm(q[1]-q[0])); h = int(np.linalg.norm(q[3]-q[0]))\n    w = max(w,10); h = max(h,10)\n    M = cv2.getPerspectiveTransform(q, np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], np.float32))\n    warped = cv2.warpPerspective(img, M, (w, h))\n    m = int(round(min(w,h)*inner_ratio))\n    if m>0 and w>2*m and h>2*m: warped = warped[m:h-m, m:w-m]\n    if warped.shape[0] > 2*SAFE_MARGIN_PX and warped.shape[1] > 2*SAFE_MARGIN_PX:\n        warped = warped[SAFE_MARGIN_PX:-SAFE_MARGIN_PX, SAFE_MARGIN_PX:-SAFE_MARGIN_PX]\n    return warped\n\ndef visualize_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:\n    vis = img.copy()\n    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n    cv2.drawContours(vis, cnts, -1, (0,0,255), 2)\n    return vis\n\n\n# ===== 仕上げ：四辺の“白さ”で控えめに追いトリム =====\ndef measure_edge_white_ratio(img: np.ndarray, band_frac: float = 0.02, thr: int = 225) -> Dict[str, float]:\n    """画像の四辺バンドの『白さ(=明るさ)比率』を返す"""\n    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    h, w = g.shape\n    b = max(2, int(min(h, w) * band_frac))\n\n    def ratio(region):\n        return float((region >= thr).mean())  # 0..1\n\n    r = {\n        "top":    ratio(g[0:b, :]),\n        "bottom": ratio(g[h-b:h, :]),\n        "left":   ratio(g[:, 0:b]),\n        "right":  ratio(g[:, w-b:w]),\n    }\n    return r\n\ndef suggest_insets_from_ratios(r: Dict[str, float], w: int, h: int) -> Dict[str, int]:\n    """\n    4辺が低い場合でも“特に低い辺”は控えめに寄せる。\n    通常は段階テーブルに従い寄せる。\n    """\n    vals = [r.get("top",1), r.get("bottom",1), r.get("left",1), r.get("right",1)]\n    names = ["top","bottom","left","right"]\n\n    base = max(4, int(min(w, h) * 0.015))  # 控えめ基準\n    tiers = [(0.98,0.0),(0.95,0.5),(0.90,1.0),(0.80,1.6),(0.70,2.2),(0.60,2.8),(0.00,3.4)]\n    def inset_for(val: float) -> int:\n        for th,k in tiers:\n            if val >= th: return int(round(base*k))\n        return int(round(base*tiers[-1][1]))\n\n    ins = {k:0 for k in names}\n    all_low = all(v <= 0.50 for v in vals)\n    if all_low:\n        anchor = float(np.median(vals))\n        def is_much_lower(v): return (v <= min(0.45, anchor - 0.07))\n        for k,v in zip(names, vals):\n            ins[k] = int(round(inset_for(v) * (0.8 if is_much_lower(v) else 0.0)))\n    else:\n        ins["top"]    = inset_for(r["top"])\n        ins["bottom"] = inset_for(r["bottom"])\n        ins["left"]   = inset_for(r["left"])\n        ins["right"]  = inset_for(r["right"])\n\n    # 上限キャップ（過切り防止）\n    v_cap = max(6, min(18, int(h * 0.06)))  # 上下\n    h_cap = max(6, min(14, int(w * 0.05)))  # 左右\n    ins["top"]    = min(ins["top"], v_cap)\n    ins["bottom"] = min(ins["bottom"], v_cap)\n    ins["left"]   = min(ins["left"], h_cap)\n    ins["right"]  = min(ins["right"], h_cap)\n    return {k:int(v) for k,v in ins.items()}\n\n\n# ===== 内側枠検出（コラム耐性） =====\ndef detect_frame_or_compose(img: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:\n    """\n    - 黒線強調（反転適応二値化）→連結→輪郭抽出\n    - \'紙っぽい\'矩形のみを強く優先（AR/幅/面積）\n    - 縦長コラムは除外。ただしコラムが複数ある場合はまとめて外接矩形へ合成して再判定\n    """\n    h, w = img.shape[:2]\n    img_area = float(h*w)\n    cx, cy = w/2.0, h/2.0\n\n    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n    g = cv2.GaussianBlur(g, (5,5), 0)\n    inv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\n                                cv2.THRESH_BINARY_INV, ADAPT_BLOCK_INV, ADAPT_C_INV)\n    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE,\n                           cv2.getStructuringElement(cv2.MORPH_RECT, FRAME_CONNECT_K), 2)\n\n    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n\n    paperish: List[Dict[str,Any]] = []\n    columns:  List[Dict[str,Any]] = []\n\n    for c in cnts:\n        peri = cv2.arcLength(c, True)\n        if peri < 0.3*(h+w): continue\n        approx = cv2.approxPolyDP(c, 0.02*peri, True)\n        if len(approx)!=4 or not cv2.isContourConvex(approx): continue\n\n        quad = order_quad(approx.reshape(-1,2))\n        ww = np.linalg.norm(quad[1]-quad[0]); hh = np.linalg.norm(quad[3]-quad[0])\n        if ww<=0 or hh<=0: continue\n        aratio = (ww*hh)/img_area\n        ar = hh/ww\n        w_frac = ww / w\n        if aratio<0.02:  # ごく小さいノイズ除去\n            continue\n\n        # 内側の明るさチェック\n        mask = np.zeros((h,w), np.uint8)\n        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)\n        inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))\n        inside_mean = cv2.mean(g, mask=inner)[0]\n\n        rec = dict(quad=quad, ar=ar, w_frac=float(w_frac),\n                   aratio=float(aratio), peri=float(peri), inside_mean=float(inside_mean))\n\n        # 細長コラム？\n        if (ar >= COLUMN_AR_MIN and hh/h >= COLUMN_MIN_H_FRAC):\n            columns.append(rec)\n            continue\n\n        # “紙っぽい”条件\n        if (FRAME_MIN_ARATIO <= aratio <= FRAME_MAX_ARATIO and\n            FRAME_PAPER_AR_MIN <= ar <= FRAME_PAPER_AR_MAX and\n            w_frac >= FRAME_MIN_W_FRAC and\n            inside_mean >= FRAME_INSIDE_MEAN_MIN):\n            # スコア：面積大・中心近い・目標アスペクトに近い\n            qcx,qcy = quad.mean(axis=0)\n            dist = ((qcx-cx)**2 + (qcy-cy)**2)**0.5\n            aspect_penalty = abs(ar - TARGET_ASPECT) / ASPECT_TOL  # 小さいほど良い\n            score = aratio - FRAME_CENTER_BIAS*dist - 0.15*aspect_penalty\n            rec["score"] = float(score)\n            paperish.append(rec)\n\n    info = dict(step="inner_frame_detect_v3",\n                paperish_candidates=len(paperish),\n                column_candidates=len(columns))\n\n    best = None\n    if paperish:\n        best = max(paperish, key=lambda r: r["score"])\n    else:\n        # コラムが複数並んでいる場合は外接矩形に“合成”\n        if len(columns) >= 2:\n            xs = []; ys = []\n            for r in columns:\n                q = r["quad"]\n                xs.extend([q[:,0].min(), q[:,0].max()])\n                ys.extend([q[:,1].min(), q[:,1].max()])\n            x0, x1 = max(0, int(min(xs))), min(w-1, int(max(xs)))\n            y0, y1 = max(0, int(min(ys))), min(h-1, int(max(ys)))\n            # 合成矩形を紙っぽいか再チェック\n            ww = x1 - x0 + 1; hh = y1 - y0 + 1\n            if ww>10 and hh>10:\n                ar = hh/ww; aratio = (ww*hh)/img_area; w_frac = ww/w\n                if (FRAME_MIN_ARATIO <= aratio <= FRAME_MAX_ARATIO and\n                    FRAME_PAPER_AR_MIN <= ar <= FRAME_PAPER_AR_MAX and\n                    w_frac >= FRAME_MIN_W_FRAC):\n                    quad = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)\n                    qcx,qcy = quad.mean(axis=0)\n                    dist = ((qcx-cx)**2 + (qcy-cy)**2)**0.5\n                    aspect_penalty = abs(ar - TARGET_ASPECT) / ASPECT_TOL\n                    score = aratio - FRAME_CENTER_BIAS*dist - 0.15*aspect_penalty\n                    best = dict(quad=quad, ar=ar, w_frac=w_frac,\n                                aratio=aratio, inside_mean=999, score=float(score))\n    if best is not None:\n        info.update(dict(found=True,\n                         quad=best["quad"].tolist(),\n                         ar=float(best["ar"]), w_frac=float(best["w_frac"]),\n                         aratio=float(best["aratio"]), score=float(best["score"])))\n        return best["quad"].astype(np.float32), info\n    else:\n        info.update(dict(found=False))\n        return None, info\n\n\n# ===== メイン処理 =====\ndef process_one(path: Path) -> Dict[str, Any]:\n    stem, ext = path.stem, (path.suffix.lower() if path.suffix else ".jpg")\n    img0 = imread(path)\n\n    img1, info_edge = auto_trim_black_edges(img0)\n\n    quad, info_frame = detect_frame_or_compose(img1)\n    route = "frame" if quad is not None else "paper_mask"\n\n    # ここに記録用フィールド（仕上げ寄せのログ）\n    refine_info: Dict[str, Any] = {}\n\n    if quad is not None:\n        out_img = warp_by_quad(img1, quad, FRAME_INNER_MARGIN_RATIO)\n        info_crop = dict(step="warp_by_quad", route=route,\n                         inner_margin_ratio=FRAME_INNER_MARGIN_RATIO)\n\n        # ★ frame ルートでも仕上げ寄せを適用（過切り防止のため控えめ）\n        ratios = measure_edge_white_ratio(out_img, band_frac=0.02, thr=225)\n        ih, iw = out_img.shape[:2]\n        inset  = suggest_insets_from_ratios(ratios, iw, ih)\n        ty, by = inset["top"], inset["bottom"]\n        lx, rx = inset["left"], inset["right"]\n        if ty+by < ih-4 and lx+rx < iw-4:\n            out_img = out_img[ty:ih-by, lx:iw-rx]\n            refine_info = dict(edge_white_ratios=ratios, applied_insets=inset)\n\n    else:\n        # 紙マスクでフォールバック\n        mask2, info_mask = build_paper_mask(img1)\n        mask3, area_ratio = largest_cc(mask2)\n        if area_ratio >= MIN_AREA_RATIO and mask3.max()>0:\n            ys, xs = np.where(mask3>0)\n            y0,y1 = int(ys.min()), int(ys.max())\n            x0,x1 = int(xs.min()), int(xs.max())\n            h,w = img1.shape[:2]\n            x0 = max(0, x0+SAFE_MARGIN_PX); y0 = max(0, y0+SAFE_MARGIN_PX)\n            x1 = min(w-1, x1-SAFE_MARGIN_PX); y1 = min(h-1, y1-SAFE_MARGIN_PX)\n            out_img = img1[y0:y1+1, x0:x1+1]\n            info_crop = dict(step="crop_by_mask", route=route, area_ratio=float(area_ratio),\n                             x0=x0,y0=y0,x1=x1,y1=y1)\n\n            # ★ 仕上げ寄せ（白さベース）\n            ratios = measure_edge_white_ratio(out_img, band_frac=0.02, thr=225)\n            ih, iw = out_img.shape[:2]\n            inset  = suggest_insets_from_ratios(ratios, iw, ih)\n            ty, by = inset["top"], inset["bottom"]\n            lx, rx = inset["left"], inset["right"]\n            if ty+by < ih-4 and lx+rx < iw-4:\n                out_img = out_img[ty:ih-by, lx:iw-rx]\n                refine_info = dict(edge_white_ratios=ratios, applied_insets=inset)\n        else:\n            out_img = img1.copy()\n            info_crop = dict(step="fallback_edge_only", route=route, area_ratio=float(area_ratio))\n\n    # ===== 保存 =====\n    dbg = DEBUG_DIR\n    imwrite(dbg/f"{stem}_00_src{ext}", img0)\n    imwrite(dbg/f"{stem}_10_edge_trim{ext}", img1)\n\n    if quad is not None:\n        dbg_frame = img1.copy()\n        cv2.polylines(dbg_frame, [order_quad(quad).astype(int)], True, (0,0,255), 3)\n        imwrite(dbg/f"{stem}_15_frame_on_img{ext}", dbg_frame)\n    else:\n        mask2,_ = build_paper_mask(img1)\n        imwrite(dbg/f"{stem}_20_mask_raw.png", mask2)\n        imwrite(dbg/f"{stem}_21_mask_on_img{ext}", visualize_mask(img1, mask2))\n        m3,_ = largest_cc(mask2)\n        imwrite(dbg/f"{stem}_30_mask_lcc.png", m3)\n        imwrite(dbg/f"{stem}_31_mask_lcc_on_img{ext}", visualize_mask(img1, m3))\n\n    out = OUTPUT_DIR / f"{stem}{ext if ext in [\'.jpg\',\'.jpeg\',\'.png\'] else \'.jpg\'}"\n    imwrite(out, out_img)\n\n    meta = dict(\n        file=str(path),\n        route=route,\n        steps=[info_edge, info_frame, info_crop],\n        refine=refine_info,  # 仕上げ寄せのログ\n        params=dict(\n            DARK_RATIO_EDGE=DARK_RATIO_EDGE, DARK_THRESH=DARK_THRESH,\n            HSV_V_MIN=HSV_V_MIN, HSV_S_MAX=HSV_S_MAX,\n            FRAME_MIN_W_FRAC=FRAME_MIN_W_FRAC,\n            FRAME_PAPER_AR_MIN=FRAME_PAPER_AR_MIN, FRAME_PAPER_AR_MAX=FRAME_PAPER_AR_MAX,\n            TARGET_ASPECT=TARGET_ASPECT\n        )\n    )\n    (DEBUG_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))\n    return meta\n\ndef main():\n    ensure_dirs()\n    files: List[Path] = []\n    for pat in ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]:\n        files += sorted(INPUT_DIR.glob(pat))\n    if not files:\n        print(f"[INFO] No images in {INPUT_DIR}")\n        return\n    print(f"[INFO] files:", len(files))\n    for i,p in enumerate(files,1):\n        try:\n            print(f"[{i}/{len(files)}] {p.name}")\n            process_one(p)\n        except Exception as e:\n            print(f"[ERROR] {p.name}: {e}")\n    print("[DONE] complete:", OUTPUT_DIR, " debug:", DEBUG_DIR)\n\n# (main() removed in embedded version)\n'

# ====== a.py をモジュール化 ======
a_mod = types.ModuleType("a_embedded")
# a.py では from pathlib import Path / cv2 / numpy / json を使っているため、
# こちらのグローバルを流用できるようにモジュールの __dict__ に注入
a_globals = a_mod.__dict__
a_globals.update({
    "Path": Path,
    "cv2": cv2,
    "np": np,
    "json": json,
})
exec(A_PY_SOURCE, a_globals)

# ===== Dropbox / パイプライン設定 =====
SRC_BASE   = "/おうち書道/共有データ/【受講生】/【添削用　作品】"
DST_PRINT  = SRC_BASE + "/添削用印刷未"
DST_FIXED  = SRC_BASE + "/補正済元画像"
DST_FAIL   = SRC_BASE + "/補正失敗"
DST_DEBUG  = SRC_BASE + "/_debug"

A4_DPI = int(os.getenv("A4_DPI", "300"))
A4_W_IN, A4_H_IN = 8.27, 11.69
A4_W_PX, A4_H_PX = int(round(A4_W_IN*A4_DPI)), int(round(A4_H_IN*A4_DPI))
MATCH_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}

# ===== a.py の process_one をラップ =====
def trim_image(in_path: str, out_path: str, debug_dir: str) -> bool:
    src = Path(in_path)
    out = Path(out_path)
    dbg = Path(debug_dir)
    tmp_out_dir = out.parent / "_tmp_out"
    tmp_out_dir.mkdir(parents=True, exist_ok=True)

    # a.py の出力/デバッグ先を差し替え
    a_mod.OUTPUT_DIR = tmp_out_dir
    a_mod.DEBUG_DIR  = Path(debug_dir)
    if hasattr(a_mod, "ensure_dirs"):
        a_mod.ensure_dirs()

    # 実行
    a_mod.process_one(src)

    # 生成物（stem + .jpg/.jpeg/.png）を out_path に移動
    ext = src.suffix.lower()
    save_ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
    produced = tmp_out_dir / f"{src.stem}{save_ext}"
    if produced.exists():
        produced.replace(out)
        return True
    return False

# ===== Dropbox 認証 =====
ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
CLIENT_ID     = os.getenv("DROPBOX_CLIENT_ID")
CLIENT_SECRET = os.getenv("DROPBOX_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")

def get_access_token() -> str:
    if ACCESS_TOKEN:
        return ACCESS_TOKEN
    url = "https://api.dropboxapi.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN or "",
        "client_id": CLIENT_ID or "",
        "client_secret": CLIENT_SECRET or "",
    }
    r = requests.post(url, data=data, timeout=30)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise RuntimeError(f"Dropbox token refresh failed ({r.status_code}): {err}")
    return r.json()["access_token"]

def ensure_folders(dbx: dropbox.Dropbox):
    for p in [DST_PRINT, DST_FIXED, DST_FAIL, DST_DEBUG]:
        try:
            dbx.files_create_folder_v2(p)
        except dropbox.exceptions.ApiError:
            pass

def list_target_files(dbx: dropbox.Dropbox):
    result = []
    entries = dbx.files_list_folder(SRC_BASE, recursive=True)
    while True:
        for e in entries.entries:
            if isinstance(e, dropbox.files.FileMetadata):
                ext = os.path.splitext(e.name)[1].lower()
                if ext in MATCH_EXTS and not any(e.path_lower.startswith(p.lower()) for p in [DST_PRINT.lower(), DST_FIXED.lower(), DST_FAIL.lower(), DST_DEBUG.lower()]):
                    result.append(e.path_lower)
        if not entries.has_more:
            break
        entries = dbx.files_list_folder_continue(entries.cursor)
    return result

def download_to_temp(dbx: dropbox.Dropbox, dropbox_path: str, tmpdir: Path) -> Path:
    local = tmpdir / Path(dropbox_path).name
    md, res = dbx.files_download(dropbox_path)
    with open(local, "wb") as f:
        f.write(res.content)
    return local

def pdf_to_image(pdf_path: Path, out_png: Path) -> bool:
    if fitz is None:
        print("[ERROR] PyMuPDF が無いので PDF を画像化できません:", pdf_path)
        return False
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            return False
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out_png))
        return True
    except Exception as e:
        print("[ERROR] PDF変換失敗:", e)
        return False

def rotate_to_portrait(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > h:
        return img.rotate(90, expand=True)
    return img

def to_a4_canvas(img: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (A4_W_PX, A4_H_PX), (255, 255, 255))
    img = rotate_to_portrait(img)
    w, h = img.size
    scale = min(A4_W_PX / w, A4_H_PX / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    ox = (A4_W_PX - new_w) // 2
    oy = (A4_H_PX - new_h) // 2
    canvas.paste(img_resized, (ox, oy))
    return canvas

def upload_bytes(dbx: dropbox.Dropbox, data: bytes, path: str):
    dbx.files_upload(data, path, mode=WriteMode("overwrite"))

def move_remote(dbx: dropbox.Dropbox, src: str, dst: str):
    try:
        dbx.files_delete_v2(dst)
    except dropbox.exceptions.ApiError:
        pass
    dbx.files_move_v2(src, dst, allow_shared_folder=True, autorename=False)

def process_one(dbx: dropbox.Dropbox, remote_path: str) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        local_src = download_to_temp(dbx, remote_path, tdir)
        stem = local_src.stem

        work_image = local_src
        if local_src.suffix.lower() == ".pdf":
            png_path = tdir / f"{stem}.png"
            ok = pdf_to_image(local_src, png_path)
            if not ok:
                move_remote(dbx, remote_path, f"{DST_FAIL}/{local_src.name}")
                return False, "PDF→画像化に失敗"
            work_image = png_path

        debug_dir = tdir / "_debug"
        debug_dir.mkdir(exist_ok=True)

        trimmed_path = tdir / f"{stem}_trimmed.png"
        ok = trim_image(str(work_image), str(trimmed_path), str(debug_dir))
        if not ok or not Path(trimmed_path).exists():
            move_remote(dbx, remote_path, f"{DST_FAIL}/{Path(remote_path).name}")
            for p in debug_dir.glob("*"):
                with open(p, "rb") as f:
                    upload_bytes(dbx, f.read(), f"{DST_DEBUG}/{stem}/{p.name}")
            return False, "トリミング失敗"

        img = Image.open(trimmed_path).convert("RGB")
        img = rotate_to_portrait(img)
        a4 = to_a4_canvas(img)

        out_remote = f"{DST_PRINT}/{stem}_A4_{A4_DPI}dpi.png"
        buf = io.BytesIO()
        a4.save(buf, format="PNG", dpi=(A4_DPI, A4_DPI))
        upload_bytes(dbx, buf.getvalue(), out_remote)

        move_remote(dbx, remote_path, f"{DST_FIXED}/{Path(remote_path).name}")

        for p in debug_dir.glob("*"):
            with open(p, "rb") as f:
                upload_bytes(dbx, f.read(), f"{DST_DEBUG}/{stem}/{p.name}")

        return True, "完了"

def main():
    token = get_access_token()
    dbx = dropbox.Dropbox(token, timeout=60)
    ensure_folders(dbx)

    targets = list_target_files(dbx)
    if not targets:
        print("[INFO] 対象ファイルはありません")
        return

    print(f"[INFO] 対象 {len(targets)} 件")
    ok_cnt = 0
    ng_cnt = 0
    for i, path in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {path}")
        try:
            ok, msg = process_one(dbx, path)
            if ok:
                ok_cnt += 1
            else:
                ng_cnt += 1
            print("  ->", msg)
        except Exception as e:
            print("  -> 例外:", e)
            try:
                move_remote(dbx, path, f"{DST_FAIL}/{Path(path).name}")
            except Exception:
                pass
            ng_cnt += 1

    print(f"[DONE] 成功 {ok_cnt} 件 / 失敗 {ng_cnt} 件")

if __name__ == "__main__":
    main()
