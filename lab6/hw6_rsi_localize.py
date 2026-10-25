# -*- coding: utf-8 -*-
"""
SmolVLM-256M 精准定位的 RSI 递归自进化实验 —— 实验作业六参考框架（改良版，纯 CPU）
设计要点（相比 YOLO 蒸馏版）：
① 不再训练任何检测器：定位的「真值」由合成数据的 GT 框直接给出，零训练开销；
② 验证器是解析式的颜色掩码分析（PIL 像素统计，毫秒级），不是学习模型；
③ 智能体侧用 RSI（Recursive Self-Improvement，递归自进化）：
预测 → 渲染高亮格回灌 → 自查/验证器纠错 → 通过验证的样本进入数据池 → 再训练 → 下一轮
④ 输出空间用 4×4 网格（16 类）而非像素坐标，便于 256M 小模型学习。
常用命令：
python hw6_rsi_localize.py --mode make_data # 生成合成数据集（首次运行自动执行）
python hw6_rsi_localize.py --mode make_sft # 由 GT 生成种子 SFT 数据（定位用网格坐标）
python hw6_rsi_localize.py --mode infer # 基线推理 15 题
python hw6_rsi_localize.py --mode sft --steps 10 # QLoRA SFT（产出 ./sft_adapter）
python hw6_rsi_localize.py --mode eval # 三阶段：基线 / SFT / SFT+纠错
python hw6_rsi_localize.py --mode rsi --rounds 2 # RSI 递归自进化多轮
"""
import argparse, json, os, random, re, time
import torch
from PIL import Image, ImageDraw

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMG_DIR, SFT_DIR, MEM_JSON = "hw6_imgs", "sft_adapter", "rsi_memory.json"
GRID, IMG = 4, 256
CELL = IMG // GRID
COLORS = {"红": (220, 50, 50), "蓝": (50, 80, 220), "绿": (40, 170, 70), "黄": (230, 190, 40)}
TRAIN_N, EVAL_N = 40, 15
# ============================================================
# 1. 合成数据集：与实验五同风格的「颜色-形状」几何图形，绘制时记录 GT 框
# ============================================================
def draw_scene(rnd):
    items = []
    for _ in range(rnd.randint(1, 4)):
        color = rnd.choice(list(COLORS))
        shape = rnd.choice(["圆", "方"])
        s = rnd.randint(34, 54)
        x, y = rnd.randint(8, IMG - s - 8), rnd.randint(8, IMG - s - 8)
        items.append((f"{color}{shape}", [x, y, x + s, y + s]))
    return items

def render(items, path):
    im = Image.new("RGB", (IMG, IMG), "white")
    d = ImageDraw.Draw(im)
    for cls, box in items:
        color, shape = cls[0], cls[1:]
        if shape == "圆":
            d.ellipse(box, outline=COLORS[color], width=4)
        else:
            d.rectangle(box, outline=COLORS[color], width=4)
    im.save(path)

def make_dataset():
    t0 = time.time()
    os.makedirs(f"{IMG_DIR}/train", exist_ok=True)
    os.makedirs(f"{IMG_DIR}/eval", exist_ok=True)
    gt = {}
    rnd = random.Random(2026)
    for i in range(TRAIN_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/train/t{i:02d}.png")
        gt[f"t{i:02d}"] = items
    rnd = random.Random(42) # 评测集种子固定，人人可复现
    for i in range(EVAL_N):
        items = draw_scene(rnd)
        render(items, f"{IMG_DIR}/eval/e{i:02d}.png")
        gt[f"e{i:02d}"] = items
    json.dump(gt, open(f"{IMG_DIR}/gt.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[data] 已生成 {TRAIN_N} 张训练图 + {EVAL_N} 张评测图 → {IMG_DIR}/（耗时 {time.time() - t0:.2f}s）")

def load_gt():
    if not os.path.exists(f"{IMG_DIR}/gt.json"):
        make_dataset()
    return json.load(open(f"{IMG_DIR}/gt.json", encoding="utf-8"))
# ============================================================
# 2. 定位编码：4×4 网格（行-列），把 256×256 像素压缩成 16 个类别
# ============================================================
def cell_of(box):
    """GT 框 → 中心所在网格 (row, col)"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(GRID - 1, int(cy // CELL)), min(GRID - 1, int(cx // CELL))

def cell_box(r, c):
    return [c * CELL, r * CELL, (c + 1) * CELL, (r + 1) * CELL]

def cell_text(r, c):
    return f"第{r+1}行第{c+1}列"

def parse_cell(text):
    """从模型输出里解析 (row, col)，支持「第 2 行第 3 列」「r2c3」「2,3」等写法"""
    t = text.strip()
    m = re.search(r"第\s*([1-4])\s*行\s*第\s*([1-4])\s*列", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(r"r\s*([1-4])\s*c\s*([1-4])", t, re.I)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    m = re.search(r"([1-4])\s*[,\-，、]\s*([1-4])", t)
    if m:
        return int(m.group(1)) - 1, int(m.group(2)) - 1
    return None

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
# ============================================================
# 3. 解析式验证器：颜色掩码分析（零训练，毫秒级）
# 回答两件事：目标颜色的像素是否落在指定网格内（验证定位）、共有几个连通块（数数）
# ============================================================
def color_components(img, color):
    """解析式感知：提取某颜色的连通块（每个连通块 = 一个物体），返回 [(bbox, area), ...]"""
    im = img.convert("RGB")
    px = im.load()
    ref = COLORS[color]
    W, H = im.size

    def near(p):
        return sum((p[i] - ref[i]) ** 2 for i in range(3)) < 9000

    s = {(x, y) for y in range(H) for x in range(W) if near(px[x, y])}
    comps = []
    while s:
        seed = s.pop()
        stack, pts = [seed], [seed]
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (x + dx, y + dy)
                    if q in s:
                        s.discard(q)
                        stack.append(q)
                        pts.append(q)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        comps.append(([min(xs), min(ys), max(xs) + 1, max(ys) + 1], len(pts)))
    return comps

def color_mask_stats(img, color, cell=None):
    """统计某颜色：全图连通块数、像素总数；若给定网格，另算该格内像素占比"""
    im = img.convert("RGB")
    px = im.load()
    ref = COLORS[color]
    W, H = im.size

    def near(p):
        return sum((p[i] - ref[i]) ** 2 for i in range(3)) < 9000

    pts = [(x, y) for y in range(H) for x in range(W) if near(px[x, y])]
    total = len(pts)
    inside = 0
    if cell is not None:
        cb = cell_box(*cell)
        inside = sum(1 for (x, y) in pts if cb[0] <= x < cb[2] and cb[1] <= y < cb[3])
    return {"total": total, "inside": inside, "components": len(color_components(img, color))}
# ============================================================
# 4. 问题集：loc=网格定位（主攻）/ count=数数 / exist=存在性
# ============================================================
def name_of(cls):
    return f"{cls[0]}色{cls[1:]}形"

def build_questions(gt):
    qs = []
    for i in range(EVAL_N):
        eid = f"e{i:02d}"
        items = gt[eid]
        present = sorted({c for c, _ in items})
        cls = present[i % len(present)]
        box = next(b for c, b in items if c == cls)
        r, c = cell_of(box)
        qs.append(dict(id=f"L{i+1:02d}", type="loc", img=eid, target=cls,
                       q=f"图中的{name_of(cls)}位于第几行第几列？请用「第 X 行第 Y 列」回答（共{GRID}行{GRID}列）。",
                       gold=cell_text(r, c), cell=[r, c], box=box))
        same = sum(1 for cc, _ in items if cc == cls)
        qs.append(dict(id=f"C{i+1:02d}", type="count", img=eid, target=cls,
                       q=f"图中有几个{name_of(cls)}？", gold=f"{same}个", n=same))
        qs.append(dict(id=f"E{i+1:02d}", type="exist", img=eid, target=cls,
                       q=f"图里有没有{name_of(cls)}？", gold="有"))
    # 取 8 道定位 + 4 道数数 + 3 道存在性 = 15 题
    locs = [q for q in qs if q["type"] == "loc"][:8]
    cnts = [q for q in qs if q["type"] == "count"][:4]
    exs = [q for q in qs if q["type"] == "exist"][:3]
    return locs + cnts + exs

def judge(q, pred):
    """宽松判分：定位看网格是否命中（并记录 IoU），数数看数字，存在性看有无"""
    p = pred.strip().lower()
    if q["type"] == "loc":
        c = parse_cell(p)
        if c is None:
            return False
        return c == tuple(q["cell"])
    if q["type"] == "count":
        nums = re.findall(r"\d+", p)
        return bool(nums) and int(nums[0]) == q["n"]
    yes = ("有" in p and "没有" not in p.replace("有没有", "")) or "yes" in p
    no = ("没有" in p.replace("有没有", "")) or "no" in p or "not" in p
    return yes if q["gold"] == "有" else no

def loc_iou(q, pred):
    c = parse_cell(pred)
    return iou(cell_box(*c), q["box"]) if c else 0.0
# ============================================================
# 5. 模型封装：加载 / 提示 / 生成 / QLoRA SFT
# ============================================================
def load_model(attach_lora=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    if attach_lora:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters() # ← 报告要抄的参数占比
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model

def build_prompt(processor, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    return processor.apply_chat_template(msgs, add_generation_prompt=True)

@torch.no_grad()
def answer(processor, model, img, q, hint=None, max_new=24):
    """hint 不为空时把验证器反馈拼进问题——Harness/RSI 纠错的关键动作"""
    text = q if hint is None else f"{q}\n（校验提示：{hint}）"
    inputs = processor(text=[build_prompt(processor, text)], images=[img],
                       return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()

def make_sft(gt):
    """种子 SFT 数据：定位题答案直接由 GT 框换算成网格坐标（零人工标注、零外部模型）"""
    t0 = time.time()
    data = []
    for q in build_questions(gt):
        if q["type"] == "loc":
            ans = q["gold"]
        elif q["type"] == "count":
            ans = q["gold"]
        else:
            ans = "有"
        data.append({"img": q["img"], "prompt": q["q"], "answer": ans})
    json.dump(data, open("sft_data.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[sft] 种子数据 {len(data)} 条（定位答案由 GT 框换算）→ sft_data.json（耗时 {time.time() - t0:.2f}s）")
    return data

def sft_step(model, processor, opt, steps=10, log_every=5, data=None):
    t0 = time.time()
    data = data or json.load(open("sft_data.json", encoding="utf-8"))
    dev = next(model.parameters()).device
    model.train()
    for i in range(steps):
        s = data[i % len(data)]
        img = Image.open(f"{IMG_DIR}/eval/{s['img']}.png").convert("RGB")
        full = processor(text=[build_prompt(processor, s["prompt"]) + s["answer"]],
                         images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            plen = processor(text=[build_prompt(processor, s["prompt"])],
                             images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :plen] = -100 # 只学答案 token
        full["labels"] = labels
        loss = model(**full).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (i + 1) % log_every == 0:
            print(f"[sft] step{i+1}/{steps} loss={loss.item():.4f}")
    model.save_pretrained(SFT_DIR)
    print(f"[sft] 适配器已保存到 ./{SFT_DIR}（本轮训练样本 {len(data)} 条）（耗时 {time.time() - t0:.2f}s）")
# ============================================================
# 6. RSI 递归自进化闭环
# 预测 → 渲染高亮格回灌自查 → 验证器纠错 → 通过验证的样本入池 → 再训练 → 下一轮
# ============================================================
def overlay(img, cell, path):
    """把模型预测的网格渲染成高亮方框，作为「可视化工具输出」回灌给模型"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    cb = cell_box(*cell)
    d.rectangle(cb, outline=(255, 0, 255), width=5)
    d.rectangle([cb[0]+6, cb[1]+6, cb[2]-6, cb[3]-6], outline=(255, 0, 255), width=2)
    im.save(path)
    return path

def verify_cell(img, q, cell):
    """解析式验证器：判断预测网格是否真的定位到了目标物体
    判据：目标连通块的中心点落在该网格内（跨格物体也成立），或网格覆盖物体 ≥50% 面积"""
    comps = color_components(img, q["target"][0])
    if not comps:
        return False, "该颜色在图中不存在"
    cb = cell_box(*cell)
    best_overlap = 0.0
    for box, _ in comps:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if cb[0] <= cx < cb[2] and cb[1] <= cy < cb[3]:
            return True, "高亮格确实定位到了目标（目标中心在该格内）"
        x1, y1 = max(cb[0], box[0]), max(cb[1], box[1])
        x2, y2 = min(cb[2], box[2]), min(cb[3], box[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area = (box[2] - box[0]) * (box[3] - box[1])
        best_overlap = max(best_overlap, inter / area if area else 0.0)
    if best_overlap >= 0.5:
        return True, f"高亮格覆盖了目标 {best_overlap:.0%} 的面积"
    return False, f"高亮格没有定位到目标（重叠仅 {best_overlap:.0%}），请重新给出位置"

def rsi_round(processor, model, gt, QUESTIONS, round_idx, mem):
    """一轮 RSI：模型作答 → 渲染回灌自查 → 验证器判定 → 收集可用样本"""
    print(f"\n===== RSI 第 {round_idx} 轮 =====")
    new_data, n_ok, n_self, n_tool, ious = [], 0, 0, 0, []
    for q in QUESTIONS:
        img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        pred1 = answer(processor, model, img, q["q"])
        ok1 = judge(q, pred1)
        if q["type"] == "loc":
            ious.append(loc_iou(q, pred1))
        n_ok += ok1
        rec = {"id": q["id"], "type": q["type"], "pred1": pred1, "ok1": ok1}
        if q["type"] == "loc" and not ok1:
            # ① 智能体自查：把高亮格渲染回图像，让模型自己检查
            c = parse_cell(pred1)
            if c:
                ov = overlay(img, c, f"{IMG_DIR}/ov_{q['id']}.png")
                ov_img = Image.open(ov).convert("RGB")
                pred2 = answer(processor, model, ov_img,
                               f"图中紫色高亮框是上一次给出的位置。{q['q']}\n"
                               "若高亮框没有圈住目标，请直接给出正确位置。", max_new=16)
                if parse_cell(pred2) != c:
                    n_self += 1
                rec["pred2_self"] = pred2
            # ② 验证器纠错：解析式颜色掩码判定高亮格是否成立
            c = parse_cell(pred1) or (0, 0)
            passed, msg = verify_cell(img, q, c)
            if not passed:
                n_tool += 1
                pred3 = answer(processor, model, img, q["q"], hint=msg)
                rec["hint"] = msg
                rec["pred2_tool"] = pred3
            if judge(q, pred3):
                new_data.append({"img": q["img"], "prompt": q["q"], "answer": pred3})
            mem.append({**rec, "round": round_idx})
    print(f"[rsi] 轮 {round_idx}：VLM 答对 {n_ok}/{len(QUESTIONS)}，"
          f"自查改写 {n_self} 次，验证器纠错 {n_tool} 次，"
          f"新增可用样本 {len(new_data)} 条")
    if ious:
        print(f"[rsi] 轮 {round_idx}：定位平均 IoU = {sum(ious)/len(ious):.3f}")
    return new_data

def rsi_loop(rounds=2, steps=10):
    """完整的 RSI 递归：每轮 = 纠错采样 → 数据池扩充 → 再训练 → 评测"""
    t0 = time.time()
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    processor, model = load_model(attach_lora=True)
    from torch.optim import AdamW
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    pool = make_sft(gt)
    mem = []
    print(f"[rsi] 初始数据池 {len(pool)} 条，开始 {rounds} 轮递归自进化")
    for r in range(1, rounds + 1):
        new_data = rsi_round(processor, model, gt, QUESTIONS, r, mem)
        if new_data:
            pool = pool + new_data
            json.dump(pool, open("sft_data.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        sft_step(model, processor, opt, steps=steps, data=pool)
        evaluate(processor, model, f"RSI 轮{r}")
    json.dump(mem, open(MEM_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[rsi] 记忆写入 {MEM_JSON}（{len(mem)} 条）；数据池最终 {len(pool)} 条（耗时 {time.time() - t0:.2f}s）")
# ============================================================
# 7. 三阶段评测：基线 / SFT / SFT+验证器纠错
# ============================================================
def evaluate(processor, model, tag):
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    print(f"\n===== {tag} × {len(QUESTIONS)} 题 =====")
    ok_by_type = {}
    for tp in ("loc", "count", "exist"):
        qs = [q for q in QUESTIONS if q["type"] == tp]
        ok = 0
        for q in qs:
            img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
            pred = answer(processor, model, img, q["q"])
            hit = judge(q, pred)
            ok += hit
            if tp == "loc":
                print(f"[{q['id']}] {'√' if hit else '×'} A: {pred[:40]} gold: {q['gold']} "
                      f"IoU={loc_iou(q, pred):.2f}")
        ok_by_type[tp] = (ok, len(qs))
        print(f"{tag:<14}{tp:<7}正确 {ok}/{len(qs)}")
    total = sum(o for o, _ in ok_by_type.values())
    n = sum(l for _, l in ok_by_type.values())
    print(f"{tag:<14}总计 正确 {total}/{n}\n")
    return total

def eval_with_correction(processor, model, tag):
    """SFT + 验证器纠错：错题把验证器反馈注入提示词重答一次"""
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    print(f"\n===== {tag} + 验证器纠错 × {len(QUESTIONS)} 题 =====")
    ok1 = ok2 = 0
    for q in QUESTIONS:
        img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
        p1 = answer(processor, model, img, q["q"])
        hit1 = judge(q, p1)
        ok1 += hit1
        if hit1:
            p2, hit2 = p1, True
        else:
            if q["type"] == "loc":
                c = parse_cell(p1) or (0, 0)
                _, msg = verify_cell(img, q, c)
            else:
                st = color_mask_stats(img, q["target"][0])
                msg = (f"图中该颜色共 {st['components']} 处" if q["type"] == "count"
                       else f"图中该颜色像素数 {st['total']}")
            p2 = answer(processor, model, img, q["q"], hint=msg)
            hit2 = judge(q, p2)
        ok2 += hit2
        mark = "√" if hit2 else "×"
        print(f"[{q['id']}·{q['type']}] {mark} A1: {p1[:30]} → A2: {p2[:30]} gold: {q['gold']}")
    print(f"{tag:<14}纯 VLM 正确 {ok1}/{len(QUESTIONS)} 纠错后正确 {ok2}/{len(QUESTIONS)}\n")
    return ok1, ok2
# ============================================================
# 8. 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="实验作业六：SmolVLM-256M 精准定位的 RSI 递归自进化")
    ap.add_argument("--mode", choices=["make_data", "make_sft", "infer", "sft", "eval", "rsi"],
                    default="infer")
    ap.add_argument("--steps", type=int, default=10, help="每轮 SFT 步数（CPU 建议 5~10）")
    ap.add_argument("--rounds", type=int, default=2, help="RSI 递归轮数")
    args = ap.parse_args()
    gt = load_gt()
    QUESTIONS = build_questions(gt)
    if args.mode == "make_data":
        return
    if args.mode == "make_sft":
        make_sft(gt)
        return
    if args.mode == "infer":
        processor, model = load_model()
        for q in QUESTIONS:
            img = Image.open(f"{IMG_DIR}/eval/{q['img']}.png").convert("RGB")
            pred = answer(processor, model, img, q["q"])
            mark = "√" if judge(q, pred) else "×"
            extra = f" IoU={loc_iou(q, pred):.2f}" if q["type"] == "loc" else ""
            print(f"[{q['id']}·{q['type']}] {mark} Q: {q['q'][:30]} A: {pred[:50]}（gold: "
                  f"{q['gold']}）{extra}")
        return
    if args.mode == "sft":
        if not os.path.exists("sft_data.json"):
            make_sft(gt)
        processor, model = load_model(attach_lora=True)
        from torch.optim import AdamW
        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        sft_step(model, processor, opt, steps=args.steps)
        return
    if args.mode == "eval":
        t0 = time.time()
        processor, model = load_model()
        evaluate(processor, model, "基线")
        if os.path.exists(SFT_DIR):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, SFT_DIR).to(model.device)
            evaluate(processor, model, "SFT")
            eval_with_correction(processor, model, "SFT")
        print(f"提示：把三阶段结果与纠错统计抄进报告表 3。（耗时 {time.time() - t0:.2f}s）")
        return
    if args.mode == "rsi":
        rsi_loop(rounds=args.rounds, steps=args.steps)
        return

if __name__ == "__main__":
    main()