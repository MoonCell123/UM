import os
import warnings

dll_candidates = [
    os.environ.get("OPENSLIDE_BIN") or os.environ.get("OPENSLIDE_DLL_DIR"),
    r"D:\Lijinghan\work\openslide-bin-4.0.0.5-windows-x64\bin",
    os.path.abspath("./openslide-win64-20171122/bin"),
]
for dll_path in dll_candidates:
    if dll_path and os.path.isdir(dll_path):
        os.add_dll_directory(os.path.abspath(dll_path))
        break
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import openslide
from PIL import Image
import glob

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning)

SUPPORTED_SLIDE_EXTS = (".svs", ".ndpi", ".kfb", ".tif", ".tiff", ".mrxs")


def resolve_slide_path(slide_dir, slide_name, exts=SUPPORTED_SLIDE_EXTS):
    if not slide_dir or not slide_name:
        return None
    for ext in exts:
        candidate = os.path.join(slide_dir, slide_name + ext)
        if os.path.exists(candidate):
            return candidate
        candidate_upper = os.path.join(slide_dir, slide_name + ext.upper())
        if os.path.exists(candidate_upper):
            return candidate_upper
    try:
        base_lower = slide_name.lower()
        for filename in os.listdir(slide_dir):
            name, ext = os.path.splitext(filename)
            if name.lower() == base_lower and ext.lower() in exts:
                return os.path.join(slide_dir, filename)
    except FileNotFoundError:
        return None
    return None


def extract_coords_from_filename(filename):
    try:
        name_without_ext = os.path.splitext(filename)[0]
        parts = name_without_ext.split('_')
        x_coord = int(parts[0][1:])
        y_coord = int(parts[1][1:])
        return x_coord, y_coord
    except:
        return None, None


def get_slide_dimensions(slide_path):
    try:
        if not slide_path or not os.path.exists(slide_path):
            return 10000, 10000
        ext = os.path.splitext(slide_path)[1].lower()
        if ext == ".kfb":
            from wsi_core.KfbSlide import kfbslide
            slide_obj = kfbslide.open_kfbslide(slide_path)
            if slide_obj is None: raise RuntimeError
            try:
                return slide_obj.dimensions
            finally:
                slide_obj.close()
        else:
            slide_obj = openslide.OpenSlide(slide_path)
            try:
                return slide_obj.dimensions
            finally:
                slide_obj.close()
    except:
        return 10000, 10000


# ==================== 关键修改：新的色柱设计 ====================

def create_custom_cmap():
    """
    创建自定义色柱：

    设计逻辑：
    - 0.0 (Low): 深紫黑色
    - 0.5 (Mid): 浅橙色/淡粉橙
    - 1.0 (High): 深橙红色（最深最饱和）

    效果：在0.5以上，注意力越高，橙红色越深！
    """
    colors = [
        # 低注意力区域 (0.0 - 0.5): 深色 → 浅色
        (0.00, (0.001, 0.001, 0.015)),  # 纯黑
        (0.15, (0.12, 0.03, 0.28)),  # 深紫黑
        (0.30, (0.30, 0.08, 0.45)),  # 深紫色
        (0.45, (0.55, 0.20, 0.50)),  # 紫红色

        # 中间过渡点 (0.5): 最浅的橙色
        (0.50, (0.98, 0.75, 0.60)),  # 浅橙色/淡粉橙（最浅点）

        # 高注意力区域 (0.5 - 1.0): 浅色 → 深色（越高越深！）
        (0.60, (0.96, 0.60, 0.45)),  # 浅橙色
        (0.70, (0.92, 0.45, 0.30)),  # 橙色
        (0.80, (0.85, 0.30, 0.20)),  # 深橙色
        (0.90, (0.75, 0.18, 0.12)),  # 深橙红
        (1.00, (0.60, 0.10, 0.05)),  # 最深的橙红色/暗红色
    ]
    return LinearSegmentedColormap.from_list('custom_inverted', colors, N=256)


def normalize_attention_scores(scores, clip_percentiles=(1, 99),
                               method="percentile", gamma=1.0, score_floor=0.0):
    """归一化注意力分数"""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores, None, None

    if method == "rank":
        if scores.size == 1:
            norm = np.array([0.5], dtype=np.float32)
        else:
            order = np.argsort(scores)
            norm = np.empty_like(scores, dtype=np.float32)
            norm[order] = np.linspace(0.0, 1.0, scores.size, dtype=np.float32)
        low, high = float(scores.min()), float(scores.max())
    else:
        low = float(np.percentile(scores, clip_percentiles[0]))
        high = float(np.percentile(scores, clip_percentiles[1]))
        if high > low:
            norm = (np.clip(scores, low, high) - low) / (high - low)
        else:
            norm = np.full_like(scores, 0.5, dtype=np.float32)

    if gamma != 1.0:
        norm = np.power(norm, gamma)
    if score_floor > 0.0:
        norm = score_floor + norm * (1.0 - score_floor)
    return norm, low, high


# ==================== 渲染方法 ====================

def render_method_magma_overlay(patch_array, attention_score, cmap, alpha_min=0.25, alpha_max=0.65):
    """叠加模式"""
    color_norm = np.array(cmap(attention_score)[:3])
    color_layer = color_norm * 255
    alpha = alpha_min + attention_score * (alpha_max - alpha_min)
    patch_float = patch_array.astype(np.float32)
    blended = patch_float * (1 - alpha) + color_layer * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_method_magma_multiply(patch_array, attention_score, cmap):
    """乘法混合模式"""
    color_norm = np.array(cmap(attention_score)[:3])
    patch_float = patch_array.astype(np.float32) / 255.0
    color_adjusted = color_norm * 0.7 + 0.3
    blended = patch_float * color_adjusted
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def render_method_magma_screen(patch_array, attention_score, cmap):
    """滤色混合模式"""
    color_norm = np.array(cmap(attention_score)[:3])
    patch_float = patch_array.astype(np.float32) / 255.0
    screen_strength = attention_score * 0.5
    screened = 1 - (1 - patch_float) * (1 - color_norm * screen_strength)
    return np.clip(screened * 255, 0, 255).astype(np.uint8)


def create_heatmap_from_attention_csv(slide_name, base_paths, attention_csv_path,
                                      thumbnail_scale=8, patch_size=256,
                                      background_color=(255, 255, 255),
                                      render_mode='overlay',
                                      alpha_min=0.25, alpha_max=0.65,
                                      clip_percentiles=(1, 99),
                                      score_mapping="percentile",
                                      score_gamma=1.0,
                                      score_floor=0.0):
    """创建热图"""

    custom_cmap = create_custom_cmap()

    print(f"\n处理: {slide_name}")
    print(f"  渲染模式: {render_mode}")
    if render_mode == 'overlay':
        print(f"  Alpha范围: {alpha_min} - {alpha_max}")
    print(f"  Score Mapping: {score_mapping}, Gamma: {score_gamma}, Floor: {score_floor}")

    slide_path = resolve_slide_path(base_paths['slide_dir'], slide_name)
    if not slide_path:
        print(f"  错误: 找不到切片文件")
        return False

    patches_path = os.path.join(base_paths['patches_dir'], f"{slide_name}")
    if not os.path.exists(patches_path):
        print(f"  错误: Patch目录不存在: {patches_path}")
        return False
    if not os.path.exists(attention_csv_path):
        print(f"  错误: CSV文件不存在: {attention_csv_path}")
        return False

    try:
        attention_df = pd.read_csv(attention_csv_path)
        coord_to_attention = {}
        for idx, row in attention_df.iterrows():
            coord_to_attention[(int(row['coords_x']), int(row['coords_y']))] = row['attention_score']

        width, height = get_slide_dimensions(slide_path)

        h, w = height // thumbnail_scale, width // thumbnail_scale
        heatmap = np.ones((h, w, 3), dtype=np.uint8)
        heatmap[:, :, 0] = background_color[0]
        heatmap[:, :, 1] = background_color[1]
        heatmap[:, :, 2] = background_color[2]

        image_files = [f for f in os.listdir(patches_path) if f.lower().endswith('.png')]
        print(f"  找到 {len(image_files)} 个patch文件")

        all_scores = []
        valid_files = []
        for f in image_files:
            x, y = extract_coords_from_filename(f)
            if x is not None and (x, y) in coord_to_attention:
                all_scores.append(coord_to_attention[(x, y)])
                valid_files.append((f, x, y, coord_to_attention[(x, y)]))

        print(f"  匹配到CSV中的patch: {len(valid_files)} 个")

        if not all_scores:
            print(f"  错误: 没有匹配的patch")
            return False

        all_scores = np.array(all_scores, dtype=np.float32)
        norm_scores, low_v, high_v = normalize_attention_scores(
            all_scores,
            clip_percentiles=clip_percentiles,
            method=score_mapping,
            gamma=score_gamma,
            score_floor=score_floor
        )

        print(f"  原始分数范围: {all_scores.min():.6f} - {all_scores.max():.6f}")
        if low_v is not None and high_v is not None:
            print(f"  归一化边界 ({clip_percentiles[0]}%-{clip_percentiles[1]}%): {low_v:.6f} - {high_v:.6f}")

        if render_mode == 'multiply':
            render_func = lambda p, s, c: render_method_magma_multiply(p, s, c)
        elif render_mode == 'screen':
            render_func = lambda p, s, c: render_method_magma_screen(p, s, c)
        else:
            render_func = lambda p, s, c: render_method_magma_overlay(p, s, c, alpha_min, alpha_max)

        placed_count = 0
        for i, (image_file, x_coord, y_coord, raw_score) in enumerate(valid_files):
            score_norm = float(norm_scores[i]) if norm_scores.size else 0.5

            x_thumb = int(x_coord / thumbnail_scale)
            y_thumb = int(y_coord / thumbnail_scale)
            ps_thumb = max(1, patch_size // thumbnail_scale)

            if (x_thumb >= 0 and y_thumb >= 0 and
                    x_thumb + ps_thumb <= w and y_thumb + ps_thumb <= h):

                try:
                    patch_img = Image.open(os.path.join(patches_path, image_file)).convert('RGB')
                    patch_img = patch_img.resize((ps_thumb, ps_thumb), Image.BILINEAR)
                    patch_arr = np.array(patch_img)

                    rendered = render_func(patch_arr, score_norm, custom_cmap)

                    heatmap[y_thumb:y_thumb + ps_thumb, x_thumb:x_thumb + ps_thumb, :] = rendered
                    placed_count += 1
                except Exception as e:
                    continue

        print(f"  成功放置: {placed_count} 个patch")

        output_filename = f'heatmap_{slide_name}_magma_{render_mode}.pdf'
        output_path = os.path.join(base_paths['output_dir'], output_filename)

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(heatmap)
        ax.axis('off')
        ax.set_title(f'Attention Heatmap - {slide_name}\n(Magma {render_mode.capitalize()} Mode)',
                     fontsize=16, pad=20)

        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Score\n(Low → High)', rotation=270, labelpad=25, fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  保存: {output_path}")
        return True

    except Exception as e:
        import traceback
        print(f"  错误: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # ========== 路径配置 ==========
    base_paths = {
        'slide_dir': r'D:\svs\WMU_F1',
        'patches_dir': r'D:\Datas_of_Lab\crc\注意力热图使用的数据\热图用patch\温附一',
        'attention_csv_dir': r'D:\Datas_of_Lab\crc\注意力热图使用的数据\注意力分数\温附一',
        'output_dir': r'D:\Datas_of_Lab\crc\注意力热图使用的数据\热图输出\温附一',
    }

    os.makedirs(base_paths['output_dir'], exist_ok=True)

    # ========== 参数配置 ==========
    thumbnail_scale = 8
    patch_size = 224

    # ========== 渲染模式 ==========
    render_mode = 'multiply'  # 'overlay', 'multiply', 'screen'

    # overlay模式参数
    alpha_min = 0.25
    alpha_max = 0.65

    # 分数映射参数
    clip_percentiles = (0, 100)
    score_mapping = "rank"  # "percentile" 或 "rank"
    score_gamma = 1.0  # 1.0 = 线性映射
    score_floor = 0.0

    print("=" * 80)
    print("注意力热图生成器 (高注意力=深色)")
    print("=" * 80)
    print(f"渲染模式: {render_mode}")
    print(f"配色逻辑:")
    print(f"  Low (0.0)  → 深紫黑色")
    print(f"  Mid (0.5)  → 浅橙色（最浅）")
    print(f"  High (1.0) → 深橙红色（最深）")
    print()

    slide_files = []
    for ext in SUPPORTED_SLIDE_EXTS:
        slide_files.extend(glob.glob(os.path.join(base_paths['slide_dir'], f"*{ext}")))

    print(f"找到 {len(slide_files)} 个切片文件")
    print()

    success_count = 0
    fail_count = 0

    for slide_idx, slide_file in enumerate(slide_files, 1):
        slide_name = os.path.splitext(os.path.basename(slide_file))[0]
        attention_csv_path = os.path.join(base_paths['attention_csv_dir'], f"{slide_name}.csv")

        print(f"[{slide_idx}/{len(slide_files)}] {slide_name}")

        if os.path.exists(attention_csv_path):
            if create_heatmap_from_attention_csv(
                    slide_name, base_paths, attention_csv_path,
                    thumbnail_scale=thumbnail_scale,
                    patch_size=patch_size,
                    render_mode=render_mode,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    clip_percentiles=clip_percentiles,
                    score_mapping=score_mapping,
                    score_gamma=score_gamma,
                    score_floor=score_floor
            ):
                success_count += 1
            else:
                fail_count += 1
        else:
            print(f"  跳过: 找不到CSV文件")
            fail_count += 1

    print()
    print("=" * 80)
    print(f"完成! 成功: {success_count}, 失败: {fail_count}")
    print("=" * 80)

