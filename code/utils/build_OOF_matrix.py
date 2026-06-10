"""
build_ensemble.py
Late Fusion — OOF Stacking

从 21 个模型的 5 折 predictions.csv 中提取 prob_class1，
构建 80×21 的 OOF 输入矩阵 + 标签向量。

输出:
  datasets/OOF_matrix.csv    — 80×21 (slide_id + 20个模型的 prob_class1)
  datasets/OOF_labels.csv    — 80×1  (slide_id + 真实标签 label)
"""

import pandas as pd
import numpy as np
import os
import sys
import re

# ── 配置 ──────────────────────────────────────────────────────────────
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
MODEL_NAMES = sorted([
    'conch_v1', 'conch_v15', 'gigapath', 'hibou_l',
    'hoptimus0', 'hoptimus1',
    'kaiko-vitb16', 'kaiko-vitb8', 'kaiko-vitl14',
    'kaiko-vits16', 'kaiko-vits8',
    'lunit-vits8', 'midnight12k', 'musk', 'phikon', 'phikon_v2',
    'resnet50', 'uni_v1', 'uni_v2', 'virchow', 'virchow2',
])
N_FOLDS = 5
LABEL_COL = 'label'       # predictions.csv 中的真实标签列名
PROB_COL = 'prob_class1'  # 正类概率列名
SLIDE_ID_COL = 'slide_id' # 切片 ID 列名
OUTPUT_DIR = os.path.join(DATASETS_DIR, 'ensemble')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_predictions():
    """
    遍历所有模型 × 所有 fold，加载 prob_class1。

    Returns
    -------
    slide_order : list[str]
        所有 slide_id 的出现顺序（按第一个模型的第一折为准）
    all_data : dict[str, dict[int, pd.DataFrame]]
        all_data[model_name][fold] = DataFrame with slide_id, prob_class1, label
    """
    all_data = {}
    slide_order = None

    for model in MODEL_NAMES:
        model_dir = os.path.join(DATASETS_DIR, model)
        if not os.path.isdir(model_dir):
            print(f'    模型目录不存在: {model_dir}')
            continue

        fold_data = {}
        for fold in range(1, N_FOLDS + 1):
            csv_path = os.path.join(model_dir, f'fold_{fold}', 'predictions.csv')
            if not os.path.exists(csv_path):
                print(f'    {model}/fold_{fold}/predictions.csv 不存在')
                continue

            df = pd.read_csv(csv_path)

            # 检查必要列
            for col in [SLIDE_ID_COL, PROB_COL, LABEL_COL]:
                if col not in df.columns:
                    raise ValueError(f'{csv_path} 缺少列: {col}')

            # 只保留需要的列
            df = df[[SLIDE_ID_COL, PROB_COL, LABEL_COL]].copy()

            # 记录 slide 顺序（以第一个模型的第一折为准）
            if slide_order is None and model == MODEL_NAMES[0] and fold == 1:
                slide_order = df[SLIDE_ID_COL].tolist()

            fold_data[fold] = df

        if fold_data:
            all_data[model] = fold_data
            print(f'   {model}: {len(fold_data)} folds loaded')
        else:
            print(f'    {model}: 没有可用的 predictions')

    return slide_order, all_data


def build_ensemble_matrix(slide_order, all_data):
    """
    构建 OOF 集成矩阵。

    OOF (Out-Of-Fold) 原则:
      - fold_1 的预测 → fold_1 的测试集 slide
      - fold_2 的预测 → fold_2 的测试集 slide
      - ...
    对于每个模型, 将所有 5 折的测试集预测拼起来, 就得到该模型在所有 slide 上的 OOF 预测。

    因此对于每个模型, 将所有 fold 的 prob_class1 按 slide 合并即可。

    Returns
    -------
    matrix_df : pd.DataFrame
        columns = [slide_id, model1, model2, ..., modelN]
        rows = 所有 slide, 按 slide_order 排序
    labels_df : pd.DataFrame
        columns = [slide_id, label]
    """
    # 首先收集所有 slide 上的真实标签
    # 从第一个有数据的模型的 fold_1 获取真实标签
    first_model = next(iter(all_data.keys()))
    label_map = {}  # slide_id -> label
    for fold in range(1, N_FOLDS + 1):
        df = all_data[first_model][fold]
        for _, row in df.iterrows():
            label_map[row[SLIDE_ID_COL]] = int(row[LABEL_COL])

    all_slide_ids = sorted(label_map.keys())
    print(f'\n总共有 {len(all_slide_ids)} 个 slide')

    # 构建矩阵: 每一列是一个模型的 prob_class1
    matrix_dict = {SLIDE_ID_COL: all_slide_ids}

    for model in MODELS_WITH_DATA:
        model_probs = {}  # slide_id -> prob_class1

        for fold in range(1, N_FOLDS + 1):
            df = all_data[model][fold]
            for _, row in df.iterrows():
                sid = row[SLIDE_ID_COL]
                prob = row[PROB_COL]
                # 如果同一个 slide 出现在多个 fold（不应该发生但做保护）
                model_probs[sid] = prob

        # 按 all_slide_ids 顺序取出
        col_values = [model_probs.get(sid, np.nan) for sid in all_slide_ids]
        nan_count = sum(1 for v in col_values if np.isnan(v))
        if nan_count > 0:
            print(f'    {model}: {nan_count}/{len(all_slide_ids)} 个 slide 缺失')

        matrix_dict[model] = col_values

    matrix_df = pd.DataFrame(matrix_dict)

    # 标签
    labels_df = pd.DataFrame({
        SLIDE_ID_COL: all_slide_ids,
        LABEL_COL: [label_map[sid] for sid in all_slide_ids],
    })

    return matrix_df, labels_df


def validate_integrity(matrix_df, labels_df):
    """校验数据的完整性"""
    print('\n' + '=' * 60)
    print('完整性校验')
    print('=' * 60)
    print(f'  矩阵形状: {matrix_df.shape} (应约为 80 × {len(MODELS_WITH_DATA) + 1})')
    print(f'  标签形状: {labels_df.shape} (应约为 80 × 2)')

    # 检查缺失值
    n_missing = matrix_df.isna().sum().sum()
    print(f'  缺失值总数: {n_missing}')

    # 检查 slide_id 一致性
    assert all(matrix_df[SLIDE_ID_COL] == labels_df[SLIDE_ID_COL]), \
        'slide_id 不匹配！'
    print('slide_id 一致')

    # 检查标签分布
    print(f'  标签分布: D3={sum(labels_df[LABEL_COL]==0)}, M3={sum(labels_df[LABEL_COL]==1)}')

    # 显示头部预览
    print(f'\n矩阵前 5 行前 5 列:')
    print(matrix_df.head().to_string())
    print(f'\n标签前 5 行:')
    print(labels_df.head().to_string())

def add_label():
    ENSEMBLE_DIR = r'G:\Fanglun\Project\UM\datasets\ensemble'

    m = pd.read_csv(f'{ENSEMBLE_DIR}/OOF_matrix.csv')
    l = pd.read_csv(f'{ENSEMBLE_DIR}/OOF_labels.csv')

    m['label'] = l['label']

    m.to_csv(f'{ENSEMBLE_DIR}/OOF_matrix.csv', index=False, encoding='utf-8-sig', float_format='%.6f')

    print(f'OOF_matrix.csv 已更新: 形状 {m.shape}')
    print(f'新增 label 列, 共 {len(m.columns)} 列')
    print(f'列名: {m.columns.tolist()[:3]} ... {m.columns.tolist()[-3:]}')

def main():
    print('=' * 60)
    print('OOF Stacking — 集成矩阵构建')
    print('=' * 60)
    print(f'模型数量: {len(MODEL_NAMES)}')
    print(f'折数: {N_FOLDS}')
    print(f'数据目录: {DATASETS_DIR}')
    print(f'输出目录: {OUTPUT_DIR}')
    print()

    # Step 1: 加载所有数据
    print('正在加载所有 predictions...')
    slide_order, all_data = load_all_predictions()

    global MODELS_WITH_DATA
    MODELS_WITH_DATA = sorted(all_data.keys())
    print(f'\n成功加载 {len(MODELS_WITH_DATA)} 个模型的数据')

    # Step 2: 构建矩阵
    print('\n正在构建 OOF 集成矩阵...')
    matrix_df, labels_df = build_ensemble_matrix(slide_order, all_data)

    # Step 3: 校验
    validate_integrity(matrix_df, labels_df)

    # Step 4: 保存
    matrix_path = os.path.join(OUTPUT_DIR, 'OOF_matrix.csv')
    labels_path = os.path.join(OUTPUT_DIR, 'OOF_labels.csv')

    matrix_df.to_csv(matrix_path, index=False, encoding='utf-8-sig', float_format='%.6f')
    labels_df.to_csv(labels_path, index=False, encoding='utf-8-sig')

    print(f'\n 集成矩阵已保存: {matrix_path}')
    print(f' 标签文件已保存: {labels_path}')
    print(f'\n后续使用:')
    print(f'  X = pd.read_csv("{matrix_path}")  # {matrix_df.shape}')
    print(f'  y = pd.read_csv("{labels_path}")  # {labels_df.shape}')
    print('  # 去掉 slide_id 列后, X 即为 80×N 的输入矩阵')
    print('  # y 为真实标签')

    # Step 5: 增加label列
    add_label()



if __name__ == '__main__':
    MODELS_WITH_DATA = []
    main()
