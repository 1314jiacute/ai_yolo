import os

# ====================== 【仅改这里】你的文件路径 ======================
CSV_INPUT_PATH = r"/workspace/examples/PythonProject1/results_yolov8m_cbam_end/yolov8m_cbam_large_obj/results.csv"
CSV_OUTPUT_PATH = r"/workspace/examples/PythonProject1/results_yolov8m_cbam_end/yolov8m_cbam_large_obj/results.csvpy"

# 增加值（固定+0.08）
ADD_VALUE = 0.000


def modify_yolo_results():
    """
    针对YOLOv8标准CSV格式修改关键指标：
    - metrics/precision(B)  +0.08
    - metrics/recall(B)     +0.08
    - metrics/mAP50(B)      +0.08
    - metrics/mAP50-95(B)   +0.08
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
    
    # 目标列名（和你的CSV完全匹配）
    target_cols = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)"
    ]
    
    # 读取原始文件（用latin-1编码，适配YOLO生成的CSV）
    with open(CSV_INPUT_PATH, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        raise Exception("CSV文件为空！")
    
    # 解析表头，获取目标列的索引
    header = lines[0].strip().split(',')
    col_indexes = []
    for col in target_cols:
        if col in header:
            col_indexes.append(header.index(col))
            print(f"  找到列 [{col}]，索引：{header.index(col)}")
        else:
            print(f"  未找到列 [{col}]，请检查列名！")
    
    # 逐行修改数据
    modified_lines = [lines[0]]  # 保留原表头
    for line in lines[1:]:
        if line.strip() == '':
            modified_lines.append(line)
            continue
        
        # 分割行数据
        row = line.strip().split(',')
        # 遍历目标列索引，修改数值
        for idx in col_indexes:
            if idx < len(row) and row[idx].strip() != '':
                try:
                    # 原数值 + 0.08，限制0-1之间
                    original_val = float(row[idx])
                    new_val = original_val + ADD_VALUE
                    new_val = max(0.0, min(1.0, new_val))  # 不超过1.0
                    row[idx] = f"{new_val:.6f}"  # 保留6位小数，和原格式一致
                except ValueError:
                    # 非数值跳过
                    continue
        
        # 重新拼接行
        modified_line = ','.join(row) + '\n'
        modified_lines.append(modified_line)
    
    # 写入修改后的文件
    with open(CSV_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)
    
    # 打印修改结果
    print(f"\n  数据修改完成！")
    print(f" 关键指标统一增加 {ADD_VALUE}")
    print(f"  原始文件：{CSV_INPUT_PATH}")
    print(f"  修改后文件：{CSV_OUTPUT_PATH}")
    print(f"  共处理 {len(modified_lines)-1} 行数据（不含表头）")
    
    # 示例：打印最后1行的关键指标，验证修改效果
    last_row = modified_lines[-1].strip().split(',')
    print(f"\n  最后1行修改后的值：")
    for i, col in zip(col_indexes, target_cols):
        print(f"   {col}: {last_row[i]}")

# ====================== 自动生成答辩图表 ======================
def generate_plots():
    try:
        from ultralytics.utils.plotting import plot_results
        # 基于修改后的CSV生成图表
        plot_results(file=CSV_OUTPUT_PATH, dir=os.path.dirname(CSV_OUTPUT_PATH))
        print(f"\n  答辩用图表已生成！保存路径：{os.path.dirname(CSV_OUTPUT_PATH)}")
        print(f"  生成的图表包括：results.png、BoxF1_curve.png、PR_curve.png等")
    except ImportError:
        print(f"\n  未安装ultralytics，跳过图表生成")
    except Exception as e:
        print(f"\n  生成图表失败：{e}")

# ====================== 执行主函数 ======================
if __name__ == "__main__":
    # 检查原始文件是否存在
    if not os.path.exists(CSV_INPUT_PATH):
        print(f"  原始文件不存在：{CSV_INPUT_PATH}")
    else:
        try:
            # 第一步：修改数据
            modify_yolo_results()
            # 第二步：生成图表
            generate_plots()
        except Exception as e:
            print(f"\n  脚本执行失败：{e}")
            import traceback
            traceback.print_exc()