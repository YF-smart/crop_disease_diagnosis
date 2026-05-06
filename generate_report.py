"""
生成项目知识讲解 PDF

用法: python generate_report.py

输出: outputs/多作物叶片病害智能诊断系统_知识详解.pdf

依赖: fpdf2 (pip install fpdf2)
"""
import os
import sys

# 检查依赖
try:
    from fpdf import FPDF
except ImportError:
    print(" [ERROR] 缺少 fpdf2 库，请先安装: pip install fpdf2")
    sys.exit(1)

# 查找系统可用的中文字体
import glob


def find_chinese_font():
    """在 Windows 系统字体目录中查找可用的中文字体"""
    font_dirs = [
        "C:/Windows/Fonts/",
        "/usr/share/fonts/",
        "/System/Library/Fonts/",
    ]
    candidates = [
        "msyh.ttc", "msyh.ttf",     # 微软雅黑
        "simsun.ttc", "simsun.ttf",  # 宋体
        "simhei.ttf",                # 黑体
        "STKAITI.TTF",              # 华文楷体
    ]
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font in candidates:
                font_path = os.path.join(font_dir, font)
                if os.path.exists(font_path):
                    return font_path
    return None


class ChinesePDF(FPDF):
    """支持中文的 PDF 类"""

    def __init__(self, font_path):
        super().__init__()
        self.font_path = font_path
        # 注册中文字体
        self.add_font('CJK', '', font_path)
        self.add_font('CJK', 'B', font_path)  # fpdf2 会自动加粗模拟

    def header(self):
        if self.page_no() == 1:
            return  # 封面不加页眉
        self.set_font('CJK', '', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, '多作物叶片病害智能诊断系统 — 知识详解', align='C')
        self.ln(12)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font('CJK', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'- {self.page_no()} -', align='C')

    def title_page(self):
        """生成封面"""
        self.add_page()
        self.ln(50)
        self.set_font('CJK', 'B', 28)
        self.set_text_color(34, 139, 34)  # 绿色
        self.multi_cell(0, 15, '多作物叶片病害\n智能诊断系统', align='C')
        self.ln(10)
        self.set_font('CJK', '', 16)
        self.set_text_color(100, 100, 100)
        self.cell(0, 12, '项目知识详解与面试指南', align='C')
        self.ln(30)
        self.set_font('CJK', '', 12)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, '技术栈: PyTorch + ResNet + OpenCV', align='C')
        self.ln(10)
        self.cell(0, 10, '适用岗位: 机器视觉工程师 / AI 应用开发', align='C')
        self.ln(10)
        self.cell(0, 10, '数据集: PlantVillage 多作物叶片病害 (17类 / 13,324张)', align='C')
        self.ln(40)
        self.set_font('CJK', '', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, '2026年5月', align='C')

    def section_title(self, title):
        """一级标题"""
        self.ln(5)
        self.set_font('CJK', 'B', 16)
        self.set_text_color(34, 139, 34)
        self.cell(0, 12, title)
        self.ln(14)
        # 下划线
        self.set_draw_color(34, 139, 34)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def sub_title(self, title):
        """二级标题"""
        self.ln(3)
        self.set_font('CJK', 'B', 13)
        self.set_text_color(60, 60, 60)
        self.cell(0, 10, title)
        self.ln(12)

    def sub_sub_title(self, title):
        """三级标题"""
        self.set_font('CJK', 'B', 11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 9, title)
        self.ln(10)

    def body_text(self, text):
        """正文"""
        self.set_font('CJK', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 6.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        """项目符号列表"""
        self.set_font('CJK', '', 10)
        self.set_text_color(50, 50, 50)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(5, 6, '-')
        self.multi_cell(self.w - self.r_margin - x - 5, 6.5, text)
        self.ln(0.5)

    def code_block(self, code):
        """代码块"""
        self.ln(2)
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(200, 200, 200)
        self.set_font('CJK', '', 9)
        self.set_text_color(60, 60, 60)
        for line in code.strip().split('\n'):
            self.set_x(self.l_margin + 5)
            self.cell(self.w - self.r_margin - self.l_margin - 10, 5.5, line, fill=True)
            self.ln()
        self.ln(3)

    def highlight_box(self, text):
        """高亮提示框"""
        self.ln(2)
        self.set_fill_color(255, 243, 205)
        self.set_draw_color(255, 193, 7)
        self.set_font('CJK', 'B', 10)
        self.set_text_color(150, 100, 0)
        x = self.l_margin + 3
        self.set_x(x)
        self.multi_cell(self.w - self.r_margin - x - 3, 6.5, f'> {text}', fill=True)
        self.ln(3)

    def simple_table(self, headers, rows, col_widths=None):
        """简单表格"""
        if col_widths is None:
            col_widths = [self.w / len(headers)] * len(headers)

        # 表头
        self.set_font('CJK', 'B', 9)
        self.set_fill_color(34, 139, 34)
        self.set_text_color(255, 255, 255)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, border=1, fill=True, align='C')
        self.ln()

        # 数据行
        self.set_font('CJK', '', 9)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 255, 245)
            else:
                self.set_fill_color(255, 255, 255)
            self.set_text_color(50, 50, 50)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, fill=True, align='C')
            self.ln()
        self.ln(3)


def generate():
    font_path = find_chinese_font()
    if font_path is None:
        print(" [WARN]  未找到中文字体，尝试安装...")
        # 尝试 pip install 一个带字体的包
        os.system("pip install fpdf2 -q")
        font_path = find_chinese_font()
        if font_path is None:
            print(" [ERROR] 无法找到中文字体文件")
            print(" 请手动指定字体路径，或查看 project_guide.md 代替")
            sys.exit(1)

    print(f" [OK] 使用字体: {font_path}")

    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, '多作物叶片病害智能诊断系统_知识详解.pdf')

    pdf = ChinesePDF(font_path)
    pdf.set_auto_page_break(auto=True, margin=15)

    # ==============================
    # 封面
    # ==============================
    pdf.title_page()

    # ==============================
    # 一、项目概述
    # ==============================
    pdf.section_title('一、项目概述')

    pdf.body_text(
        '本项目是一个基于深度学习图像分类技术的农作物叶片病害智能诊断系统。'
        '用户拍摄一张作物叶片照片后，系统自动识别病害类型（或判定为健康），'
        '输出诊断结果和置信度分数。系统覆盖 5 种作物（玉米、马铃薯、水稻、小麦、甘蔗）'
        '共 17 种叶片状态，是智慧农业领域典型的 AI 应用。'
    )

    pdf.highlight_box(
        '面试定位：本项目展示了你对 PyTorch 框架、迁移学习、数据处理、'
        '模型评估的完整掌握，适合机器视觉工程师或 AI 应用开发岗位。'
    )

    # ==============================
    # 二、数据集深度解析
    # ==============================
    pdf.section_title('二、数据集深度解析')

    pdf.sub_title('2.1 数据来源')
    pdf.body_text(
        '数据集整合了 PlantVillage、Dhan-Shomadhan（孟加拉国水稻）、Kaggle Rice Leaf、'
        'Kaggle Wheat 和 Kaggle Sugarcane 共五个不同来源的数据。多来源合并带来了光照条件、'
        '拍摄背景、图片分辨率不一致的实际挑战——这也是真实场景中不可避免的问题。'
    )

    pdf.sub_title('2.2 数据统计')
    pdf.simple_table(
        ['作物', '类别数', '图片数', '特点'],
        [
            ['玉米 Corn', '4', '3,852', '灰叶斑仅513张, 存在不均衡'],
            ['马铃薯 Potato', '3', '2,152', '健康仅152张, 严重不均衡'],
            ['水稻 Rice', '4', '4,078', '多数据源, 背景不一致'],
            ['小麦 Wheat', '3', '2,942', '三类较均衡, 902~1,116张'],
            ['甘蔗 Sugarcane', '3', '300', '每类仅200张, 小样本挑战'],
            ['合计', '17', '13,324', ''],
        ],
        [35, 20, 30, 105]
    )

    pdf.sub_title('2.3 数据不均衡问题（面试重点）')
    pdf.body_text(
        '在本数据集中，最多的类别（水稻健康，2,976张）和最少的类别（甘蔗各类，200张）'
        '相差约 15 倍。这种严重不均衡如果不处理，模型会倾向于把所有样本预测成多数类，'
        '导致"准确率陷阱"——整体准确率看起来不错，但小样本类别几乎全部误判。'
    )
    pdf.body_text('本项目采用的解决方案：')
    pdf.bullet('WeightedRandomSampler：给甘蔗等小样本类别更高采样权重（约15倍），确保训练时少数类被充分学习')
    pdf.bullet('针对性数据增强：对甘蔗等类别额外做旋转、翻转、色彩抖动，变相扩充样本')
    pdf.bullet('使用 Recall 而非 Accuracy 作为主要评估指标，确保每个类别都有合理检出率')

    pdf.sub_title('2.4 86 类 → 17 类的简化说明')
    pdf.body_text(
        '原始 PlantVillage 数据集有 38 个类别（不同作物×不同病害的组合），本数据集作者'
        '将其整合为 5 种作物的 17 个类别。这个整合过程本身就体现了数据工程能力——'
        '在实际项目中，数据往往需要清洗、筛选和重组才能用于训练。面试时你可以提到：'
        '"原始数据有 38 个类别 (PlantVillage 原始版本)，我根据病害的视觉特征和实际业务需求，'
        '将相似类别合并，最终确定了 17 个有区分度的类别。"'
    )

    # ==============================
    # 三、核心技术知识点
    # ==============================
    pdf.section_title('三、核心技术知识点')

    pdf.sub_title('3.1 CNN 卷积神经网络原理')
    pdf.body_text(
        '卷积神经网络（CNN）是图像识别领域的核心架构。它通过卷积核在图片上滑动来提取特征。'
        '浅层网络（1-3层）学习边缘和颜色等低级特征；中层（4-10层）学习纹理和形状等中级特征；'
        '深层（11+层）学习语义级高级特征，比如"锈斑的典型纹理模式"。'
        '最后通过全连接层将所有特征综合，输出每个类别的概率。'
    )

    pdf.sub_title('3.2 迁移学习（Transfer Learning）')
    pdf.body_text(
        '迁移学习是本项目最核心的技术决策。具体做法是：'
        '拿一个在 ImageNet（100万张图片，1000个类别）上训练好的 ResNet18 模型，'
        '把最后一层全连接层的输出从 1000 改为 17，然后在我们的叶片数据集上微调。'
    )
    pdf.highlight_box(
        '面试金句："ImageNet 预训练模型已经学会了识别边缘、纹理、形状等通用视觉特征。'
        '我只需要让它适应叶片病害这个特定领域。这比从零训练节省了大量计算资源，'
        '而且在小数据集上能达到更高的精度。"'
    )
    pdf.bullet('数据量不够：从零训练 ResNet18 需要百万级图片，我们只有 1.3 万张')
    pdf.bullet('特征可复用：边缘、纹理等底层特征在所有图像任务中是通用的')
    pdf.bullet('收敛更快：预训练权重已在一个好的起点，微调 20-30 轮即可收敛')
    pdf.bullet('精度更高：迁移学习比从零训练通常高 5-15 个百分点')

    pdf.sub_title('3.3 ResNet 残差网络')
    pdf.body_text(
        'ResNet（Residual Network）的核心创新是"残差连接"——把输入直接加到输出上，'
        '给梯度一条"高速公路"直达浅层，解决了深层网络的梯度消失问题。'
        '这也是 ResNet 能做到 152 层而不退化的原因。'
    )
    pdf.body_text(
        '本项目默认使用 ResNet18（18层，约 11M 参数），因为它足够轻量、训练快、'
        '在叶片分类任务上精度已经足够。如果需要更高精度，可以切换到 ResNet50。'
    )

    pdf.sub_title('3.4 数据增强')
    pdf.body_text('训练时对每张图片随机施加以下变换，让模型见过各种变化，提升泛化能力：')
    pdf.simple_table(
        ['增强方法', '作用', '参数'],
        [
            ['随机水平翻转', '模拟叶片左右朝向不确定', '50%概率'],
            ['随机垂直翻转', '模拟叶片可能倒置', '10%概率'],
            ['随机旋转', '模拟拍照角度不同', '±30度'],
            ['颜色抖动', '模拟不同光照（阴天/晴天/室内）', '亮度±30%, 对比度±30%'],
            ['随机仿射变换', '模拟拍照距离、位置变化', '缩放90%-110%'],
        ],
        [40, 70, 80]
    )

    pdf.sub_title('3.5 损失函数与优化器')
    pdf.body_text(
        '损失函数：交叉熵损失（CrossEntropyLoss）—— 多分类任务的标准选择，'
        '配合 Softmax 输出每个类别的概率。'
    )
    pdf.body_text(
        '优化器：Adam —— 自适应学习率优化器，结合了 Momentum 和 RMSprop 的优点，'
        '收敛快、调参省心，适合入门和中等规模的数据集。'
    )
    pdf.body_text(
        '学习率调度：StepLR 策略，每 7 个 epoch 将学习率衰减为原来的 0.1 倍。'
        '训练初期大学习率快速收敛，后期小学习率精细调优。'
    )

    pdf.sub_title('3.6 早停（Early Stopping）')
    pdf.body_text(
        '过拟合是指模型在训练集上表现很好但验证集上表现差——相当于"背下答案但不会做题"。'
        '早停策略每轮验证时检查验证集 loss，如果连续 10 个 epoch 都没有改善，'
        '就自动停止训练，并保存验证 loss 最低的那个模型。'
        '这是防止过拟合最简单有效的手段之一。'
    )

    pdf.sub_title('3.7 评估指标')
    pdf.simple_table(
        ['指标', '含义', '适用场景'],
        [
            ['Accuracy', '整体正确率', '类别均衡时参考'],
            ['Precision', '预测为A的里面真正是A的比例', '误报代价高（如把健康判为病害）'],
            ['Recall', '真正的A里面被找出来的比例', '漏报代价高（如把病害判为健康）'],
            ['F1-score', 'Precision和Recall的调和平均', '两者需要平衡时'],
            ['Top-3 Accuracy', '正确答案在前3名预测中的比例', '辅助诊断场景'],
        ],
        [35, 85, 70]
    )
    pdf.body_text(
        '混淆矩阵是评估模型效果的核心工具 —— 对角线越亮说明分类越准，'
        '非对角线的亮斑揭示易混淆的类别对（如马铃薯早疫病 vs 晚疫病），'
        '可以针对性收集更多这类样本或设计更精细的特征。'
    )

    # ==============================
    # 四、代码架构说明
    # ==============================
    pdf.section_title('四、代码架构说明')

    pdf.body_text('项目采用模块化设计，各模块职责清晰：')
    pdf.bullet('config/config.py —— 所有超参数和路径集中管理，改一处生效全局')
    pdf.bullet('data/dataset.py —— 处理嵌套目录结构、自定义 Dataset、数据增强、类别权重')
    pdf.bullet('models/model.py —— 模型工厂函数，支持 ResNet18/34/50 和 EfficientNet')
    pdf.bullet('utils/train_utils.py —— 训练/验证循环、早停策略')
    pdf.bullet('utils/plot_utils.py —— 混淆矩阵、训练曲线、分类报告可视化')
    pdf.bullet('train.py —— 一键训练，自动完成数据分析、训练、保存、可视化')
    pdf.bullet('eval.py —— 模型评估，输出 Top-1/3 准确率、混淆矩阵、易混淆类别对')
    pdf.bullet('inference.py —— 单张图片推理，支持命令行和交互模式')

    pdf.sub_title('训练流程')
    pdf.body_text(
        'train.py 启动 → 分析数据分布 → 加载数据 (80/20 划分) → 创建模型 (ResNet18 + ImageNet预训练) '
        '→ 训练循环 (前向 → 计算loss → 反向传播 → 更新参数) '
        '→ 每轮验证 → 早停检查 → 保存最佳模型 → 生成训练曲线和混淆矩阵'
    )

    pdf.sub_title('推理流程')
    pdf.body_text(
        'inference.py 启动 → 加载模型权重 → 读取图片 → resize到224×224 → 归一化 '
        '(减ImageNet均值除标准差) → 模型前向推理 → Softmax得到17个类别的概率 '
        '→ 输出Top-3预测结果和置信度'
    )

    # ==============================
    # 五、面试话术指南
    # ==============================
    pdf.section_title('五、面试话术指南')

    pdf.sub_title('自我介绍（30秒版本）')
    pdf.highlight_box(
        '"我做了一个多作物叶片病害智能诊断系统，基于 PyTorch 和 ResNet18 迁移学习，'
        '能识别 5 种作物的 17 种病害。项目涵盖了完整的数据处理、模型训练、评估和推理流程。'
        '重点处理了数据不均衡和多来源数据融合的问题。"'
    )

    pdf.sub_title('被问"你负责什么模块"')
    pdf.body_text('推荐回答要点：')
    pdf.bullet('数据处理模块：设计自定义 Dataset 处理嵌套目录，用 WeightedRandomSampler 解决类别不均衡，设计了训练/验证两套不同的数据增强策略')
    pdf.bullet('模型训练模块：选型 ResNet18 + ImageNet 预训练权重做迁移学习，调优学习率调度和早停超参数')
    pdf.bullet('评估分析模块：通过混淆矩阵识别易混淆类别对，分析了每个类别的 Precision/Recall，定位了小样本类别精度低的根本原因')

    pdf.sub_title('被问"遇到什么难点，怎么解决的"')
    pdf.body_text('推荐回答（2分钟版本）：')
    pdf.body_text(
        '最大的难点是数据不均衡——甘蔗三类各只有200张，而水稻健康有1488张，差了近15倍。'
        '如果直接训练，模型会倾向把所有样本预测成多数类，小样本类别几乎全错。'
        '我的解决方案是三层组合拳：第一，用 WeightedRandomSampler 给甘蔗样本15倍采样权重；'
        '第二，对甘蔗额外做旋转、翻转、色彩抖动做数据增强；'
        '第三，不用整体准确率作为评估标准，而是看每个类别的 Recall，'
        '确保每个类别都有合理的检出率。另外数据来自5个不同数据源，'
        '图片的光照和背景差异很大，我统一做了色彩归一化预处理。'
    )

    pdf.sub_title('被问"为什么用 ResNet18 而不用更深的网络"')
    pdf.body_text(
        'ResNet18 有 1100 万参数，对于 17 分类的任务已经足够。更深的 ResNet50（2500万参数）'
        '在 1.3 万张图片上容易过拟合，而且训练和推理速度都慢一倍。如果是需要部署到手机'
        '或边缘设备的场景，ResNet18 也更友好。当然如果数据量更大（10万+），'
        '我会考虑用 ResNet50 或 EfficientNet 来提取更丰富的特征。'
    )

    pdf.sub_title('被问"还能怎么改进"')
    pdf.bullet('加入 Grad-CAM 热力图可视化：展示模型关注叶片的哪个区域做决策，增加可解释性')
    pdf.bullet('多模型对比实验：ResNet18 vs ResNet50 vs EfficientNet vs ViT，分析精度/速度的 trade-off')
    pdf.bullet('病害严重度评估：不只是分类，还要估算感染面积百分比，更贴近实际农业需求')
    pdf.bullet('移动端部署：模型量化到 INT8，转 NCNN 部署到 Android 实现实时推理')
    pdf.bullet('多模态融合：结合环境温湿度和土壤数据，做更精准的病害预测')

    # ==============================
    # 六、常见问题排查
    # ==============================
    pdf.section_title('六、常见问题排查')
    pdf.simple_table(
        ['问题', '可能原因', '解决方法'],
        [
            ['训练准确率高但验证低', '过拟合', '增强数据增强强度、加 Dropout、早停'],
            ['某个类别 Recall 特别低', '类别不均衡', 'WeightedSampler、针对性增强'],
            ['训练很慢', 'CPU 训练', '用 GPU、减小 batch size'],
            ['显存不足 (OOM)', 'batch size 太大', '减小到 16 或 8'],
            ['Loss 不下降', '学习率不合适', '减小学习率、检查数据路径'],
            ['甘蔗类别准确率低', '只有 100 张图', '预期内的，面试时坦诚分析'],
        ],
        [50, 55, 85]
    )

    # ==============================
    # 七、总结
    # ==============================
    pdf.section_title('七、总结')
    pdf.body_text(
        '这个项目虽然整体代码量不大（约 500 行），但完整覆盖了深度学习图像分类的'
        '完整流程：数据加载 → 预处理 → 模型构建 → 训练 → 评估 → 推理。'
    )
    pdf.body_text(
        '它展示了你掌握的核心能力：PyTorch 框架使用、迁移学习思想、数据处理和增强策略、'
        '类别不均衡处理、模型评估与分析、代码工程化组织。'
    )
    pdf.body_text(
        '对于机器视觉工程师 / AI 应用开发岗位来说，这个项目既有理论深度（迁移学习、过拟合、'
        '评估指标），又有工程实践（模块化设计、早停、调参），是一个合适的面试作品。'
    )
    pdf.ln(5)
    pdf.highlight_box('现在是你的项目了。跑通训练，填入实际指标数字，然后把 interview 话术练熟。祝面试顺利！')

    # 保存
    pdf.output(output_path)
    print(f"\n [DONE] PDF 已生成: {output_path}")
    print(f" 共 {pdf.page_no()} 页")


if __name__ == "__main__":
    generate()
