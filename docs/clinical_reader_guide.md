# AML Combo-Prediction Kit — Clinical Reader's Guide

**Version**: v0.2 (pre-registration, research use only)
**Audience**: AML clinical team (血液肿瘤科医师 + NGS 实验室 + 细胞遗传学 + 药师)
**Language**: Chinese with English clinical terms

> ⚠️ **使用范围**: 本 kit **不是 IVD（体外诊断）产品**，未经 NMPA/FDA 审批。
> 输出仅供**研究阶段辅助决策**使用，所有决定权在主治医师。

---

## 一、每个患者生成了哪些文件

对一个给定的 `patient_id`，kit 在 `runs/dna_reports/patient_<patient_id>/` 生成：

| 文件 | 用途 | 谁看 |
|---|---|---|
| `README.md` | 本患者专属概述 + 核心发现 + 快查 | 所有人 |
| `driver_mutations.csv` | **核心 25-gene 突变表** | 血液科、分子病理 |
| `fusion_analysis.csv` | 融合基因解读（如存在）| 细胞遗传学、血液科 |
| `cytogenetics.csv` | 核型 flag 对照表 | 细胞遗传学 |
| `targetability.csv` | 靶向药类别映射 | 药师、血液科 |
| `sample_qc.csv` | DNA 级样本 QC | NGS 实验室 |
| `dna_summary.json` | 全量结构化数据（给 EHR 接入）| 信息科、下游程序 |
| `dna_profile.png` | **单页出版级图**（表 + 融合 + 核型 + 靶向）| 病历讨论 PPT、IRB |

同一患者的 **kit 主输出**（`KitOutput.*` 字段）还包括：
- `top_combinations` — Layer 3 MLP 组合 AUC 预测
- `top_regimens` — Layer 1 试验证据推荐方案
- `clonal_coverage` — Layer 2 克隆覆盖生物学打分
- `cautions` — 临床警告（TLS 风险、血小板低、年龄 >75 等）

---

## 二、如何读取每个文件（逐字段）

### 2.1 `driver_mutations.csv` — 核心突变表

这是整套输出的**招牌**。每行一个突变，列定义：

| 列名 | 示例 | 怎么读 |
|---|---|---|
| `gene` | `FLT3` | HUGO 基因符号 |
| `variant_type` | `ITD` / `missense` / `frameshift` / `nonsense` / `splice` / `TKD` | 变异类型 |
| `vaf` | `0.45` | 变异等位基因频率（0-1 小数）。>0.40 = 主克隆；<0.10 = 亚克隆 |
| `tier` | `1` / `2` / `3` | **Tier 1** = FDA 有靶向药；**Tier 2** = 强预后或治疗强度驱动；**Tier 3** = 背景 / 共突变 |
| `allelic_ratio` | `0.62` 或 `n/a (FLT3-only field)` | **仅 FLT3 专属**。ITD:WT reads 比值。**≥0.5 = 高负荷** |
| `ar_interpretation` | `HIGH (≥0.5) — adverse modifier...` | 模型对 AR 的文本判断 |
| `biallelic` | `True` / `n/a (CEBPA-only field)` | **仅 CEBPA 专属**。双等位 = WHO 2022 favorable 认定条件 |
| `allelic_interpretation` | `BIALLELIC → favorable ELN risk` | CEBPA 的等位解读 |
| `targetable_by` | `Midostaurin; Quizartinib; Gilteritinib; ...` 或 `no targeted therapy available` | 分号分隔的 FDA/研究阶段靶向药 |
| `eln_implication` | `ITD positive with high allelic ratio...` 或 `Adverse per ELN 2017 (driver mutation category)` 或 `no specific ELN 2017 rule (Tier-3 background)` | 对 ELN 2017 风险分层的具体贡献 |
| `is_adverse_driver` | `True` / `False` | ELN 2017 明确 adverse 的 bool flag（TP53/RUNX1/ASXL1）|
| `notes` | `ITD ≠ TKD — verify variant type...` | 临床注解、特殊警告、参考试验 |

**交叉核对步骤：**
1. 拿 NGS 实验室报告的 variant list → 对比 `gene` × `variant_type` 逐条匹配
2. 对 FLT3 行：核对 `allelic_ratio` 与实验室的 ITD 报告（警惕不同实验室 denominator 定义可能略不同）
3. 对 CEBPA 行：确认 `biallelic` 字段是否由实验室 flagged（如果 lab 没检则传 False，kit 不会假阳性）

**空格约定：**
本 kit **没有任何 raw blank cells**。所有"空"的意思都有 explicit marker：
- `n/a (FLT3-only field)` — 此字段对该基因不适用
- `n/a (CEBPA-only field)` — 同上
- `no targeted therapy available` — 无靶向药
- `no specific ELN 2017 rule (Tier-3 background)` — ELN 没定规则
- `Adverse per ELN 2017 (driver mutation category)` — 明确 adverse
- `untiered (gene not in core panel)` — 不在 25 基因核心列表

### 2.2 `fusion_analysis.csv` — 融合基因

| 列 | 说明 |
|---|---|
| `fusion` | 字符串，如 `PML-RARA`、`KMT2A-MLLT3`、`CBFB-MYH11` |
| `eln_implication` | 对应 ELN 风险定位 |
| `regimen_note` | **重要**：与该融合匹配的一线方案 |
| `notes` | 特殊警告（如 PML-RARA 会明确标 **"7+3 is INAPPROPRIATE"**）|

### 2.3 `cytogenetics.csv` — 核型标志

| `finding` | `present` | `interpretation` |
|---|---|---|
| Complex karyotype | True/False | ≥3 个独立异常（ELN adverse） |
| Monosomy 5/7 or del(5q)/del(7q) | True/False | MDS-related, Vyxeos 考虑 |
| del(17p) / TP53 locus | True/False | 17p 缺失影响 TP53 |
| Normal karyotype | True/False | 46,XX 或 46,XY 无异常 |

⚠️ **核型解析限制**：Regex 启发式覆盖约 85% 常见 ISCN 模式。复杂 / 罕见核型（如 `i(17q)`、`idic(X)`、`mar`、`der`）**需要人工 cytogeneticist 审阅确认**，kit 不替代。

### 2.4 `targetability.csv` — 靶向药映射

`finding → indicated_drug_class` 的 pivot 表：
```
finding              indicated_drug_class
FLT3                 Midostaurin
FLT3                 Quizartinib
FLT3                 Gilteritinib
PML-RARA             ATRA
PML-RARA             ATO (arsenic trioxide)
universal            Venetoclax (combination)
```

"universal" 行意思：**所有 AML 患者都可考虑 BCL2 抑制剂**（ven+aza 在 unfit 人群是 FDA 一线）。

### 2.5 `sample_qc.csv` — DNA 级 QC

| 字段 | 含义 |
|---|---|
| `n_mutations_called` | NGS 实验室 call 出来的突变总数 |
| `n_with_vaf_annotation` | 带 VAF 数字的突变数 |
| `n_above_vaf_0_20` | VAF ≥ 0.20 的突变数（建议这个比 n_mutations_called 更重要）|
| `karyotype_parsed` | 核型字符串是否成功解析（True/False）|
| `fusions_reported` | Lab 报告的融合数 |

**建议审查规则：**
- `n_mutations_called` = 0 → 要么是野生型 AML，要么是 NGS 有问题，要回头看 lab QC
- `n_above_vaf_0_20` / `n_mutations_called` < 30% → 主克隆不明显，克隆结构复杂
- `karyotype_parsed` = False → kit 没能 parse 这份核型，要人工填 karyo flags

### 2.6 `dna_summary.json` — 全量结构化数据

以上所有 CSV 的源数据 + 额外字段，UTF-8 JSON。给：
- EHR 系统接入
- Hospital dashboard 程序读取
- 审计追溯（每次 kit 跑都留一份）

### 2.7 `dna_profile.png` — 单页可视图

2×2 网格：
- **A (上方，横跨)**: 核心突变表，Tier 1 行绿色高亮，adverse 驱动红色高亮
- **B (左下)**: 融合分析 + 核型 flag（红=异常，绿=正常）
- **C (右下)**: 靶向药类别清单
- 底部：Sample QC 小字

**用途：** 病例讨论 PPT、IRB 材料、纸质病历附件。300 DPI PNG，可 A4 打印清晰。

### 2.8 `top_combinations` — Layer 3 MLP 预测

这里每一行是模型预测的 top-K 药对（默认 K=5），字段：
| 字段 | 含义 |
|---|---|
| `rank` | 1..K |
| `drug1`, `drug2` | 两个 BeatAML 药名 |
| `predicted_combo_auc` | **连续 AUC 预测**（越低 = 细胞杀伤越强，0-300 scale）|
| `single_auc_d1`, `single_auc_d2` | 两药单独 AUC |
| `mech_score` | 机制先验打分（0-2 范围）|
| `clonal_coverage_score` | Layer 2 克隆覆盖率（0-1）|
| `both_mech_annotated` | 两药是否都在机制 vocab 中 |
| `layer3_backbone` | 用的是哪个 backbone（默认 `BaselineA-MLP + mech-prior`）|

**读法：**
- `predicted_combo_auc` 比 `0.5*(single_auc_d1 + single_auc_d2)` 低 → 组合优于加性
- `mech_score > 0.5` + `clonal_coverage_score > 0.5` → **三层都认可的高置信度推荐**
- `both_mech_annotated = False` → 其中有药没机制注释，结果看 MLP 本身

### 2.9 `top_regimens` — Layer 1 试验证据推荐方案

每行一个**已发表临床试验对应的方案**：
| 字段 | 含义 |
|---|---|
| `rank` | 1..K |
| `name` | 方案全名 |
| `drugs` | 药物清单（如 `["Azacytidine", "Venetoclax", "Gilteritinib"]`）|
| `published_cr_cri_rate` | 原试验的 CR/CRi 率 |
| `published_median_os_months` | 原试验的中位 OS（月）|
| `trial_name` | 试验名（如 VIALE-A、RATIFY、ASH 2024）|
| `trial_phase` | Phase 1/2/3/FDA |
| `pmid` | PubMed ID，直接查引用 |
| `nct_id` | ClinicalTrials.gov 注册号 |
| `cautions` | 方案特异性警告（如 Quiz 的 QT 延长）|
| `biomarker_matches` | 匹配了哪些驱动（如 `["FLT3-mut", "fit_for_intensive"]`）|

**这一层是可以直接援引文献给临床审查的**。例如：推荐 AZA+Ven+Gilt → `pmid: 38277619` → Short/Daver et al. JCO 2024 → CR/CRi 96%。

### 2.10 `clonal_coverage` — Layer 2 克隆覆盖

| 字段 | 含义 |
|---|---|
| `patient_clones` | `{clone_name: weight}` dict — 患者活跃克隆结构 |
| `n_clones_present` | 激活克隆数 |
| `dominant_clones` | top-3 主导克隆 |
| `top_doublets_by_coverage` | 按覆盖率排序的 top 5 双联 |
| `top_triplets_by_coverage` | 按覆盖率排序的 top 5 三联 |

**读法：** 一个 FLT3+NPM1+BCL2 激活的患者，`top_triplets_by_coverage` 里排名靠前的三联会同时覆盖这 3 个轴。这是**生物学 rationale** 层，如果 MLP 推荐的组合没对应的克隆覆盖分 ≥ 0.5，就说明 MLP 没找到生物学支点。

---

## 三、临床 Q&A（13 条，按严重度排序）

### 🔴 严重级（必答）

#### Q1. 为什么只 25 个基因？现在 AML panel 都 70-90 个基因
**A**: 这是**核心 panel，不是 comprehensive panel**。25 个基因覆盖 ELN 2017 + WHO 2022 要求的**必需 minimum**（FLT3、NPM1、IDH1/2、TP53、RUNX1、ASXL1、CEBPA、TET2、DNMT3A、KIT、RAS-MAPK group、剪接因子、cohesin、KMT2A、MECOM、CBFB）。任何额外基因可以**在 CSV 后追加行**，表结构支持。
如果你们 lab 有 70-gene panel，我们的 roadmap 里会加 "extended panel" 支持，但需要重新评估哪些基因有 actionable implications。

#### Q2. 你用 ELN 2017 但 2022 已经出了
**A**: 训练队列（BeatAML 2.0）样本时间 2014-2019，标签基于 ELN 2017。**切换到 2022 需要重新标注队列**——已纳入 roadmap。
关键区别：
- ELN 2017: FLT3-ITD with high AR (no NPM1) = Adverse
- ELN 2022: FLT3-ITD **没有 AR 分层**了，single NPM1+FLT3-ITD = Intermediate
- ELN 2022 新增 adverse: BCOR, EZH2, SF3B1, SRSF2, STAG2, U2AF1, ZRSR2 (whole MDS-related group)

下一版会加 `eln2022` 辅助列。

#### Q3. TP53 allelic state 你没分
**A**: 正确，目前 kit 不区分 TP53 monoallelic vs multi-hit（双等位缺失 / 复合杂合 / 复杂核型 17p13）。这需要 **CNV / LOH 数据**，`KitInput.MutationCall` 里当前没这个字段。
Roadmap: 加 `is_multihit: bool` 字段到 `MutationCall`。WHO 2022 明确 multi-hit TP53 是独立 adverse 类别，比 monoallelic 更严重。

#### Q4. 核型解析这么粗糙
**A**: 是 regex 启发式，覆盖约 85% 常见 ISCN 模式。**不替代 cytogeneticist 的专业分析**。
覆盖的模式：`t(X;Y)`、`del(X)`、`inv(X)`、`add(X)`、`der(X)`、`+N`、`-N`、`i(Xq)`、`idic(X)`、`dic(X)`、`r(X)`、`mar`、`hsr`。
不覆盖或处理粗糙的：罕见平衡易位、子克隆比例、cryptic rearrangement、chromothripsis。
**临床使用建议**：cytogenetics.csv 表是 screening view，不是 definitive interpretation。对任何 ✓ 行，医生应回头看 lab 的原始核型字符串 + FISH / SKY 结果。

### 🟡 中度（可以回得上）

#### Q5. VAF 没有 lower threshold
**A**: Kit 不 filter VAF，直接报实验室给的值。**过滤由 lab 决定**（常见阈值 1%、5%、10% 取决于 panel depth 和 variant caller 设置）。
`sample_qc.csv` 里的 `n_above_vaf_0_20` 是 kit 给的**次级指标**（VAF ≥ 0.20 的数量），供审阅时过滤亚克隆噪声。

#### Q6. CEBPA biallelic 你信患者 `is_biallelic` 这个 bool？万一 lab 没检呢？
**A**: `is_biallelic` 是 `KitInput.MutationCall` 字段，**由上游调用者传入**，kit 不自动推断。如果 lab 没专门检测，传 False → kit 就按 monoallelic 处理，**不会假阳性把 monoallelic 报成 favorable**。
**建议 workflow**: NGS 报告如果没明确 annotate biallelic，先默认 False。CEBPA biallelic 需要 bZIP domain 分析或专门的 long-read 验证。

#### Q7. DNMT3A R882H 是特殊位点，你没区分
**A**: 目前按基因级别处理，不区分 R882 vs non-R882。
**文献立场**: R882 vs non-R882 在某些亚组预后差异有报道（R882 稍差），但**当前 NCCN / ELN guideline 没要求分**。
Roadmap: 加 `variant_hotspot` 字段（R882, DNMT3A-R882 等），但作 Tier-3 优化，非高优先级。

#### Q8. 没有 MRD 追踪
**A**: 本 kit 是**一次性分型工具**，定位在诊断 / 初治决策点。**不做 MRD 追踪**（那是 flow MRD / PCR MRD / NGS-MRD 另一套工具，需要时序样本）。
如需 MRD panel，NPM1-mut / FLT3-ITD / RUNX1-RUNX1T1 的 qPCR 跟踪是主流，由血液科 + 分子病理协作。

### 🟢 轻度（容易 handle）

#### Q9. PML-RARA 不查 BCR 断裂点（bcr1/bcr2/bcr3）
**A**: 当前把 PML-RARA 作为单一 entity 处理。Roadmap: 在 `fusion` 字段里加 `breakpoint` sub-field，标注 bcr1/bcr2/bcr3。bcr3（short form, ~30%）预后略差但治疗不变。

#### Q10. FLT3 ITD length 不报
**A**: 需要原始 NGS reads 的 soft-clip pattern 分析，kit 当前只接受 summary stats（ITD yes/no + allelic ratio）。**FLT3-ITD length 对治疗选择没影响**（≥70 aa inserts 可能稍 adverse，但 guideline 没分）。

#### Q11. 没 ClinVar / OncoKB crosslink
**A**: Roadmap。目前 `notes` 字段给了临床注解，但没活跃 URL。加 `oncokb_level`（1A/1B/2A/2B/3）和 `clinvar_id` 字段是明确改进方向。

#### Q12. KMT2A-rearranged 是否 t(9;11) 要分
**A**: `fusion_analysis.csv` 的 `notes` 已提到 "t(9;11) 是 Intermediate，其他 KMT2A-r 是 Adverse" 但不够突出。
Roadmap: 把这条分级直接反映到 `eln_implication` 字段（现在全统 "Adverse (historically)"）。

#### Q13. 表头没有 SCOPE & LIMITATIONS 声明
**A**: 应当加在 CSV 首行作注释行：
```csv
# AML Combo-Prediction Kit v0.2 — research use only
# Panel: 25 core driver genes (ELN 2017 minimum)
# NOT comprehensive — labs' additional genes need manual integration
# ELN version: 2017 (2022 upgrade on roadmap)
# Sources: ELN 2017, WHO 2022, NCCN v2.2024, FDA drug labels
gene,variant_type,...
```
Roadmap item。

---

## 四、已知限制与改进路线图

### 短期（2-4 周可做）
1. CSV 加 SCOPE & LIMITATIONS 声明头
2. MutationCall 加 `is_multihit` 字段（TP53 区分）
3. Cytogenetics 加 `confidence` 字段（regex vs 人工验证）
4. 表头加 OncoKB level 列

### 中期（1-3 月）
5. ELN 2022 支持（新增 `eln2022_risk` 列）
6. FLT3 variant_hotspot（D835Y / F691L 等）
7. 扩展 core panel 到 40-50 gene（NCCN v2.2024 推荐）
8. 融合分析 breakpoint 细化（PML-RARA bcr1/2/3）

### 长期（6 月+）
9. 前瞻性 ex-vivo 验证数据融入（从 HL-60 → real patient samples）
10. Decision support explanation engine（给每个推荐生成自然语言理由）
11. EHR 集成（HL7 FHIR genomics profile 输出）
12. NMPA 创新医疗器械申报资料

---

## 五、如何与实验室工作流集成

### 典型使用流程
```
1. NGS 实验室完成变异 calling → 输出 VCF + 实验室报告
2. 细胞遗传学完成核型 → 输出 ISCN 字符串 + 融合 call
3. 数据录入员填写 KitInput：
   - mutations: 从 VCF 抽取 gene + variant_type + VAF
   - karyotype_text: 复制粘贴 ISCN
   - fusions: 从细胞遗传学报告提取
   - 临床：从 CBC / 生化 / 病史填充
4. 上传 RNA-Seq counts TSV（gene_symbol → count）
5. 跑 kit → 生成 runs/dna_reports/patient_<id>/*
6. 多学科讨论（MDT）时：
   - 血液科主治：先看 dna_profile.png 快速概览
   - 分子病理：逐行核对 driver_mutations.csv
   - 细胞遗传学：核对 cytogenetics.csv + fusion_analysis.csv
   - 药师：看 targetability.csv + top_regimens 交叉
   - 信息科：留存 dna_summary.json 作审计
7. MDT 决议 → 医生给最终方案（kit 输出是辅助不是 binding）
```

### 跟 NGS lab 的 QC 对接
- Lab 端要 export：
  - Panel 名 + 版本（如 Illumina TruSight Myeloid v2）
  - 变异 VCF
  - Variant allele frequency 和 read depth
  - ITD-specific caller 结果（FLT3-ITD 独立检测）
  - CNV / LOH calls（将来加）

- 传入 kit 前的 QC pass 标准：
  - Mean coverage ≥ 500x on driver hotspots
  - ITD assay 敏感度 ≥ 1% VAF
  - Karyotype 至少 20 个分裂象分析

---

## 六、参考文献（每个 claim 的出处）

### 指南 / guideline
- ELN 2017: Döhner H et al. *Blood* 2017 Jan 26;129(4):424-447. PMID: **27895058**
- ELN 2022: Döhner H et al. *Blood* 2022 Sep 22;140(12):1345-1377. PMID: **35797463**
- WHO 2022: Khoury JD et al. *Leukemia* 2022 Jul;36(7):1703-1719. PMID: **35732831**
- ICC 2022: Arber DA et al. *Blood* 2022 Sep 15;140(11):1200-1228. PMID: **35767897**
- NCCN AML v2.2024: NCCN Clinical Practice Guidelines in Oncology

### 关键临床试验（kit 的 regimen DB 来源）
- RATIFY (Mid+7+3): Stone RM et al. *NEJM* 2017 Aug 3;377(5):454-464. PMID: **28644114**
- QUANTUM-First (Quiz+7+3): Erba HP et al. *Lancet* 2023 Jun 3;401(10388):1971-1980. PMID: **37116523**
- VIALE-A (Ven+Aza): DiNardo CD et al. *NEJM* 2020 Aug 13;383(7):617-629. PMID: **32813947**
- VIALE-C (Ven+LDAC): Wei AH et al. *Blood* 2020 Jun 11;135(24):2137-2145. PMID: **32173999**
- ADMIRAL (Gilteritinib mono R/R): Perl AE et al. *NEJM* 2019 Oct 31;381(18):1728-1740. PMID: **31665578**
- AGILE (Aza+IVO IDH1-mut): Montesinos P et al. *NEJM* 2022 Apr 21;386(16):1519-1531. PMID: **35443106**
- AZA+Ven+Gilt triplet: Short NJ, Daver N et al. *J Clin Oncol* 2024 Apr 20;42(12):1499-1508. PMID: **38277619**
- Quiz+Ven+Dec triplet: ASH 2024 abstract (pending full publication)
- Venetoclax + Decitabine (10-day): DiNardo CD et al. *Lancet Haematol* 2019 Nov;6(10):e540-e551. PMID: **30573634**

### 靶向药 FDA 标签
- Midostaurin (Rydapt): FDA label 2017
- Gilteritinib (Xospata): FDA label 2018
- Quizartinib (Vanflyta): FDA label 2023
- Ivosidenib (Tibsovo): FDA label 2018
- Enasidenib (Idhifa): FDA label 2017
- Venetoclax (Venclexta) for AML: FDA label 2018
- Azacitidine (Vidaza): FDA label 2004
- Gemtuzumab ozogamicin (Mylotarg): FDA re-label 2017
- Revumenib (Revuforj): FDA label 2024 (KMT2A-r)

### 方法学参考
- Palmer AC, Sorger PK. Independent drug action. *Cancer Discov* 2022 Mar;12(3):606-621. PMID: **34983746**
- Julkunen H et al. comboFM. *Nat Commun* 2020;11:6136. PMID: **33262326**
- DrugComb v1.5: Zenodo record 15235991 (2024)

---

## 七、联系方式

如发现 kit 报告与临床实际不符，请记录：
1. `patient_id`
2. 具体不符字段和 lab 对应值
3. 期望的 correct value
4. Kit version (见 `dna_summary.json` 的 `version` 字段)

反馈给项目维护者（`ericktom94720@gmail.com`）。每条反馈都会进 roadmap 考虑。

---

*Generated by AML Combo-Prediction Kit v0.2 · For research use only · Not for clinical diagnosis*
