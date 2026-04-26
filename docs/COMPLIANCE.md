# COMPLIANCE.md — Regulatory framework analysis

This document is the regulatory exposure analysis for amlcombo.org under
the three frameworks most relevant to AML clinical-research users:

- **HIPAA** (US, 45 CFR 160 / 164)
- **GDPR** (EU 2016/679)
- **个人信息保护法 / PIPL** (China, 2021)

The analysis is **not legal advice**. Institutional reviewers should
consult their compliance officer before deploying the kit on real patient
data at their center. This file documents the platform's design choices
and what they imply for each regulation.

---

## 1. Platform classification

| Attribute | Value |
|-----------|-------|
| Service type | Software-as-a-Service, web-based clinical-research tool |
| Regulatory class (FDA) | **NOT FDA-cleared SaMD**. Marketed as "research-only, not for clinical diagnosis." |
| Regulatory class (NMPA China) | **NOT NMPA 三类 medical device**. Same research-only positioning. |
| CE marking | None |
| Intended user | Hematology researcher / clinician (decision-supporting reference, not autonomous decision-maker) |
| Patient data role | Processor (the user's institution is the controller / 个人信息处理者) |

Per the **research-only** positioning, the platform does not require
FDA / NMPA / CE pre-market approval. However, this status is **only
sustainable if the platform stays out of the autonomous-decision loop**.
If a user's workflow places the kit's output between the medical record
and the prescribing decision without independent clinician review, the
classification becomes more aggressive (potentially SaMD class IIa under
EU MDR or 二类 under NMPA AI 医疗器械 guidance).

---

## 2. HIPAA exposure (US users)

### 2.1 Are we a HIPAA Covered Entity?

**No.** We are not a healthcare provider, health plan, or healthcare
clearinghouse. We do not receive PHI as part of payment / treatment /
operations of a Covered Entity.

### 2.2 Are we a HIPAA Business Associate?

**Conditional.** If a US Covered Entity (a hospital) directs its workforce
to upload patient data to amlcombo.org as part of treatment or
operations, the platform would be acting as a Business Associate and
would require a Business Associate Agreement (BAA) under 45 CFR 164.504(e).

**Current platform status**: We do NOT offer a BAA. Therefore:

- US hospital users **must NOT upload PHI** to the platform.
- The Terms of Service explicitly prohibit PHI upload (§3 of Terms).
- The PHI detector (`amlcombo_web/app/phi_detector.py`) is a technical
  enforcement layer — not perfect, but a meaningful second line.

### 2.3 If a user accidentally uploads PHI?

The PHI detector catches the most common patterns (MRN, name, DOB,
national ID, address) and:

- Default `phi_policy="redact"`: text reaching the LLM is PHI-clean
- The patient submission record is purged on user deletion (cascade)
- The audit log records *that* PHI was caught, not the PHI itself

**This does not retroactively make the platform HIPAA-compliant** for
that submission. We would notify the user immediately and recommend
they delete the submission and re-submit a de-identified version.

### 2.4 If a US researcher submits de-identified data?

Per 45 CFR 164.514(b)(1) Safe Harbor, if all 18 identifiers are removed,
the data is no longer PHI. The platform is designed for this use case.
Researchers should:

1. Use only research IDs (e.g., `A-2026-001`), not MRN / name / DOB
2. Round ages > 89 to "90+"
3. Remove dates more granular than year (treatment "2024-Q1" instead of
   "2024-03-15") OR confirm dates fall within HIPAA Safe Harbor allowances
4. Strip all geographic detail more granular than state

The PHI detector enforces ~80% of these mechanically. The user must
verify the rest.

---

## 3. GDPR exposure (EU users)

### 3.1 Personal data and special categories

Patient health data is **Special Category Data** under GDPR Art. 9.
Processing requires a legal basis under Art. 9(2) — for AML clinical
research, typically:

- **Art. 9(2)(a)**: explicit consent of the data subject, OR
- **Art. 9(2)(j)**: scientific research with appropriate safeguards
  + EU/MS law authorization

The platform's design assumes the user (researcher's institution) has
established the appropriate Art. 9(2) basis for collecting the patient
data BEFORE uploading. We are a processor under Art. 28; the user is
controller.

### 3.2 Data Processing Agreement (DPA)

For EU institutional users, a Data Processing Agreement under Art. 28(3)
is available on request (`eric@amlcombo.org`). The DPA covers:

- Processor obligations (Art. 28(3)(a)-(h))
- Sub-processor approval (LLM providers OpenAI / Anthropic — see §3.4)
- International transfer mechanism (SCCs — see §3.5)
- Notification of breach within 24 hours (we exceed the 72-hour Art. 33
  requirement for controller notification)

### 3.3 Data minimization (Art. 5(1)(c))

The platform's input form explicitly asks ONLY for clinically necessary
fields (mutations, karyotype, lab values, age, sex, fitness). It does NOT
ask for or accept:

- Patient name
- Date of birth (only age / age band)
- Address, phone, email of the patient
- MRN or hospital identifier
- National ID

The PHI detector blocks accidental inclusion in free-text smart-paste.

### 3.4 Sub-processor: LLM providers

When the user enables Smart-Paste with their BYOK API key:

- **They** are sending the prompt to OpenAI / Anthropic via their account
- **We** are merely the conduit; we never retain their API key in plaintext
- Default `phi_policy="redact"` strips detectable PHI before forwarding
- The user's contract with OpenAI / Anthropic governs the LLM-side
  processing (their respective DPAs apply)

For EU institutional users wishing to disable LLM features entirely
(no third-country transfer), set `LLM_FEATURES_DISABLED=true` in the
deployment env vars. The platform still functions in full without LLM.

### 3.5 International transfer (Art. 44-49)

| Transfer | Mechanism |
|----------|-----------|
| User browser (EU) → amlcombo.org server (Hetzner DE = EU) | No transfer (intra-EU) |
| amlcombo.org server (Hetzner DE) → OpenAI/Anthropic (US) | Standard Contractual Clauses (SCCs) — OpenAI EU SCCs since 2022; Anthropic EU SCCs since 2023 |
| amlcombo.org server (Hetzner DE) → Cloudflare (global edge) | Cloudflare EU SCCs |

EU users can verify Hetzner Falkenstein (DE) location via a network
trace. We commit to keeping the primary database in EU jurisdiction.

### 3.6 Right to erasure (Art. 17)

Implemented via `DELETE /api/me/submissions/<id>` (single submission)
and `DELETE /api/me` (account + all submissions + LLM keys). After
account deletion, audit-log rows are anonymized (user_id NULLed) and
retained 2 years per the legitimate-interest basis (security audit).

---

## 4. PIPL exposure (中国大陆用户)

### 4.1 个人信息分类 — 患者健康信息属于敏感个人信息

患者健康信息按 PIPL 第二十八条属于**敏感个人信息**, 处理需要满足:

- 第二十九条: 取得**单独同意** (separate consent), 不能通过 ToS 笼统授权
- 第三十条: 处理活动应当有特定的目的 + 充分的必要性
- 平台设计假设上传方 (研究者所在机构) 已按上述要求获得了患者的单独同意。我们作为受托处理方按第二十一条履行义务。

### 4.2 数据出境 — 第三十八条 + 第三十九条

中国大陆用户启用 BYOK Smart-Paste 调用境外 LLM (OpenAI 美国 / Anthropic 美国) 时, 涉及个人信息出境, 需依据第三十八条选择以下任一路径:

| 路径 | 适用场景 | 平台支持 |
|------|---------|---------|
| (一) 通过国家网信部门**安全评估** | 关键信息基础设施运营者 / 处理 100 万人以上个人信息的处理者 | 平台没有此规模, 个人用户也不属于关基运营者 |
| (二) 经专业机构**个人信息保护认证** | 跨国/集团内传输 | 平台暂未取得 |
| (三) 与境外接收方订立**国家网信部门标准合同** | 一般情形 | 平台 ToS 已声明, 但用户应自行与 OpenAI/Anthropic 签署 SCCs |
| (四) 法律行政法规 / 网信部门规定的其他条件 | 例外 | 不适用 |

**实际操作建议** (中国大陆机构用户):

1. **强烈推荐** 关闭 Smart-Paste 功能 (设 `LLM_FEATURES_DISABLED=true`), 仅用平台手工录入 + 本地报告生成 — 全部流程在 EU 服务器内, 不出境到美国。
2. 如必须使用 Smart-Paste, 默认 `phi_policy="redact"` 已经把 PHI 在出境前剥除; 但仍建议提交合规审查。
3. **绝不可** 启用 `phi_policy="passthrough"` 除非有机构级合规授权。

### 4.3 数据本地化 — 第四十条

医疗机构存储的患者数据是否需要在境内存储, 视医院类型与监管要求 (国卫规划发 2018-25 号《国家健康医疗大数据标准、安全和服务管理办法》试行), 一般认为:

- 三甲医院的诊疗数据应在境内存储
- 学术研究队列在脱敏后跨境传输有一定空间
- 平台数据库目前在 Hetzner DE (EU 境内, 非中国境内) — 所以中国大陆三甲医院**不应将原始诊疗数据**上传, 只应上传脱敏后的研究数据。

### 4.4 患者权利 (第四十四条 - 第五十条)

- **知情权 + 决定权** (44 条): 用户机构对患者获得知情同意时应说明数据可能上传到境外研究平台
- **查询权 + 复制权** (45 条): 平台 `GET /api/me/export` 接口支持
- **删除权** (47 条): `DELETE /api/me` 实现删除并级联清除所有 submission

---

## 5. Synthesis — recommended deployment posture by region

### US (HIPAA-conscious institutional user)

- ✅ Use de-identified research IDs only
- ✅ Default `phi_policy="redact"`
- ❌ Do NOT upload data on behalf of a Covered Entity without a BAA (we don't offer one)
- ⚠ Verify with institutional IRB before clinical workflow integration

### EU (GDPR-conscious institutional user)

- ✅ Request DPA via `eric@amlcombo.org`
- ✅ Verify Hetzner DE location (intra-EU storage)
- ✅ Default `phi_policy="redact"` for any LLM features
- ⚠ If LLM-free deployment is preferred, set `LLM_FEATURES_DISABLED=true`

### China (PIPL-conscious institutional user)

- ✅ 优先使用平台手工录入 (无 LLM 出境)
- ✅ 如需 Smart-Paste, 保持默认 `phi_policy="redact"`
- ✅ 仅上传脱敏后的研究数据 (使用 `A-2026-001` 风格研究 ID)
- ❌ 切勿启用 `phi_policy="passthrough"` 处理真实临床数据
- ⚠ 三甲医院诊疗数据建议本地部署 (Docker self-host) 而非使用 amlcombo.org SaaS

---

## 6. Self-hosting option (recommended for compliance-sensitive deployments)

For institutions that cannot use the SaaS, the platform is self-hostable:

```bash
git clone https://github.com/Smugpigeon/AML-Combo-Validation.git
cd AML-Combo-Validation/amlcombo_web
cp .env.example .env  # set LLM_FEATURES_DISABLED=true if no LLM allowed
docker compose up -d
```

Self-hosted deployment puts ALL data in the institution's own
infrastructure. The platform retains no telemetry to amlcombo.org. This
is the recommended posture for any institution that wishes to deploy on
real (non-de-identified) patient data.

---

## 7. Open compliance items

These are known gaps not yet resolved:

| Gap | Status |
|-----|--------|
| Formal DPA template for EU users | Drafted, awaiting legal review |
| BAA template for US users | Not offered (intentional — research-only positioning) |
| 国家网信部门 standard contract for China users | Not signed (we recommend self-host instead) |
| Periodic third-party penetration test | Not scheduled |
| SOC 2 Type II audit | Not in scope (research tool) |

---

## 8. Contact

Compliance / legal questions: **eric@amlcombo.org** (Eric Tom, platform
operator).

This document is reviewed annually or upon material regulatory change.
Last review: 2026-04-26.
