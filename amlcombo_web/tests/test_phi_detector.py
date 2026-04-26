"""Tests for the PHI detector (post-v0.3 concern #4).

The detector must:
  1. Catch high-confidence PHI: MRN, names, DOB, national IDs, addresses
  2. NOT flag clinically-necessary content: lab values, drug doses,
     karyotypes, mutation VAFs, treatment dates
  3. NOT flag de-identified research IDs (A-2026-001 style)
  4. Produce a clean redacted string suitable for LLM dispatch
"""

from __future__ import annotations

import pytest

from amlcombo_web.app.phi_detector import (
    PHICategory,
    PHIDetected,
    assert_no_high_severity_phi,
    detect_phi,
)


# ---------------------------------------------------------------------------
# True positives — must catch these
# ---------------------------------------------------------------------------


def test_detects_mrn_chinese_keyword():
    text = "病历号: 1234567 入院后查血常规"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.MRN for m in r.matches)


def test_detects_mrn_english_keyword():
    text = "MRN: H123456 admitted on 2024-03-15"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.MRN for m in r.matches)


def test_detects_chinese_name():
    text = "患者姓名: 张三, 男, 45岁"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.NAME_CN for m in r.matches)


def test_detects_chinese_name_in_clinical_context():
    text = "姓名: 王小明 性别: 男 年龄: 45 核型: 46,XX[20]"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.NAME_CN for m in r.matches)


def test_detects_western_name_with_title():
    text = "Mr. John Smith, 58 years old, presenting with FLT3-ITD AML"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.NAME_WESTERN for m in r.matches)


def test_detects_dob():
    text = "DOB: 1980-05-12, presenting with AML"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.DOB for m in r.matches)


def test_detects_dob_chinese():
    text = "出生日期: 1980/05/12"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.DOB for m in r.matches)


def test_detects_china_national_id():
    text = "身份证号: 110101198001011234"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.NATIONAL_ID_CN for m in r.matches)


def test_detects_us_ssn():
    text = "SSN: 123-45-6789"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.SSN_US for m in r.matches)


def test_detects_email():
    text = "Contact: john.smith@hospital.org for follow-up"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.EMAIL for m in r.matches)


def test_detects_chinese_address():
    text = "住址: 北京市朝阳区建国路88号"
    r = detect_phi(text)
    assert r.has_phi
    assert any(m.category == PHICategory.ADDRESS for m in r.matches)


# ---------------------------------------------------------------------------
# True negatives — MUST NOT flag these clinically-necessary items
# ---------------------------------------------------------------------------


def test_does_not_flag_deidentified_research_id():
    text = "Patient: A-2026-001, age 45, FLT3-ITD"
    r = detect_phi(text)
    # MRN detector should skip A-2026-001 — it matches the de-identified
    # research ID pattern, even though the prefix says "Patient".
    mrn_matches = [m for m in r.matches if m.category == PHICategory.MRN]
    assert len(mrn_matches) == 0


def test_does_not_flag_lab_values():
    text = "WBC 95.0 x10^9/L, Platelet 32, Hgb 8.5, LDH 1240"
    r = detect_phi(text)
    assert not r.has_phi or all(m.severity != "high" for m in r.matches)


def test_does_not_flag_karyotype():
    text = "Karyotype: 46,XX,t(8;21)(q22;q22)[20]"
    r = detect_phi(text)
    assert not r.has_phi


def test_does_not_flag_mutation_vaf():
    text = "FLT3-ITD AR=0.62 VAF 45%, NPM1 missense VAF 42%, DNMT3A R882H VAF 48%"
    r = detect_phi(text)
    assert not r.has_phi


def test_does_not_flag_treatment_dates():
    text = "Induction started 2024-03-15, day-14 marrow on 2024-03-29"
    r = detect_phi(text)
    # Plain dates without DOB keyword should not be PHI
    assert not r.has_phi or all(m.category != PHICategory.DOB for m in r.matches)


def test_does_not_flag_chinese_drug_names():
    text = "维奈克拉 + 阿扎胞苷, 米托蒽醌, 阿糖胞苷"
    r = detect_phi(text)
    assert not r.has_phi


def test_does_not_flag_phone_lookalike_lab_value():
    """A 11-digit number near 'x10^9/L' is a lab value, not a phone."""
    text = "WBC 13800000000 x10^9/L (this number is silly but exercises the heuristic)"
    r = detect_phi(text)
    phone = [m for m in r.matches if m.category == PHICategory.PHONE_CN]
    assert len(phone) == 0


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_redacted_text_replaces_matches():
    text = "病历号: 1234567 患者姓名: 张三 化验 WBC 95"
    r = detect_phi(text)
    assert "[REDACTED-MRN]" in r.redacted_text
    assert "[REDACTED-NAME_CN]" in r.redacted_text
    # Lab value preserved
    assert "WBC 95" in r.redacted_text


def test_redacted_text_preserves_clinical_content():
    text = ("Mr. John Smith, MRN: H999999, DOB: 1968-04-12. "
            "Karyotype 46,XX[20]; FLT3-ITD AR=0.62; WBC 95.")
    r = detect_phi(text)
    # PHI redacted
    assert "John Smith" not in r.redacted_text
    assert "H999999" not in r.redacted_text
    assert "1968-04-12" not in r.redacted_text
    # Clinical content preserved
    assert "46,XX[20]" in r.redacted_text
    assert "FLT3-ITD" in r.redacted_text
    assert "AR=0.62" in r.redacted_text


# ---------------------------------------------------------------------------
# assert_no_high_severity_phi (gate use)
# ---------------------------------------------------------------------------


def test_assert_raises_on_high_severity():
    text = "病历号: 1234567 患者姓名: 张三"
    with pytest.raises(PHIDetected):
        assert_no_high_severity_phi(text)


def test_assert_passes_on_clean_text():
    text = "Karyotype 46,XX,t(8;21)[20]; FLT3-ITD AR=0.62 VAF 45%; WBC 95"
    r = assert_no_high_severity_phi(text)
    assert not r.has_phi


def test_empty_text_no_phi():
    r = detect_phi("")
    assert not r.has_phi
    assert r.high_severity_count == 0


# ---------------------------------------------------------------------------
# Realistic clinical paste (the test-suite A-2026-001 patient)
# ---------------------------------------------------------------------------


def test_realistic_test_suite_paste_no_phi():
    """The test-suite A-2026-001 row should be PHI-clean (de-identified)."""
    text = ("A-2026-001, 45, F, Initial diagnosis, No, 46\\,XX[20], None, "
            "ITD AR=0.62 VAF 45%, Frameshift VAF 42% W288Cfs*12, ..., "
            "WBC 95.0, PLT 32, Hgb 8.5, LDH 1240, BM blast 78%, PB blast 65%")
    r = detect_phi(text)
    # The research ID A-2026-001 should NOT be flagged as MRN
    assert not r.has_phi


def test_realistic_chinese_paste_with_phi():
    text = ("姓名: 王小明 性别: 男 年龄: 45 病历号: 2024030512 "
            "DOB: 1979-08-15 身份证: 110101197908151234 "
            "诊断: AML, FLT3-ITD AR=0.62, NPM1 mut, "
            "核型 46\\,XX[20], WBC 95")
    r = detect_phi(text)
    assert r.has_phi
    cats = r.categories_found()
    # Should catch at least 3 of 4 PHI categories present
    assert sum(c in cats for c in [
        PHICategory.NAME_CN.value, PHICategory.MRN.value,
        PHICategory.DOB.value, PHICategory.NATIONAL_ID_CN.value,
    ]) >= 3
    # Clinical content preserved in redacted
    assert "FLT3-ITD" in r.redacted_text
    assert "AR=0.62" in r.redacted_text
