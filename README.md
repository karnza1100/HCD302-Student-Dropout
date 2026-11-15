# HCD302-Student-Dropout

# 🎓 Student Dropout Early Warning System (Model Card & Explainable AI)

**รายวิชา:** HCD-302 Creating Explainable AI  
**กลุ่มผู้จัดทำ:** กลุ่ม OK67 (วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยสงขลานครินทร์ วิทยาเขตภูเก็ต)

---

## 👥 สมาชิกในกลุ่ม (Group OK67)

| ลำดับ | ชื่อ - นามสกุล | รหัสนักศึกษา |
| :---: | :--- | :---: |
| 1 | นาย กษิ ศรีสุวรรณ | `6730614006` |
| 2 | นาย ศุภสิณ จรัสอัครมนตรี | `6730614008` |
| 3 | นาย นนธวัช สำลี | `6730614035` |
| 4 | นาย เดชพล จิมะ | `6730614034` |
| 5 | นาย ณัฐศิลป์ โชติประดิษฐ์ | `6730614041` |

---

## 📌 ภาพรวมโครงการ (Project Overview)

โครงการนี้เป็นการพัฒนาระบบเตือนภัยล่วงหน้าสำหรับคาดการณ์และอธิบายความเสี่ยงการออกกลางคันของนักศึกษา (**Student Dropout Prediction**) พร้อมจัดทำเอกสาร **Model Card** ตามกรอบแนวคิดของ *Mitchell et al. (2019)* เพื่อความโปร่งใส ตรวจสอบได้ และคำนึงถึงความยุติธรรม (Fairness) ก่อนนำไปใช้งานจริง

* **โมเดลหลักที่เลือกใช้:** `LightGBM Classifier` (v1.0) ร่วมกับ `SMOTE` Pipeline (ป้องกัน Data Leakage)
* **Decision Threshold ที่เลือกใช้:** `0.35` (ปรับลดจาก `0.50` เพื่อเพิ่มค่า Recall และลดความเสียหายจาก False Negative)
* **ประสิทธิภาพบน Test Set ($n = 885$):**
  * **ROC-AUC:** `0.9309` (95% CI: `0.9112 – 0.9495`)
  * **Recall (Sensitivity):** `84.86%`
  * **Precision:** `77.49%`
  * **F1-Score:** `0.8101`
* **เครื่องมืออธิบายโมเดล (XAI):**
  * **Global Explanation:** Permutation Feature Importance (PFI) และ Partial Dependence Plot (PDP)
  * **Local Explanation:** TreeSHAP (Beeswarm & Waterfall Plots) และ LIME (ตรวจสอบ Fidelity & Stability)

> ⚠️ **ข้อจำกัดสำคัญและขอบเขตการใช้งาน:** โมเดลนี้ถูกออกแบบมาเพื่อเป็น **"ระบบคัดกรองและเตือนภัยล่วงหน้า"** สำหรับอาจารย์ที่ปรึกษาใช้ประกอบการพูดคุยให้คำปรึกษาเชิงรุกเท่านั้น **ห้าม** นำผลทำนายไปใช้ตัดสิทธิ์ทุนการศึกษา ลงโทษทางวินัย หรือตัดสินพ้นสภาพนักศึกษาโดยอัตโนมัติ (ต้องมีมนุษย์ร่วมพิจารณาเสมอ)

---

## 📂 โครงสร้างไฟล์ใน Repository (Repository Structure)

| ชื่อไฟล์ | คำอธิบาย |
| :--- | :--- |
| **`HCD302_A1_OK67.pdf`** | เอกสาร Proposal โครงการ (ชิ้นงานที่ 1)[cite: 41] |
| **`HCD302_OK67_ModelCard.pdf`** | เอกสาร Model Card ฉบับสมบูรณ์ 9 หัวข้อ (ชิ้นงานที่ 2)[cite: 41] |
| **`HCD302_A1_OK67.ipynb`** | สมุดบันทึก (Google Colab Notebook) โค้ดการทดลอง เทรนโมเดล ประเมิน Fairness และสร้างกราฟทั้งหมด[cite: 41] |
| **`supplementary_checks.py`** | สคริปต์ Python สำหรับคำนวณค่าสถิติเชิงลึกเพิ่มเติม (95% CI, Subgroup AUC, PFI ตัวแปรอ่อนไหว และ Threshold Sweep กลุ่มทุน)[cite: 41] |
| **`best_lgbm_model.joblib`** | ไฟล์โมเดล `LightGBM Pipeline` ที่ผ่านการฝึกสอนแล้ว (`random_state=42`)[cite: 41] |
| **`lgbm_trees_full.txt`** | โครงสร้างต้นไม้ตัดสินใจทั้งหมด 200 ต้นของโมเดล LightGBM ในรูปแบบข้อความ[cite: 41] |
| **`raw_dataset.csv`** | ชุดข้อมูลดิบ *Predict students' dropout and academic success* จาก UCI Repository (ID: 697, CC BY 4.0)[cite: 41] |
| **`data_dictionary.csv`** | พจนานุกรมข้อมูล อธิบายชื่อคอลัมน์ ชนิดข้อมูล และตัวอย่างค่าของทั้ง 36 ฟีเจอร์[cite: 41] |
| **`train_split_seed42.csv`** | ชุดข้อมูลสำหรับฝึกสอน (80%, $n = 3,539$) แบ่งแบบ Stratified Split[cite: 41] |
| **`test_split_seed42.csv`** | ชุดข้อมูลสำหรับทดสอบ (20%, $n = 885$) แบ่งแบบ Stratified Split[cite: 41] |
| **`fig1_pfi_global.png`** | กราฟแสดงความสำคัญของตัวแปรภาพรวม Top 10 (Permutation Feature Importance)[cite: 41] |
| **`fig2_pdp_global.png`** | กราฟแสดงทิศทางความสัมพันธ์เฉลี่ยของตัวแปรสำคัญสูงสุด 3 อันดับแรก (Partial Dependence Plot)[cite: 41] |
| **`fig3_shap_beeswarm.png`** | กราฟ SHAP Summary (Beeswarm Plot) แสดงผลกระทบของตัวแปรภาพรวมทั้ง Test Set[cite: 41] |
| **`fig4_shap_case1_high.png`** | กราฟ SHAP Waterfall อธิบายกรณีศึกษานักศึกษากลุ่มเสี่ยงสูง (`Prob = 96.97%`)[cite: 41] |
| **`fig5_shap_case2_low.png`** | กราฟ SHAP Waterfall อธิบายกรณีศึกษานักศึกษากลุ่มเสี่ยงต่ำ (`Prob = 7.96%`)[cite: 41] |

---

## 🚀 วิธีรันตรวจสอบผลลัพธ์ซ้ำ (Reproducibility)

1. **ติดตั้งไลบรารีที่จำเป็น** (ทดสอบบนสภาพแวดล้อม `LightGBM 4.6.0`, `scikit-learn 1.6.1`, `imbalanced-learn 0.14.2`, `SHAP 0.54.1`):
   ```bash
   pip install ucimlrepo shap lime lightgbm imbalanced-learn scikit-learn pandas numpy matplotlib seaborn joblib

   รันสคริปต์ตรวจสอบค่าสถิติเพิ่มเติม:
   python supplementary_checks.py
