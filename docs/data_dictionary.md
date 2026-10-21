# ASGP Data Dictionary & Schema Documentation

This document describes all database entities, CSV data schemas, feature variables, and taxonomy structures used across the ASGP platform.

---

## 1. Entities & Relational Tables

### 1.1 `occupations` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `occupation_id` | String(50) | NO (PK) | Unique canonical identifier for occupation | `OCC-001` |
| `title` | String(200) | NO | Standard occupational job title | `Data Scientist` |
| `description` | Text | YES | Summary of role responsibilities | `Designs and trains predictive models...` |
| `primary_cluster_id` | Integer | NO | Index of primary functional skill cluster | `2` (0 to 7) |
| `primary_cluster_name` | String(100) | YES | Human-readable title of primary cluster | `Machine Learning & AI` |
| `secondary_cluster_id` | Integer | NO | Index of secondary functional skill cluster | `5` (0 to 7) |
| `secondary_cluster_name`| String(100) | YES | Title of secondary cluster | `Data Engineering & Databases` |
| `source` | String(50) | NO | Provenance marker | `synthetic_baseline`, `ESCO` |

---

### 1.2 `skills` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `skill_id` | String(50) | NO (PK) | Canonical skill identifier | `SKL-042` |
| `name` | String(200) | NO | Preferred competency or skill name | `PyTorch Deep Learning` |
| `skill_type` | String(50) | NO | Taxonomy classification | `skill/competence`, `knowledge` |
| `cluster_id` | Integer | NO | Latent functional cluster index | `2` (0 to 7) |
| `cluster_name` | String(100) | YES | Title of assigned functional cluster | `Machine Learning & AI` |
| `is_generic` | Boolean | NO | Indicator if skill is cross-occupational | `True`, `False` |
| `complexity` | Float | NO | Estimated conceptual complexity | `0.30` to `0.95` |
| `source` | String(50) | NO | Origin of skill entity | `synthetic_baseline`, `ESCO` |

---

### 1.3 `occupation_skill_edges` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `occupation_id` | String(50) | NO (FK) | Identifier of parent occupation | `OCC-001` |
| `skill_id` | String(50) | NO (FK) | Identifier of required skill | `SKL-002` |
| `relation_type` | String(50) | NO | Importance category | `essential`, `optional` |
| `relevance_weight`| Float | NO | Quantitative importance weight $w_{r,s}$ | `0.40` to `0.98` |
| `tier` | String(50) | YES | Domain hierarchy level | `primary_domain`, `cross_cutting` |
| `source` | String(50) | NO | Origin of edge | `synthetic_baseline`, `ESCO` |

---

### 1.4 `employees` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `employee_id` | String(50) | NO (PK) | Unique employee identifier | `EMP-0012` |
| `name` | String(150) | NO | Full name of employee | `Jordan Chen` |
| `email` | String(150) | NO (Unique)| Internal company email address | `emp-0012@corp.asgp.internal` |
| `occupation_id` | String(50) | NO (FK) | Assigned target occupational role | `OCC-002` |
| `department` | String(100) | NO | Organizational business unit | `Engineering & Technology` |
| `hired_date` | String(50) | NO | Date of recruitment onboarding | `2026-01-15` |

---

### 1.5 `employee_skills` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `employee_id` | String(50) | NO (FK) | Target employee | `EMP-0012` |
| `skill_id` | String(50) | NO (FK) | Evaluated skill entity | `SKL-011` |
| `state` | String(30) | NO | Current assessment mastery status | `not_assessed`, `not_mastered`, `in_progress`, `mastered` |
| `assessment_score`| Float | YES | Diagnostic test or exam score | `0.00` to `1.00` (or `None` if unassessed) |
| `updated_at` | DateTime | NO | Timestamp of latest state change | ISO 8601 string |

---

### 1.6 `training_modules` (Table & CSV)
| Column Name | Data Type | Nullable | Description | Example / Range |
|---|---|---|---|---|
| `module_id` | String(50) | NO (PK) | Identifier of instructional unit | `MOD-005` |
| `target_skill_id` | String(50) | NO (FK) | Skill addressed by curriculum | `SKL-005` |
| `title` | String(200) | NO | Course or curriculum title | `Applied Training: Feature Engineering` |
| `duration_hours`| Integer | NO | Expected completion duration | `4` to `24` hours |
| `difficulty_level`| String(30) | NO | Pedagogical complexity tier | `beginner`, `intermediate`, `advanced` |
| `learning_cost` | Float | NO | Simulated cognitive load unit | `10.0` to `35.0` |
| `assessment_pass_threshold` | Float | NO | Score required to achieve mastery | `0.75` |

---

## 2. Machine Learning Feature Matrix Schema

Extracted by `SkillGapAnalyzer.generate_dataset_gap_features()`:

| Feature Name | Data Type | Range | Mathematical Symbol | Description |
|---|---|---|---|---|
| `relevance_weight` | Float | $[0.40, 0.98]$ | $w_{r,s}$ | Direct occupation-skill importance weight. |
| `is_generic` | Integer | $\{0, 1\}$ | $g_s$ | Binary indicator of cross-occupational competency. |
| `possessed_skills_count` | Integer | $[0, 25]$ | $|P_e|$ | Total count of mastered skills currently held by employee. |
| `cluster_coverage` | Float | $[0.00, 1.00]$ | $\text{Cov}(e, c_s)$ | Ratio of required cluster competencies already mastered by employee. |
| `is_critical` (Label) | Integer | $\{0, 1\}$ | $y$ | Binary ground truth label (Phase 6: Top-1/3 relevance; Phase 7: Downstream readiness gain). |
