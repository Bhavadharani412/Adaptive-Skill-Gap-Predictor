import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ESCOImporter:
    """
    Importer and preprocessor for ESCO (European Skills, Competences, Qualifications and Occupations)
    taxonomy and synthetic fallback datasets.
    
    Generates standardized CSVs and a data quality validation report.
    """

    def __init__(self, raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        from backend.app.core.config import settings
        self.raw_dir = raw_dir or settings.DATA_RAW_DIR
        self.processed_dir = processed_dir or settings.DATA_PROCESSED_DIR
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def find_esco_files(self, esco_dir: Path) -> Dict[str, Optional[Path]]:
        """Locates ESCO CSV files by matching common naming patterns."""
        files = {
            "occupations": None,
            "skills": None,
            "occupation_skill_relations": None,
            "skill_skill_relations": None
        }

        if not esco_dir.exists() or not esco_dir.is_dir():
            return files

        for f in esco_dir.glob("*.csv"):
            fname = f.name.lower()
            if "occupation" in fname and "skill" in fname and ("relation" in fname or "edge" in fname):
                files["occupation_skill_relations"] = f
            elif "skill" in fname and "relation" in fname:
                files["skill_skill_relations"] = f
            elif "occupation" in fname and "skill" not in fname:
                files["occupations"] = f
            elif "skill" in fname and "occupation" not in fname and "relation" not in fname:
                files["skills"] = f

        return files

    def run_import(self, esco_dir: Optional[Path] = None, force_fallback: bool = False) -> Dict:
        """
        Executes data ingestion pipeline.
        If real ESCO data is found, processes it; otherwise uses synthetic fallback.
        """
        esco_path = esco_dir or (self.raw_dir / "esco")
        files = self.find_esco_files(esco_path) if esco_path.exists() else {}

        has_esco = (
            not force_fallback
            and files.get("occupations") is not None
            and files.get("skills") is not None
            and files.get("occupation_skill_relations") is not None
        )

        if has_esco:
            logger.info("Processing ESCO dataset from %s", esco_path)
            return self._process_esco(files)
        else:
            logger.info("ESCO dataset not found or fallback requested. Generating synthetic fallback...")
            return self._generate_synthetic_fallback()

    def _resolve_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Resolves matching column name case-insensitively."""
        cols = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return None

    def _process_esco(self, files: Dict[str, Path]) -> Dict:
        """Parses, cleans and validates official ESCO CSV files."""
        report = {
            "source": "ESCO European Commission",
            "imported_at": datetime.utcnow().isoformat(),
            "status": "processing",
            "warnings": [],
            "errors": [],
            "metrics": {}
        }

        # 1. Occupations
        occ_file = files["occupations"]
        occ_df = pd.read_csv(occ_file)
        uri_col = self._resolve_column(occ_df, ["conceptUri", "uri", "occupation_uri", "id"])
        label_col = self._resolve_column(occ_df, ["preferredLabel", "title", "label", "occupation_title"])
        desc_col = self._resolve_column(occ_df, ["description", "definition"])

        if not uri_col or not label_col:
            err = f"Missing required columns in {occ_file.name}. Found: {list(occ_df.columns)}"
            report["errors"].append(err)
            raise ValueError(err)

        occ_df["occupation_id"] = occ_df[uri_col].astype(str)
        occ_df["title"] = occ_df[label_col].astype(str).str.strip()
        occ_df["description"] = occ_df[desc_col].astype(str).str.strip() if desc_col else ""
        occ_df["source_uri"] = occ_df[uri_col].astype(str)
        occ_df["source"] = "ESCO"

        initial_occ_count = len(occ_df)
        occ_clean = occ_df[["occupation_id", "title", "description", "source_uri", "source"]].drop_duplicates(subset=["occupation_id"])
        occ_dups = initial_occ_count - len(occ_clean)

        # 2. Skills
        skill_file = files["skills"]
        skill_df = pd.read_csv(skill_file)
        s_uri_col = self._resolve_column(skill_df, ["conceptUri", "uri", "skill_uri", "id"])
        s_label_col = self._resolve_column(skill_df, ["preferredLabel", "title", "label", "skill_name"])
        s_type_col = self._resolve_column(skill_df, ["skillType", "conceptType", "type"])

        if not s_uri_col or not s_label_col:
            err = f"Missing required columns in {skill_file.name}. Found: {list(skill_df.columns)}"
            report["errors"].append(err)
            raise ValueError(err)

        skill_df["skill_id"] = skill_df[s_uri_col].astype(str)
        skill_df["name"] = skill_df[s_label_col].astype(str).str.strip()
        skill_df["skill_type"] = skill_df[s_type_col].astype(str) if s_type_col else "skill/competence"
        skill_df["cluster_id"] = 0  # To be clustered or mapped
        skill_df["is_generic"] = False
        skill_df["source_uri"] = skill_df[s_uri_col].astype(str)
        skill_df["source"] = "ESCO"

        initial_skill_count = len(skill_df)
        skill_clean = skill_df[["skill_id", "name", "skill_type", "cluster_id", "is_generic", "source_uri", "source"]].drop_duplicates(subset=["skill_id"])
        skill_dups = initial_skill_count - len(skill_clean)

        # 3. Occupation-Skill Relations
        rel_file = files["occupation_skill_relations"]
        rel_df = pd.read_csv(rel_file)
        r_occ_col = self._resolve_column(rel_df, ["occupationUri", "occupation_uri", "occupation_id"])
        r_skill_col = self._resolve_column(rel_df, ["skillUri", "skill_uri", "skill_id"])
        r_type_col = self._resolve_column(rel_df, ["relationType", "relation_type", "type"])

        if not r_occ_col or not r_skill_col:
            err = f"Missing required columns in {rel_file.name}. Found: {list(rel_df.columns)}"
            report["errors"].append(err)
            raise ValueError(err)

        rel_df["occupation_id"] = rel_df[r_occ_col].astype(str)
        rel_df["skill_id"] = rel_df[r_skill_col].astype(str)
        rel_df["relation_type"] = rel_df[r_type_col].astype(str) if r_type_col else "essential"
        
        # Transparent heuristic for relevance weight based on relationType
        # Essential: 0.85 default, Optional: 0.50 default (explicitly documented)
        rel_df["relevance_weight"] = rel_df["relation_type"].apply(
            lambda t: 0.85 if "essential" in t.lower() else 0.50
        )
        rel_df["source"] = "ESCO"

        # Validate unresolved foreign keys
        valid_occ_ids = set(occ_clean["occupation_id"])
        valid_skill_ids = set(skill_clean["skill_id"])

        unresolved_occs = (~rel_df["occupation_id"].isin(valid_occ_ids)).sum()
        unresolved_skills = (~rel_df["skill_id"].isin(valid_skill_ids)).sum()

        if unresolved_occs > 0:
            report["warnings"].append(f"{unresolved_occs} relations reference unknown occupation URIs (filtered).")
        if unresolved_skills > 0:
            report["warnings"].append(f"{unresolved_skills} relations reference unknown skill URIs (filtered).")

        rel_clean = rel_df[
            rel_df["occupation_id"].isin(valid_occ_ids) & rel_df["skill_id"].isin(valid_skill_ids)
        ][["occupation_id", "skill_id", "relation_type", "relevance_weight", "source"]].drop_duplicates(
            subset=["occupation_id", "skill_id"]
        )

        # 4. Skill-Skill relations (if available)
        skill_rel_clean = pd.DataFrame(columns=["skill_id_1", "skill_id_2", "relation_type", "weight", "source"])
        if files.get("skill_skill_relations"):
            ssr_df = pd.read_csv(files["skill_skill_relations"])
            s1_col = self._resolve_column(ssr_df, ["skillUri1", "originalSkillUri", "skill_id_1", "source_uri"])
            s2_col = self._resolve_column(ssr_df, ["skillUri2", "relatedSkillUri", "skill_id_2", "target_uri"])
            ssr_type_col = self._resolve_column(ssr_df, ["relationType", "type"])
            if s1_col and s2_col:
                ssr_df["skill_id_1"] = ssr_df[s1_col].astype(str)
                ssr_df["skill_id_2"] = ssr_df[s2_col].astype(str)
                ssr_df["relation_type"] = ssr_df[ssr_type_col].astype(str) if ssr_type_col else "related"
                ssr_df["weight"] = 0.50
                ssr_df["source"] = "ESCO"
                skill_rel_clean = ssr_df[
                    ssr_df["skill_id_1"].isin(valid_skill_ids) & ssr_df["skill_id_2"].isin(valid_skill_ids)
                ][["skill_id_1", "skill_id_2", "relation_type", "weight", "source"]].drop_duplicates(
                    subset=["skill_id_1", "skill_id_2"]
                )

        # Save outputs
        occ_clean.to_csv(self.processed_dir / "occupations.csv", index=False)
        skill_clean.to_csv(self.processed_dir / "skills.csv", index=False)
        rel_clean.to_csv(self.processed_dir / "occupation_skill_edges.csv", index=False)
        skill_rel_clean.to_csv(self.processed_dir / "skill_relationships.csv", index=False)

        report["status"] = "completed"
        report["metrics"] = {
            "total_occupations": int(len(occ_clean)),
            "total_skills": int(len(skill_clean)),
            "total_occupation_skill_edges": int(len(rel_clean)),
            "total_skill_relationships": int(len(skill_rel_clean)),
            "duplicate_occupations_filtered": int(occ_dups),
            "duplicate_skills_filtered": int(skill_dups),
            "unresolved_occupation_references": int(unresolved_occs),
            "unresolved_skill_references": int(unresolved_skills),
            "essential_edges_count": int((rel_clean["relation_type"] == "essential").sum()),
            "optional_edges_count": int((rel_clean["relation_type"] != "essential").sum())
        }

        with open(self.processed_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_synthetic_fallback(self) -> Dict:
        """
        Generates a transparent, runnable synthetic fallback dataset representing
        10 key software engineering & data occupations, 60 skills across 6 clusters.
        Explicitly labeled as 'synthetic_fallback'.
        """
        np.random.seed(42)

        occupations_data = [
            ("OCC-001", "Data Scientist", "Applies machine learning, statistical modeling, and data analysis to extract business insights."),
            ("OCC-002", "Machine Learning Engineer", "Designs, builds, and deploys production machine learning systems and pipelines."),
            ("OCC-003", "Cloud Solutions Architect", "Architects scalable, secure, and resilient cloud infrastructures across hybrid environments."),
            ("OCC-004", "Backend Developer", "Develops robust server-side APIs, microservices, and database persistence architectures."),
            ("OCC-005", "Frontend Developer", "Builds responsive, accessible, and high-performance interactive user interfaces."),
            ("OCC-006", "DevOps & SRE Engineer", "Automates CI/CD pipelines, container orchestration, and system reliability monitoring."),
            ("OCC-007", "Cybersecurity Analyst", "Monitors threat telemetry, designs security controls, and performs vulnerability remediation."),
            ("OCC-008", "Data Engineer", "Constructs large-scale ETL/ELT data pipelines, warehouses, and streaming architectures."),
            ("OCC-009", "Product Manager (Technical)", "Defines technical roadmaps, stakeholder requirements, and product delivery metrics."),
            ("OCC-010", "Database Administrator", "Manages database clustering, query tuning, replication, backup, and high availability.")
        ]
        occ_df = pd.DataFrame(occupations_data, columns=["occupation_id", "title", "description"])
        occ_df["source_uri"] = "urn:synthetic:asgp:fallback:" + occ_df["occupation_id"]
        occ_df["source"] = "synthetic_fallback"

        clusters = {
            0: "Machine Learning & AI",
            1: "Cloud & Infrastructure",
            2: "Software Engineering & APIs",
            3: "Data Engineering & Analytics",
            4: "Security & Compliance",
            5: "Cross-Functional & Agile"
        }

        skills_raw = [
            # Cluster 0: ML & AI
            ("SKL-001", "Python Programming", 0, True),
            ("SKL-002", "Scikit-Learn & Classical ML", 0, False),
            ("SKL-003", "Deep Learning (PyTorch)", 0, False),
            ("SKL-004", "Feature Engineering", 0, False),
            ("SKL-005", "Model Evaluation & Metrics", 0, False),
            ("SKL-006", "Natural Language Processing", 0, False),
            ("SKL-007", "Computer Vision", 0, False),
            ("SKL-008", "MLOps & Model Registry", 0, False),
            ("SKL-009", "Hyperparameter Optimization", 0, False),
            ("SKL-010", "Statistical Hypothesis Testing", 0, False),

            # Cluster 1: Cloud & Infrastructure
            ("SKL-011", "Docker Containerization", 1, True),
            ("SKL-012", "Kubernetes Orchestration", 1, False),
            ("SKL-013", "AWS Cloud Services", 1, False),
            ("SKL-014", "Terraform Infrastructure as Code", 1, False),
            ("SKL-015", "Linux System Administration", 1, True),
            ("SKL-016", "Networking & DNS Basics", 1, False),
            ("SKL-017", "Serverless Architecture", 1, False),
            ("SKL-018", "Cloud Cost Optimization", 1, False),
            ("SKL-019", "Distributed Systems Design", 1, False),
            ("SKL-020", "Load Balancing & Auto-scaling", 1, False),

            # Cluster 2: Software Engineering & APIs
            ("SKL-021", "REST API Development", 2, True),
            ("SKL-022", "FastAPI Framework", 2, False),
            ("SKL-023", "TypeScript & JavaScript", 2, False),
            ("SKL-024", "React Web Development", 2, False),
            ("SKL-025", "Unit & Integration Testing", 2, True),
            ("SKL-026", "Object-Oriented Design", 2, False),
            ("SKL-027", "Microservice Patterns", 2, False),
            ("SKL-028", "Git Version Control", 2, True),
            ("SKL-029", "GraphQL API Design", 2, False),
            ("SKL-030", "Code Profiling & Benchmarking", 2, False),

            # Cluster 3: Data Engineering & Analytics
            ("SKL-031", "SQL Query Optimization", 3, True),
            ("SKL-032", "PostgreSQL Administration", 3, False),
            ("SKL-033", "Apache Spark Distributed Computing", 3, False),
            ("SKL-034", "ETL Pipeline Orchestration", 3, False),
            ("SKL-035", "Data Warehousing (Snowflake)", 3, False),
            ("SKL-036", "Apache Kafka Event Streaming", 3, False),
            ("SKL-037", "Data Quality Validation", 3, False),
            ("SKL-038", "Pandas Data Manipulation", 3, True),
            ("SKL-039", "NoSQL Document Databases", 3, False),
            ("SKL-040", "Business Intelligence Dashboards", 3, False),

            # Cluster 4: Security & Compliance
            ("SKL-041", "Application Security (OWASP Top 10)", 4, True),
            ("SKL-042", "OAuth2 & JWT Authentication", 4, False),
            ("SKL-043", "Vulnerability Scanning & SAST", 4, False),
            ("SKL-044", "Cryptographic Protocols & TLS", 4, False),
            ("SKL-045", "Identity & Access Management (IAM)", 4, False),
            ("SKL-046", "Incident Response & Forensics", 4, False),
            ("SKL-047", "Security Compliance (SOC2/GDPR)", 4, False),
            ("SKL-048", "Penetration Testing Principles", 4, False),
            ("SKL-049", "Secrets Management (Vault)", 4, False),
            ("SKL-050", "Zero-Trust Architecture", 4, False),

            # Cluster 5: Cross-Functional & Delivery
            ("SKL-051", "Agile & Scrum Delivery", 5, True),
            ("SKL-052", "Technical Documentation & RFCs", 5, True),
            ("SKL-053", "Continuous Integration (CI/CD)", 5, True),
            ("SKL-054", "Stakeholder Communication", 5, True),
            ("SKL-055", "Product Backlog Prioritization", 5, False),
            ("SKL-056", "System Architecture Diagrams", 5, False),
            ("SKL-057", "Observability & Prometheus/Grafana", 5, False),
            ("SKL-058", "A/B Testing & Experimentation", 5, False),
            ("SKL-059", "Root Cause Analysis (Post-mortems)", 5, False),
            ("SKL-060", "Mentoring & Technical Leadership", 5, False)
        ]

        skill_records = []
        for s_id, name, c_id, is_gen in skills_raw:
            skill_records.append({
                "skill_id": s_id,
                "name": name,
                "skill_type": "knowledge" if "Basics" in name or "Testing" in name else "skill/competence",
                "cluster_id": c_id,
                "cluster_name": clusters[c_id],
                "is_generic": is_gen,
                "source_uri": f"urn:synthetic:asgp:fallback:{s_id}",
                "source": "synthetic_fallback"
            })
        skill_df = pd.DataFrame(skill_records)

        # Build occupation-skill bipartite edges with realistic relevance weights
        # Data Scientist requirements
        occ_skill_map = {
            "OCC-001": [
                ("SKL-001", "essential", 0.95), ("SKL-002", "essential", 0.92),
                ("SKL-004", "essential", 0.90), ("SKL-005", "essential", 0.88),
                ("SKL-010", "essential", 0.85), ("SKL-038", "essential", 0.90),
                ("SKL-031", "essential", 0.82), ("SKL-003", "optional", 0.70),
                ("SKL-006", "optional", 0.65),  ("SKL-058", "optional", 0.60),
                ("SKL-028", "optional", 0.55),  ("SKL-052", "optional", 0.50),
            ],
            "OCC-002": [
                ("SKL-001", "essential", 0.95), ("SKL-002", "essential", 0.90),
                ("SKL-003", "essential", 0.92), ("SKL-008", "essential", 0.94),
                ("SKL-011", "essential", 0.85), ("SKL-012", "essential", 0.80),
                ("SKL-021", "essential", 0.78), ("SKL-028", "essential", 0.82),
                ("SKL-004", "optional", 0.70),  ("SKL-009", "optional", 0.72),
                ("SKL-013", "optional", 0.68),  ("SKL-053", "optional", 0.65),
            ],
            "OCC-003": [
                ("SKL-013", "essential", 0.95), ("SKL-014", "essential", 0.92),
                ("SKL-011", "essential", 0.90), ("SKL-012", "essential", 0.90),
                ("SKL-019", "essential", 0.92), ("SKL-020", "essential", 0.85),
                ("SKL-045", "essential", 0.82), ("SKL-050", "essential", 0.80),
                ("SKL-017", "optional", 0.75),  ("SKL-018", "optional", 0.72),
                ("SKL-056", "optional", 0.68),  ("SKL-057", "optional", 0.60),
            ],
            "OCC-004": [
                ("SKL-001", "essential", 0.88), ("SKL-021", "essential", 0.95),
                ("SKL-022", "essential", 0.90), ("SKL-025", "essential", 0.88),
                ("SKL-026", "essential", 0.86), ("SKL-027", "essential", 0.84),
                ("SKL-031", "essential", 0.89), ("SKL-042", "essential", 0.80),
                ("SKL-011", "optional", 0.72),  ("SKL-028", "optional", 0.75),
                ("SKL-032", "optional", 0.70),  ("SKL-041", "optional", 0.65),
            ],
            "OCC-005": [
                ("SKL-023", "essential", 0.96), ("SKL-024", "essential", 0.95),
                ("SKL-021", "essential", 0.82), ("SKL-025", "essential", 0.80),
                ("SKL-028", "essential", 0.78), ("SKL-029", "optional", 0.68),
                ("SKL-041", "optional", 0.62),  ("SKL-051", "optional", 0.60),
                ("SKL-052", "optional", 0.55),  ("SKL-054", "optional", 0.52),
            ],
            "OCC-006": [
                ("SKL-011", "essential", 0.94), ("SKL-012", "essential", 0.92),
                ("SKL-014", "essential", 0.90), ("SKL-015", "essential", 0.92),
                ("SKL-053", "essential", 0.95), ("SKL-057", "essential", 0.88),
                ("SKL-020", "essential", 0.82), ("SKL-049", "essential", 0.80),
                ("SKL-013", "optional", 0.75),  ("SKL-016", "optional", 0.70),
                ("SKL-028", "optional", 0.72),  ("SKL-059", "optional", 0.68),
            ],
            "OCC-007": [
                ("SKL-041", "essential", 0.96), ("SKL-042", "essential", 0.90),
                ("SKL-043", "essential", 0.92), ("SKL-044", "essential", 0.88),
                ("SKL-045", "essential", 0.90), ("SKL-046", "essential", 0.89),
                ("SKL-050", "essential", 0.86), ("SKL-047", "optional", 0.78),
                ("SKL-048", "optional", 0.75),  ("SKL-049", "optional", 0.72),
                ("SKL-016", "optional", 0.65),  ("SKL-052", "optional", 0.60),
            ],
            "OCC-008": [
                ("SKL-001", "essential", 0.92), ("SKL-031", "essential", 0.95),
                ("SKL-033", "essential", 0.92), ("SKL-034", "essential", 0.94),
                ("SKL-035", "essential", 0.88), ("SKL-036", "essential", 0.86),
                ("SKL-037", "essential", 0.84), ("SKL-038", "essential", 0.88),
                ("SKL-011", "optional", 0.70),  ("SKL-032", "optional", 0.75),
                ("SKL-039", "optional", 0.68),  ("SKL-053", "optional", 0.65),
            ],
            "OCC-009": [
                ("SKL-051", "essential", 0.95), ("SKL-052", "essential", 0.92),
                ("SKL-054", "essential", 0.96), ("SKL-055", "essential", 0.94),
                ("SKL-058", "essential", 0.85), ("SKL-040", "essential", 0.80),
                ("SKL-021", "optional", 0.60),  ("SKL-031", "optional", 0.58),
                ("SKL-056", "optional", 0.65),  ("SKL-060", "optional", 0.62),
            ],
            "OCC-010": [
                ("SKL-031", "essential", 0.96), ("SKL-032", "essential", 0.95),
                ("SKL-015", "essential", 0.88), ("SKL-016", "essential", 0.80),
                ("SKL-020", "essential", 0.82), ("SKL-039", "essential", 0.85),
                ("SKL-049", "optional", 0.72),  ("SKL-057", "optional", 0.70),
                ("SKL-011", "optional", 0.68),  ("SKL-044", "optional", 0.62),
            ]
        }

        edge_records = []
        for occ_id, skills in occ_skill_map.items():
            for s_id, rel_type, rel_weight in skills:
                edge_records.append({
                    "occupation_id": occ_id,
                    "skill_id": s_id,
                    "relation_type": rel_type,
                    "relevance_weight": round(float(rel_weight), 2),
                    "source": "synthetic_fallback"
                })
        edge_df = pd.DataFrame(edge_records)

        # Skill-to-skill relationships (prerequisites and co-occurrence within cluster)
        skill_rels = [
            ("SKL-001", "SKL-002", "prerequisite", 0.90),
            ("SKL-001", "SKL-004", "prerequisite", 0.85),
            ("SKL-002", "SKL-003", "prerequisite", 0.88),
            ("SKL-004", "SKL-005", "related", 0.80),
            ("SKL-011", "SKL-012", "prerequisite", 0.92),
            ("SKL-015", "SKL-011", "prerequisite", 0.80),
            ("SKL-013", "SKL-014", "related", 0.85),
            ("SKL-021", "SKL-022", "prerequisite", 0.90),
            ("SKL-023", "SKL-024", "prerequisite", 0.95),
            ("SKL-028", "SKL-053", "related", 0.82),
            ("SKL-031", "SKL-032", "related", 0.88),
            ("SKL-031", "SKL-033", "related", 0.80),
            ("SKL-041", "SKL-042", "related", 0.85),
            ("SKL-044", "SKL-042", "prerequisite", 0.80),
            ("SKL-045", "SKL-050", "related", 0.82),
            ("SKL-051", "SKL-055", "related", 0.85)
        ]
        skill_rel_df = pd.DataFrame([
            {
                "skill_id_1": s1,
                "skill_id_2": s2,
                "relation_type": rtype,
                "weight": w,
                "source": "synthetic_fallback"
            }
            for s1, s2, rtype, w in skill_rels
        ])

        # Write fallback files to data/processed
        occ_df.to_csv(self.processed_dir / "occupations.csv", index=False)
        skill_df.to_csv(self.processed_dir / "skills.csv", index=False)
        edge_df.to_csv(self.processed_dir / "occupation_skill_edges.csv", index=False)
        skill_rel_df.to_csv(self.processed_dir / "skill_relationships.csv", index=False)

        report = {
            "source": "synthetic_fallback",
            "imported_at": datetime.utcnow().isoformat(),
            "status": "completed",
            "warnings": ["Using synthetic fallback data as no real ESCO files were found in data/raw/esco."],
            "errors": [],
            "metrics": {
                "total_occupations": int(len(occ_df)),
                "total_skills": int(len(skill_df)),
                "total_occupation_skill_edges": int(len(edge_df)),
                "total_skill_relationships": int(len(skill_rel_df)),
                "duplicate_occupations_filtered": 0,
                "duplicate_skills_filtered": 0,
                "unresolved_occupation_references": 0,
                "unresolved_skill_references": 0,
                "essential_edges_count": int((edge_df["relation_type"] == "essential").sum()),
                "optional_edges_count": int((edge_df["relation_type"] != "essential").sum()),
                "clusters_count": len(clusters),
                "generic_skills_count": int(skill_df["is_generic"].sum())
            }
        }

        with open(self.processed_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
