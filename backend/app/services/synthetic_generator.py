import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticBaselineGenerator:
    """
    Deterministic Synthetic Data Generator matching the paper:
    - 40 Occupations
    - 220 Skills
    - 8 Latent Clusters
    - ~12% Generic skills
    - 600 Simulated Employees
    - Probabilistic initial skill possession correlated with relevance
    - Incomplete/noisy skill assessment states
    - Training modules with duration, difficulty, and prerequisites
    """

    CLUSTER_NAMES = [
        "Software Engineering & Architecture",
        "Cloud Computing & DevOps",
        "Machine Learning & AI",
        "Web & Mobile Applications",
        "Cybersecurity & Network Defense",
        "Data Engineering & Databases",
        "Distributed Systems & Reliability",
        "Agile Leadership & Product Management"
    ]

    GENERIC_SKILL_TEMPLATES = [
        "Git Version Control", "Linux Command Line Basics", "Unit Testing Fundamentals",
        "Code Review Principles", "Agile & Scrum Methodologies", "Technical Writing & RFCs",
        "Problem Solving & Debugging", "REST API Communication", "Continuous Integration Basics",
        "Container Basics (Docker)", "Database Querying (SQL)", "Security Awareness & Hygiene",
        "Data Structures & Algorithms", "Clean Code & Refactoring", "Design Patterns",
        "Monitoring & Logging Basics", "Time Management & Sprint Planning", "Stakeholder Communication",
        "Incident Triage & Root Cause", "Performance Benchmarking", "API Documentation",
        "CI/CD Pipeline Usage", "Cloud Fundamentals", "Object-Oriented Programming",
        "Asynchronous Programming", "Data Serialization (JSON/YAML)"
    ]

    DOMAIN_SKILL_PREFIXES = {
        0: ["System Architecture", "Microservices Design", "Domain-Driven Design", "Concurrency Patterns", "IPC Mechanisms", "Design System Implementation", "Dependency Injection", "Memory Management", "Compilers & Interpreters"],
        1: ["Kubernetes Cluster Admin", "Terraform IaC", "AWS Infrastructure", "Azure Cloud Services", "GCP Core Services", "Ansible Automation", "Prometheus & Grafana", "Helm Chart Packaging", "Service Mesh (Istio)"],
        2: ["Scikit-Learn Modeling", "PyTorch Deep Learning", "Feature Engineering", "Model Evaluation & Drift", "NLP Transformer Pipelines", "Computer Vision CNNs", "MLOps & MLflow", "Hyperparameter Tuning", "Bayesian Optimization"],
        3: ["React UI Development", "TypeScript Type Systems", "State Management (Redux/Zustand)", "CSS Grid & Responsive Design", "GraphQL API Clients", "Web Performance Optimization", "Progressive Web Apps", "Accessibility (WCAG)", "Next.js Framework"],
        4: ["OWASP Top 10 Mitigation", "Penetration Testing", "OAuth2 & OIDC Security", "PKI & Certificate Management", "SIEM Telemetry & Splunk", "Network Packet Inspection", "Threat Modeling", "Zero-Trust Architecture", "Vulnerability Remediation"],
        5: ["PostgreSQL Administration", "Apache Kafka Streaming", "Apache Spark Distributed ETL", "Data Warehousing (Snowflake)", "Database Indexing & Tuning", "NoSQL Document Stores", "Data Quality & Great Expectations", "dbt Transformations", "Data Lakehouse Design"],
        6: ["Distributed Consensus (Raft)", "gRPC & Protobuf RPCs", "High Availability & Clustering", "Distributed Cache (Redis)", "Chaos Engineering", "Rate Limiting & Circuit Breaking", "Message Queueing (RabbitMQ)", "Disaster Recovery", "SRE SLO/SLI Management"],
        7: ["Product Backlog Prioritization", "OKRs & Metric Definition", "Sprint Retrospectives", "Release Management", "Cost Governance & FinOps", "Engineering Mentorship", "Vendor Risk Assessment", "Cross-Functional Alignment", "Compliance Auditing"]
    }

    OCCUPATION_TITLES = [
        # Cluster 0
        "Senior Software Architect", "Principal Backend Engineer", "Systems Programmer", "Enterprise Platform Engineer", "Core Infrastructure Developer",
        # Cluster 1
        "DevOps Engineer", "Cloud Solutions Architect", "Site Reliability Engineer (SRE)", "Kubernetes Platform Engineer", "Cloud Security Engineer",
        # Cluster 2
        "Data Scientist", "Machine Learning Engineer", "AI Research Engineer", "MLOps Engineer", "Computer Vision Specialist",
        # Cluster 3
        "Frontend Engineer", "Full-Stack Developer", "Mobile App Engineer (iOS/Android)", "UI/UX Engineer", "Design Systems Engineer",
        # Cluster 4
        "Cybersecurity Analyst", "Security Operations Center (SOC) Lead", "Application Security Engineer", "Penetration Tester", "Information Security Officer",
        # Cluster 5
        "Data Engineer", "Database Administrator (DBA)", "Big Data Platform Engineer", "Analytics Engineer", "Business Intelligence Architect",
        # Cluster 6
        "Distributed Systems Engineer", "Reliability Architect", "Messaging Infrastructure Specialist", "High-Throughput Systems Engineer", "Edge Computing Engineer",
        # Cluster 7
        "Technical Product Manager", "Engineering Manager", "Agile Delivery Lead", "Scrum Master", "IT Governance & Compliance Manager"
    ]

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        num_occupations: int = 40,
        num_skills: int = 220,
        num_clusters: int = 8,
        generic_ratio: float = 0.12,
        num_employees: int = 600,
        random_seed: int = 42
    ):
        from backend.app.core.config import settings
        self.output_dir = output_dir or settings.DATA_SYNTHETIC_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_occupations = num_occupations
        self.num_skills = num_skills
        self.num_clusters = num_clusters
        self.generic_ratio = generic_ratio
        self.num_employees = num_employees
        self.random_seed = random_seed

    def generate(self) -> Dict:
        """
        Executes deterministic generation of baseline synthetic data.
        Returns a dictionary of generated metrics and file paths.
        """
        rng = np.random.default_rng(self.random_seed)

        logger.info(
            "Generating synthetic baseline: %d occs, %d skills, %d clusters, %d employees (seed=%d)",
            self.num_occupations, self.num_skills, self.num_clusters, self.num_employees, self.random_seed
        )

        # 1. Generate Skills
        skills_df = self._generate_skills(rng)

        # 2. Generate Occupations
        occupations_df = self._generate_occupations(rng)

        # 3. Generate Bipartite Occupation-Skill Edges
        edges_df = self._generate_occupation_skill_edges(occupations_df, skills_df, rng)

        # 4. Generate Training Modules
        modules_df = self._generate_training_modules(skills_df, rng)

        # 5. Generate Employees and Incomplete Skill Profiles
        employees_df, emp_skills_df = self._generate_employees(occupations_df, edges_df, skills_df, rng)

        # 6. Save all artifacts to data/synthetic
        skills_path = self.output_dir / "skills.csv"
        occupations_path = self.output_dir / "occupations.csv"
        edges_path = self.output_dir / "occupation_skill_edges.csv"
        modules_path = self.output_dir / "training_modules.csv"
        employees_path = self.output_dir / "employees.csv"
        emp_skills_path = self.output_dir / "employee_skills.csv"
        meta_path = self.output_dir / "metadata.json"

        skills_df.to_csv(skills_path, index=False)
        occupations_df.to_csv(occupations_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        modules_df.to_csv(modules_path, index=False)
        employees_df.to_csv(employees_path, index=False)
        emp_skills_df.to_csv(emp_skills_path, index=False)

        metadata = {
            "dataset_type": "synthetic_paper_reproduction_baseline",
            "generated_at": datetime.utcnow().isoformat(),
            "random_seed": self.random_seed,
            "parameters": {
                "num_occupations": self.num_occupations,
                "num_skills": self.num_skills,
                "num_clusters": self.num_clusters,
                "generic_ratio": self.generic_ratio,
                "num_employees": self.num_employees
            },
            "counts": {
                "occupations": len(occupations_df),
                "skills": len(skills_df),
                "generic_skills": int(skills_df["is_generic"].sum()),
                "occupation_skill_edges": len(edges_df),
                "training_modules": len(modules_df),
                "employees": len(employees_df),
                "employee_skill_records": len(emp_skills_df),
                "mean_skills_per_occupation": round(float(edges_df.groupby("occupation_id").size().mean()), 2),
                "mean_initial_mastered_ratio": round(
                    float((emp_skills_df["state"] == "mastered").sum() / len(emp_skills_df)), 4
                )
            },
            "assumptions_documentation": [
                "Occupations require 15-22 skills comprising primary cluster skills, secondary cluster skills, and generic cross-functional skills.",
                "Occupation-skill relevance weights are sampled from Beta(4, 2) scaled to [0.45, 0.98] to reflect realistic post-recruitment importance distributions.",
                "Employees are post-recruitment hires, initialized with probabilistic skill possession proportional to occupation relevance weight (mean ~48% initial readiness).",
                "Assessments noise: 10% of required skills are initialized as 'not_assessed' to simulate realistic incomplete onboarding portfolios."
            ]
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return metadata

    def _generate_skills(self, rng: np.random.Generator) -> pd.DataFrame:
        target_generic = int(round(self.num_skills * self.generic_ratio))
        domain_skills_count = self.num_skills - target_generic

        records = []
        skill_counter = 1

        # 1. Generic Skills
        for i in range(target_generic):
            s_id = f"SKL-{skill_counter:03d}"
            name = (
                self.GENERIC_SKILL_TEMPLATES[i % len(self.GENERIC_SKILL_TEMPLATES)]
                if i < len(self.GENERIC_SKILL_TEMPLATES)
                else f"General Professional Competency {i + 1}"
            )
            # Generic skills assigned cluster -1 or distributed
            records.append({
                "skill_id": s_id,
                "name": name,
                "cluster_id": int(i % self.num_clusters),
                "cluster_name": self.CLUSTER_NAMES[i % self.num_clusters],
                "is_generic": True,
                "complexity": round(float(rng.uniform(0.3, 0.7)), 2),
                "source": "synthetic_baseline"
            })
            skill_counter += 1

        # 2. Cluster Domain Skills
        skills_per_cluster = domain_skills_count // self.num_clusters
        remainder = domain_skills_count % self.num_clusters

        for c_id in range(self.num_clusters):
            c_name = self.CLUSTER_NAMES[c_id]
            prefixes = self.DOMAIN_SKILL_PREFIXES.get(c_id, ["Technical Competency"])
            count_for_c = skills_per_cluster + (1 if c_id < remainder else 0)

            for j in range(count_for_c):
                s_id = f"SKL-{skill_counter:03d}"
                prefix = prefixes[j % len(prefixes)]
                name = prefix if j < len(prefixes) else f"{prefix} Advanced Practice {j + 1}"
                records.append({
                    "skill_id": s_id,
                    "name": name,
                    "cluster_id": c_id,
                    "cluster_name": c_name,
                    "is_generic": False,
                    "complexity": round(float(rng.uniform(0.5, 0.95)), 2),
                    "source": "synthetic_baseline"
                })
                skill_counter += 1

        return pd.DataFrame(records)

    def _generate_occupations(self, rng: np.random.Generator) -> pd.DataFrame:
        records = []
        for i in range(self.num_occupations):
            occ_id = f"OCC-{i + 1:03d}"
            title = (
                self.OCCUPATION_TITLES[i]
                if i < len(self.OCCUPATION_TITLES)
                else f"Specialized Technology Role {i + 1}"
            )
            primary_c = i % self.num_clusters
            secondary_c = (primary_c + int(rng.choice([1, 2, 3]))) % self.num_clusters

            records.append({
                "occupation_id": occ_id,
                "title": title,
                "primary_cluster_id": primary_c,
                "primary_cluster_name": self.CLUSTER_NAMES[primary_c],
                "secondary_cluster_id": secondary_c,
                "secondary_cluster_name": self.CLUSTER_NAMES[secondary_c],
                "source": "synthetic_baseline"
            })
        return pd.DataFrame(records)

    def _generate_occupation_skill_edges(
        self, occ_df: pd.DataFrame, skills_df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        edges = []

        generic_skills = skills_df[skills_df["is_generic"]]["skill_id"].tolist()
        cluster_skills = {
            c_id: skills_df[(~skills_df["is_generic"]) & (skills_df["cluster_id"] == c_id)]["skill_id"].tolist()
            for c_id in range(self.num_clusters)
        }

        for _, occ in occ_df.iterrows():
            occ_id = occ["occupation_id"]
            p_cluster = occ["primary_cluster_id"]
            s_cluster = occ["secondary_cluster_id"]

            # Number of skills: 16 to 22 skills per occupation
            num_primary = rng.integers(9, 13)
            num_secondary = rng.integers(4, 7)
            num_generic = rng.integers(3, 5)

            # Sample primary
            p_available = cluster_skills.get(p_cluster, [])
            p_selected = rng.choice(p_available, size=min(num_primary, len(p_available)), replace=False)

            # Sample secondary
            s_available = cluster_skills.get(s_cluster, [])
            s_selected = rng.choice(s_available, size=min(num_secondary, len(s_available)), replace=False)

            # Sample generic
            g_selected = rng.choice(generic_skills, size=min(num_generic, len(generic_skills)), replace=False)

            # Assign relevance weights
            # Primary cluster skills: higher weights [0.75, 0.98]
            for s_id in p_selected:
                weight = round(float(np.clip(rng.beta(5, 1.5) * 0.35 + 0.65, 0.65, 0.98)), 3)
                edges.append({
                    "occupation_id": occ_id,
                    "skill_id": s_id,
                    "relation_type": "essential" if weight >= 0.75 else "optional",
                    "relevance_weight": weight,
                    "tier": "primary_domain",
                    "source": "synthetic_baseline"
                })

            # Secondary cluster skills: medium weights [0.55, 0.85]
            for s_id in s_selected:
                weight = round(float(np.clip(rng.beta(3, 2) * 0.35 + 0.50, 0.50, 0.85)), 3)
                edges.append({
                    "occupation_id": occ_id,
                    "skill_id": s_id,
                    "relation_type": "essential" if weight >= 0.72 else "optional",
                    "relevance_weight": weight,
                    "tier": "secondary_domain",
                    "source": "synthetic_baseline"
                })

            # Generic skills: medium-low weights [0.45, 0.75]
            for s_id in g_selected:
                weight = round(float(np.clip(rng.beta(2.5, 2.5) * 0.35 + 0.40, 0.40, 0.75)), 3)
                edges.append({
                    "occupation_id": occ_id,
                    "skill_id": s_id,
                    "relation_type": "optional" if weight < 0.65 else "essential",
                    "relevance_weight": weight,
                    "tier": "cross_cutting",
                    "source": "synthetic_baseline"
                })

        return pd.DataFrame(edges).drop_duplicates(subset=["occupation_id", "skill_id"])

    def _generate_training_modules(
        self, skills_df: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        modules = []
        mod_counter = 1

        for _, skill in skills_df.iterrows():
            s_id = skill["skill_id"]
            s_name = skill["name"]
            complexity = float(skill["complexity"])

            # 1 module per skill
            m_id = f"MOD-{mod_counter:03d}"
            duration = int(round(complex_val := complexity * 16 + rng.integers(4, 10)))
            difficulty = "beginner" if complexity < 0.55 else ("intermediate" if complexity < 0.80 else "advanced")

            modules.append({
                "module_id": m_id,
                "target_skill_id": s_id,
                "title": f"Applied Training: {s_name}",
                "description": f"Comprehensive hands-on training module covering foundational principles and industrial workflows in {s_name}.",
                "duration_hours": duration,
                "difficulty_level": difficulty,
                "learning_cost": round(float(duration * 1.5 + complexity * 10), 1),
                "assessment_pass_threshold": 0.75,
                "prerequisite_skill_id": None
            })
            mod_counter += 1

        return pd.DataFrame(modules)

    def _generate_employees(
        self,
        occ_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        skills_df: pd.DataFrame,
        rng: np.random.Generator
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        employees = []
        emp_skills = []

        all_occ_ids = occ_df["occupation_id"].tolist()
        occ_edge_map = edges_df.groupby("occupation_id")

        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Casey", "Riley", "Avery", "Jamie", "Logan", "Dakota", "Reese", "Cameron", "Quinn"]
        last_names = ["Chen", "Patel", "Smith", "Kim", "Garcia", "Muller", "O'Connor", "Dubois", "Tanaka", "Silva", "Johnson", "Williams", "Brown", "Jones", "Davis"]

        for i in range(self.num_employees):
            emp_id = f"EMP-{i + 1:04d}"
            occ_id = all_occ_ids[i % len(all_occ_ids)]
            name = f"{first_names[i % len(first_names)]} {last_names[(i // len(first_names)) % len(last_names)]}_{i + 1}"

            employees.append({
                "employee_id": emp_id,
                "name": name,
                "email": f"{emp_id.lower()}@corp.asgp.internal",
                "occupation_id": occ_id,
                "department": "Engineering & Technology",
                "hired_date": "2026-01-15",
                "source": "synthetic_baseline"
            })

            # Skills required for assigned occupation
            if occ_id in occ_edge_map.groups:
                req_edges = occ_edge_map.get_group(occ_id)
            else:
                req_edges = pd.DataFrame()

            for _, edge in req_edges.iterrows():
                s_id = edge["skill_id"]
                rel_weight = float(edge["relevance_weight"])

                # Post-recruitment possession probability:
                # Correlated with relevance weight (candidates are selected for their relevance, but have gaps)
                # Baseline probability between 0.30 and 0.65
                prob_mastered = np.clip(0.20 + 0.40 * rel_weight + rng.normal(0, 0.08), 0.15, 0.85)
                
                # Check unassessed noise rate (10%)
                is_unassessed = rng.random() < 0.10

                if is_unassessed:
                    state = "not_assessed"
                    score = None
                else:
                    if rng.random() < prob_mastered:
                        state = "mastered"
                        score = round(float(rng.uniform(0.80, 0.98)), 2)
                    else:
                        # Split between not_mastered and in_progress
                        if rng.random() < 0.70:
                            state = "not_mastered"
                            score = round(float(rng.uniform(0.10, 0.45)), 2)
                        else:
                            state = "in_progress"
                            score = round(float(rng.uniform(0.50, 0.74)), 2)

                emp_skills.append({
                    "employee_id": emp_id,
                    "skill_id": s_id,
                    "state": state,
                    "assessment_score": score,
                    "updated_at": datetime.utcnow().isoformat()
                })

        return pd.DataFrame(employees), pd.DataFrame(emp_skills)
