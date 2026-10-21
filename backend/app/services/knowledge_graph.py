import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OccupationSkillKnowledgeGraph:
    """
    Heterogeneous Knowledge Graph capturing:
    - Occupation Nodes (attributes: id, title, clusters, source)
    - Skill Nodes (attributes: id, name, cluster_id, cluster_name, is_generic, complexity)
    - Occupation-Skill Bipartite Edges (attributes: weight/relevance, relation_type, tier)
    - Skill-Skill Relationships (attributes: relation_type: prerequisite/related, weight)
    
    Provides query methods, graph topological metrics, and serialization for frontend.
    """

    def __init__(self):
        # We use a directed graph to allow directed prerequisite relationships
        # while representing bidirectional relations as reciprocal edges
        self.graph = nx.DiGraph()
        self.occupation_ids: Set[str] = set()
        self.skill_ids: Set[str] = set()
        self.cluster_map: Dict[int, List[str]] = {}
        self.skill_to_cluster: Dict[str, int] = {}
        self.is_generic_map: Dict[str, bool] = {}
        self.occ_required_skills: Dict[str, Dict[str, float]] = {}

    def build_from_directory(self, data_dir: Path) -> Dict[str, Any]:
        """Loads data from CSVs in data_dir and constructs the heterogeneous graph."""
        occ_file = data_dir / "occupations.csv"
        skill_file = data_dir / "skills.csv"
        edge_file = data_dir / "occupation_skill_edges.csv"
        skill_rel_file = data_dir / "skill_relationships.csv"

        if not (occ_file.exists() and skill_file.exists() and edge_file.exists()):
            raise FileNotFoundError(f"Missing required CSVs in {data_dir}")

        occ_df = pd.read_csv(occ_file)
        skill_df = pd.read_csv(skill_file)
        edge_df = pd.read_csv(edge_file)
        skill_rel_df = pd.read_csv(skill_rel_file) if skill_rel_file.exists() else None

        return self.build(occ_df, skill_df, edge_df, skill_rel_df)

    def build(
        self,
        occ_df: pd.DataFrame,
        skill_df: pd.DataFrame,
        edge_df: pd.DataFrame,
        skill_rel_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Constructs graph nodes and edges from DataFrames."""
        self.graph.clear()
        self.occupation_ids.clear()
        self.skill_ids.clear()
        self.cluster_map.clear()
        self.skill_to_cluster.clear()
        self.is_generic_map.clear()
        self.occ_required_skills.clear()

        # 1. Add Skill Nodes
        for _, row in skill_df.iterrows():
            s_id = str(row["skill_id"])
            c_id = int(row.get("cluster_id", 0))
            is_gen = bool(row.get("is_generic", False))

            self.skill_ids.add(s_id)
            self.skill_to_cluster[s_id] = c_id
            self.is_generic_map[s_id] = is_gen
            self.cluster_map.setdefault(c_id, []).append(s_id)

            self.graph.add_node(
                s_id,
                node_type="skill",
                name=str(row["name"]),
                cluster_id=c_id,
                cluster_name=str(row.get("cluster_name", f"Cluster {c_id}")),
                is_generic=is_gen,
                complexity=float(row.get("complexity", 0.5)),
                source=str(row.get("source", "unknown"))
            )

        # 2. Add Occupation Nodes
        for _, row in occ_df.iterrows():
            o_id = str(row["occupation_id"])
            self.occupation_ids.add(o_id)
            self.occ_required_skills[o_id] = {}

            self.graph.add_node(
                o_id,
                node_type="occupation",
                title=str(row["title"]),
                primary_cluster_id=int(row.get("primary_cluster_id", 0)),
                secondary_cluster_id=int(row.get("secondary_cluster_id", 0)),
                source=str(row.get("source", "unknown"))
            )

        # 3. Add Occupation-Skill Bipartite Edges
        for _, row in edge_df.iterrows():
            o_id = str(row["occupation_id"])
            s_id = str(row["skill_id"])
            rel_weight = float(row.get("relevance_weight", 0.70))
            rel_type = str(row.get("relation_type", "essential"))
            tier = str(row.get("tier", "domain"))

            if o_id in self.occupation_ids and s_id in self.skill_ids:
                self.occ_required_skills[o_id][s_id] = rel_weight

                # Directed edge: occupation requires skill
                self.graph.add_edge(
                    o_id,
                    s_id,
                    edge_type="requires",
                    weight=rel_weight,
                    relation_type=rel_type,
                    tier=tier
                )
                # Reciprocal edge: skill supports occupation
                self.graph.add_edge(
                    s_id,
                    o_id,
                    edge_type="supports",
                    weight=rel_weight,
                    relation_type=rel_type,
                    tier=tier
                )

        # 4. Add Skill-Skill Relationships (Prerequisites / Relatedness)
        if skill_rel_df is not None and not skill_rel_df.empty:
            for _, row in skill_rel_df.iterrows():
                s1 = str(row["skill_id_1"])
                s2 = str(row["skill_id_2"])
                rtype = str(row.get("relation_type", "related"))
                w = float(row.get("weight", 0.80))

                if s1 in self.skill_ids and s2 in self.skill_ids:
                    if rtype == "prerequisite":
                        # s1 is prerequisite of s2: s1 -> s2
                        self.graph.add_edge(s1, s2, edge_type="prerequisite_of", weight=w)
                        self.graph.add_edge(s2, s1, edge_type="has_prerequisite", weight=w)
                    else:
                        # Bidirectional related
                        self.graph.add_edge(s1, s2, edge_type="related_to", weight=w)
                        self.graph.add_edge(s2, s1, edge_type="related_to", weight=w)

        # Also add implicit intra-cluster skill co-occurrence links if no explicit skill relations
        if skill_rel_df is None or len(skill_rel_df) == 0:
            self._add_cluster_cooccurrence_edges()

        metrics = self.compute_metrics()
        logger.info(
            "Knowledge graph built: %d nodes (%d occs, %d skills), %d edges",
            self.graph.number_of_nodes(),
            len(self.occupation_ids),
            len(self.skill_ids),
            self.graph.number_of_edges()
        )
        return metrics

    def _add_cluster_cooccurrence_edges(self):
        """Adds intra-cluster co-occurrence edges to represent conceptual affinity."""
        for c_id, s_ids in self.cluster_map.items():
            # Link nearest skills within same cluster
            for i in range(len(s_ids) - 1):
                s1, s2 = s_ids[i], s_ids[i + 1]
                self.graph.add_edge(s1, s2, edge_type="cluster_affinity", weight=0.65)
                self.graph.add_edge(s2, s1, edge_type="cluster_affinity", weight=0.65)

    def get_required_skills(self, occupation_id: str) -> Dict[str, float]:
        """Returns map of {skill_id: relevance_weight} for given occupation."""
        return self.occ_required_skills.get(occupation_id, {})

    def get_skill_cluster(self, skill_id: str) -> Optional[int]:
        """Returns cluster ID for given skill."""
        return self.skill_to_cluster.get(skill_id)

    def get_cluster_skills(self, cluster_id: int) -> List[str]:
        """Returns list of skill IDs belonging to the cluster."""
        return self.cluster_map.get(cluster_id, [])

    def is_generic_skill(self, skill_id: str) -> bool:
        """Returns whether a skill is generic/cross-cutting."""
        return self.is_generic_map.get(skill_id, False)

    def get_prerequisites(self, skill_id: str) -> List[str]:
        """Returns skills that are prerequisites for this skill."""
        prereqs = []
        if skill_id in self.graph:
            for pred in self.graph.predecessors(skill_id):
                edge_data = self.graph.get_edge_data(pred, skill_id)
                if edge_data and edge_data.get("edge_type") == "prerequisite_of":
                    prereqs.append(pred)
        return prereqs

    def compute_metrics(self) -> Dict[str, Any]:
        """Computes comprehensive topological and domain statistics."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        # Undirected view for standard degree/connectivity analysis
        u_graph = self.graph.to_undirected()

        degrees = [d for _, d in u_graph.degree()]
        mean_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = int(np.max(degrees)) if degrees else 0

        occ_degrees = [u_graph.degree(o) for o in self.occupation_ids if o in u_graph]
        skill_degrees = [u_graph.degree(s) for s in self.skill_ids if s in u_graph]

        # Connected components in undirected projection
        components = list(nx.connected_components(u_graph))
        num_components = len(components)
        largest_cc_size = max(len(c) for c in components) if components else 0

        # Cluster distribution
        cluster_sizes = {c_id: len(s_list) for c_id, s_list in self.cluster_map.items()}

        metrics = {
            "total_nodes": num_nodes,
            "total_edges": num_edges,
            "occupation_nodes_count": len(self.occupation_ids),
            "skill_nodes_count": len(self.skill_ids),
            "generic_skills_count": int(sum(self.is_generic_map.values())),
            "num_clusters": len(self.cluster_map),
            "cluster_distribution": cluster_sizes,
            "graph_density": round(float(nx.density(self.graph)), 6),
            "mean_node_degree": round(mean_degree, 2),
            "max_node_degree": max_degree,
            "mean_occupation_degree": round(float(np.mean(occ_degrees)), 2) if occ_degrees else 0.0,
            "mean_skill_degree": round(float(np.mean(skill_degrees)), 2) if skill_degrees else 0.0,
            "connected_components_count": num_components,
            "largest_component_size": largest_cc_size,
            "is_bipartite_compatible": True
        }
        return metrics

    def to_cytoscape_elements(self, max_skills: int = 100) -> List[Dict[str, Any]]:
        """
        Exports a lightweight subgraph formatted for frontend visualization (Cytoscape / React-Flow).
        """
        elements = []

        # Add occupation nodes
        for o_id in list(self.occupation_ids)[:20]:
            attrs = self.graph.nodes[o_id]
            elements.append({
                "data": {
                    "id": o_id,
                    "label": attrs.get("title", o_id),
                    "type": "occupation",
                    "cluster": attrs.get("primary_cluster_id", 0)
                }
            })

        # Add skill nodes
        sampled_skills = list(self.skill_ids)[:max_skills]
        for s_id in sampled_skills:
            attrs = self.graph.nodes[s_id]
            elements.append({
                "data": {
                    "id": s_id,
                    "label": attrs.get("name", s_id),
                    "type": "skill",
                    "cluster": attrs.get("cluster_id", 0),
                    "is_generic": attrs.get("is_generic", False)
                }
            })

        # Add edges between present nodes
        present_nodes = {el["data"]["id"] for el in elements}
        for u, v, data in self.graph.edges(data=True):
            if u in present_nodes and v in present_nodes and data.get("edge_type") == "requires":
                elements.append({
                    "data": {
                        "id": f"e_{u}_{v}",
                        "source": u,
                        "target": v,
                        "weight": data.get("weight", 0.5),
                        "type": data.get("edge_type", "requires")
                    }
                })

        return elements
