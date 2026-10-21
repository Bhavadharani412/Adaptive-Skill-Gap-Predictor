export interface Occupation {
  occupation_id: string;
  title: string;
  description?: string;
  primary_cluster_id: number;
  primary_cluster_name?: string;
  secondary_cluster_id: number;
  secondary_cluster_name?: string;
  source: string;
  required_skills_count: number;
}

export interface Skill {
  skill_id: string;
  name: string;
  skill_type: string;
  cluster_id: number;
  cluster_name?: string;
  is_generic: boolean;
  complexity: number;
  source: string;
}

export interface EmployeeSkill {
  skill_id: string;
  skill_name?: string;
  state: 'not_assessed' | 'not_mastered' | 'in_progress' | 'mastered';
  assessment_score?: number | null;
}

export interface Employee {
  employee_id: string;
  name: string;
  email: string;
  occupation_id: string;
  occupation_title?: string;
  department: string;
  hired_date: string;
  readiness: number;
  mastered_count: number;
  total_gaps_count: number;
  skills?: EmployeeSkill[];
}

export interface SkillGapItem {
  skill_id: string;
  skill_name: string;
  cluster_id: number;
  cluster_name: string;
  relevance_weight: number;
  is_generic: number;
  possessed_skills_count: number;
  cluster_coverage: number;
  current_state: string;
  current_score?: number | null;
  predicted_criticality: number;
  criticality_probability: number;
  criticality_source: string;
  rationale: string;
}

export interface EmployeeGapsResponse {
  employee_id: string;
  occupation_id: string;
  occupation_title: string;
  current_readiness: number;
  target_readiness: number;
  total_required_skills: number;
  mastered_skills_count: number;
  gaps_count: number;
  gaps: SkillGapItem[];
}

export interface RecommendationItem {
  module_id: string;
  target_skill_id: string;
  skill_name: string;
  occupation_id: string;
  relevance_weight: number;
  cluster_name: string;
  cluster_transfer_bonus: number;
  prerequisite_fulfilled: boolean;
  is_generic: boolean;
  predicted_criticality?: number | null;
  composite_score: number;
  duration_hours: number;
  difficulty_level: string;
  policy_used: string;
  rationale: string;
}

export interface RecommendationResponse {
  employee_id: string;
  policy: string;
  current_readiness: number;
  recommended_module?: RecommendationItem | null;
  alternative_modules: RecommendationItem[];
}

export interface SimulationStepLog {
  step: number;
  module_id: string;
  target_skill_id: string;
  passed: boolean;
  score: number;
  readiness_after_step: number;
  rationale: string;
}

export interface SimulationEpisodeResponse {
  employee_id: string;
  policy: string;
  initial_readiness: number;
  final_readiness: number;
  target_readiness: number;
  steps_to_target: number;
  reached_target: boolean;
  censored: boolean;
  total_learning_hours: number;
  history: SimulationStepLog[];
}

export interface DataQualityReport {
  source: string;
  imported_at: string;
  status: string;
  warnings: string[];
  errors: string[];
  metrics: {
    total_occupations: number;
    total_skills: number;
    total_occupation_skill_edges: number;
    total_skill_relationships?: number;
    essential_edges_count: number;
    optional_edges_count: number;
    clusters_count?: number;
    generic_skills_count?: number;
  };
}
