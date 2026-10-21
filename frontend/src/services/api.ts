import {
  Occupation, Skill, Employee, EmployeeGapsResponse,
  RecommendationResponse, SimulationEpisodeResponse, DataQualityReport
} from '../types';

const API_BASE = '/api';

export const api = {
  // Health
  getHealth: async () => {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch health');
    return res.json();
  },

  // Occupations & Skills
  getOccupations: async (): Promise<Occupation[]> => {
    const res = await fetch(`${API_BASE}/occupations`);
    if (!res.ok) throw new Error('Failed to fetch occupations');
    return res.json();
  },

  getSkills: async (clusterId?: number, isGeneric?: boolean): Promise<Skill[]> => {
    const params = new URLSearchParams();
    if (clusterId !== undefined) params.append('cluster_id', String(clusterId));
    if (isGeneric !== undefined) params.append('is_generic', String(isGeneric));
    const res = await fetch(`${API_BASE}/skills?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch skills');
    return res.json();
  },

  getGraphMetrics: async () => {
    const res = await fetch(`${API_BASE}/graph/metrics`);
    if (!res.ok) throw new Error('Failed to fetch graph metrics');
    return res.json();
  },

  // Employees
  getEmployees: async (occupationId?: string, limit = 100): Promise<Employee[]> => {
    const params = new URLSearchParams();
    if (occupationId) params.append('occupation_id', occupationId);
    params.append('limit', String(limit));
    const res = await fetch(`${API_BASE}/employees?${params.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch employees');
    return res.json();
  },

  getEmployeeDetail: async (id: string): Promise<Employee> => {
    const res = await fetch(`${API_BASE}/employees/${id}`);
    if (!res.ok) throw new Error(`Failed to fetch employee ${id}`);
    return res.json();
  },

  createEmployee: async (data: { name: string; email: string; occupation_id: string; department?: string }) => {
    const res = await fetch(`${API_BASE}/employees`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Failed to create employee');
    }
    return res.json();
  },

  updateSkillState: async (employeeId: string, skillId: string, state: string, score?: number) => {
    const params = new URLSearchParams({ state });
    if (score !== undefined) params.append('score', String(score));
    const res = await fetch(`${API_BASE}/employees/${employeeId}/skills/${skillId}?${params.toString()}`, {
      method: 'PUT'
    });
    if (!res.ok) throw new Error('Failed to update skill state');
    return res.json();
  },

  // Gaps
  getEmployeeGaps: async (employeeId: string, unassessedPolicy = 'pessimistic'): Promise<EmployeeGapsResponse> => {
    const res = await fetch(`${API_BASE}/gaps/${employeeId}?unassessed_policy=${unassessedPolicy}`);
    if (!res.ok) throw new Error('Failed to fetch skill gaps');
    return res.json();
  },

  // Recommendations & Modules
  getRecommendations: async (employeeId: string, policy = 'adaptive'): Promise<RecommendationResponse> => {
    const res = await fetch(`${API_BASE}/recommendations/${employeeId}?policy=${policy}`);
    if (!res.ok) throw new Error('Failed to fetch recommendations');
    return res.json();
  },

  overrideRecommendation: async (employeeId: string, moduleId: string, reason: string) => {
    const res = await fetch(`${API_BASE}/recommendations/${employeeId}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ selected_module_id: moduleId, reason })
    });
    if (!res.ok) throw new Error('Failed to override recommendation');
    return res.json();
  },

  completeModule: async (employeeId: string, moduleId: string, score?: number, manualPassed?: boolean) => {
    const res = await fetch(`${API_BASE}/recommendations/${employeeId}/complete-module`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ module_id: moduleId, score, manual_passed: manualPassed })
    });
    if (!res.ok) throw new Error('Failed to complete module');
    return res.json();
  },

  // Simulator
  simulateEpisode: async (employeeId: string, policy: string, targetReadiness = 0.80, maxBudget = 50, seed = 42): Promise<SimulationEpisodeResponse> => {
    const res = await fetch(`${API_BASE}/simulator/run-episode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ employee_id: employeeId, policy, target_readiness: targetReadiness, max_budget: maxBudget, seed })
    });
    if (!res.ok) throw new Error('Failed to simulate episode');
    return res.json();
  },

  compareCohort: async (cohortSize = 50, targetReadiness = 0.80, maxBudget = 50, seed = 42) => {
    const res = await fetch(`${API_BASE}/simulator/compare-cohort`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cohort_size: cohortSize, target_readiness: targetReadiness, max_budget: maxBudget, seed })
    });
    if (!res.ok) throw new Error('Failed to run cohort simulation');
    return res.json();
  },

  // Experiments
  getClassifierMetrics: async () => {
    const res = await fetch(`${API_BASE}/experiments/classifier`);
    if (!res.ok) throw new Error('Failed to fetch classifier metrics');
    return res.json();
  },

  getPolicyMetrics: async () => {
    const res = await fetch(`${API_BASE}/experiments/policies`);
    if (!res.ok) throw new Error('Failed to fetch policy metrics');
    return res.json();
  },

  getAblationMetrics: async () => {
    const res = await fetch(`${API_BASE}/experiments/ablations`);
    if (!res.ok) throw new Error('Failed to fetch ablation metrics');
    return res.json();
  },

  // Admin
  getDataQualityReport: async (): Promise<DataQualityReport> => {
    const res = await fetch(`${API_BASE}/admin/quality-report`);
    if (!res.ok) throw new Error('Failed to fetch data quality report');
    return res.json();
  },

  importDataset: async (forceFallback = true) => {
    const res = await fetch(`${API_BASE}/admin/import-dataset?force_fallback=${forceFallback}`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to import dataset');
    return res.json();
  },

  reseedDatabase: async () => {
    const res = await fetch(`${API_BASE}/admin/reseed-database`, {
      method: 'POST'
    });
    if (!res.ok) throw new Error('Failed to reseed database');
    return res.json();
  }
};
