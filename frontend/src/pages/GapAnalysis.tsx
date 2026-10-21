import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { Employee, EmployeeGapsResponse, SkillGapItem } from '../types';

interface GapAnalysisProps {
  initialEmployeeId?: string;
  onNavigateToRecommendations?: (empId: string) => void;
}

export const GapAnalysis: React.FC<GapAnalysisProps> = ({
  initialEmployeeId,
  onNavigateToRecommendations
}) => {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [selectedEmpId, setSelectedEmpId] = useState<string>(initialEmployeeId || 'EMP-0001');
  const [unassessedPolicy, setUnassessedPolicy] = useState<string>('pessimistic');
  const [gapData, setGapData] = useState<EmployeeGapsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [filterCluster, setFilterCluster] = useState<string>('all');

  useEffect(() => {
    api.getEmployees(undefined, 200).then(res => {
      setEmployees(res);
      if (!initialEmployeeId && res.length > 0) {
        setSelectedEmpId(res[0].employee_id);
      }
    });
  }, []);

  useEffect(() => {
    if (selectedEmpId) {
      loadGaps(selectedEmpId, unassessedPolicy);
    }
  }, [selectedEmpId, unassessedPolicy]);

  async function loadGaps(empId: string, policy: string) {
    setLoading(true);
    try {
      const data = await api.getEmployeeGaps(empId, policy);
      setGapData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  const clusters = gapData?.gaps ? Array.from(new Set(gapData.gaps.map(g => g.cluster_name))) : [];
  const filteredGaps = gapData?.gaps.filter(g =>
    filterCluster === 'all' ? true : g.cluster_name === filterCluster
  ) || [];

  return (
    <div className="gap-analysis-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Skill-Gap & Criticality Analysis</h1>
          <p className="page-subtitle">
            Formulation: $G_e = S_r \setminus P_e$. Evaluates 4 paper features with Random Forest Criticality predictions and explainability rationales.
          </p>
        </div>
        {gapData && onNavigateToRecommendations && (
          <button
            className="btn btn-primary"
            onClick={() => onNavigateToRecommendations(gapData.employee_id)}
          >
            💡 Recommended Next Module
          </button>
        )}
      </div>

      {/* Selectors */}
      <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 280px' }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>SELECT EMPLOYEE</label>
          <select
            className="select"
            value={selectedEmpId}
            onChange={(e) => setSelectedEmpId(e.target.value)}
          >
            {employees.map(e => (
              <option key={e.employee_id} value={e.employee_id}>
                {e.employee_id}: {e.name} ({e.occupation_title})
              </option>
            ))}
          </select>
        </div>

        <div style={{ flex: '0 0 250px' }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>UNASSESSED SKILL POLICY</label>
          <select
            className="select"
            value={unassessedPolicy}
            onChange={(e) => setUnassessedPolicy(e.target.value)}
          >
            <option value="pessimistic">Pessimistic (Treat unassessed as gaps)</option>
            <option value="optimistic">Optimistic (Assume mastered until assessed)</option>
            <option value="in_progress_as_gap">In-Progress Only (Skip unassessed)</option>
          </select>
        </div>

        <div style={{ flex: '0 0 220px' }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>FILTER BY CLUSTER</label>
          <select
            className="select"
            value={filterCluster}
            onChange={(e) => setFilterCluster(e.target.value)}
          >
            <option value="all">All Clusters ({clusters.length})</option>
            {clusters.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Overview Cards */}
      {gapData && (
        <div className="stats-grid">
          <div className="glass-panel stat-card">
            <div className="stat-label">Role Readiness</div>
            <div className="stat-value" style={{ color: gapData.current_readiness >= 0.8 ? 'var(--color-success)' : '#a5b4fc' }}>
              {(gapData.current_readiness * 100).toFixed(1)}%
            </div>
            <div className="stat-sub">Target: 80% Threshold</div>
          </div>

          <div className="glass-panel stat-card">
            <div className="stat-label">Mastered Skills</div>
            <div className="stat-value">{gapData.mastered_skills_count}</div>
            <div className="stat-sub">of {gapData.total_required_skills} Required Skills</div>
          </div>

          <div className="glass-panel stat-card">
            <div className="stat-label">Outstanding Gaps</div>
            <div className="stat-value" style={{ color: 'var(--color-warning)' }}>{gapData.gaps_count}</div>
            <div className="stat-sub">Deficits to Remediate</div>
          </div>

          <div className="glass-panel stat-card">
            <div className="stat-label">Critical Gaps</div>
            <div className="stat-value" style={{ color: 'var(--color-danger)' }}>
              {gapData.gaps.filter(g => g.predicted_criticality === 1).length}
            </div>
            <div className="stat-sub">High Impact / Vulnerable</div>
          </div>
        </div>
      )}

      {/* Gaps Table */}
      <div className="glass-panel table-container">
        {loading ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Computing skill gaps & feature vectors...</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Skill Gap & Cluster</th>
                <th>Relevance (w_r,s)</th>
                <th>Generic</th>
                <th>Cluster Coverage</th>
                <th>Predicted Criticality</th>
                <th>Decision Rationale</th>
              </tr>
            </thead>
            <tbody>
              {filteredGaps.map(g => (
                <tr key={g.skill_id}>
                  <td>
                    <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{g.skill_name}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      ID: {g.skill_id} • Cluster: <span style={{ color: '#93c5fd' }}>{g.cluster_name}</span>
                    </div>
                  </td>
                  <td>
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{g.relevance_weight.toFixed(2)}</span>
                  </td>
                  <td>
                    {g.is_generic === 1 ? (
                      <span className="badge badge-info">Generic</span>
                    ) : (
                      <span className="badge" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>Domain</span>
                    )}
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '100px' }}>
                      <div className="progress-bar-bg" style={{ flex: 1 }}>
                        <div className="progress-bar-fill" style={{ width: `${Math.round(g.cluster_coverage * 100)}%` }} />
                      </div>
                      <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>{(g.cluster_coverage * 100).toFixed(0)}%</span>
                    </div>
                  </td>
                  <td>
                    {g.predicted_criticality === 1 ? (
                      <div>
                        <span className="badge badge-danger">Critical Gap</span>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                          Prob: {(g.criticality_probability * 100).toFixed(0)}%
                        </div>
                      </div>
                    ) : (
                      <div>
                        <span className="badge badge-success">Secondary</span>
                        <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                          Prob: {(g.criticality_probability * 100).toFixed(0)}%
                        </div>
                      </div>
                    )}
                  </td>
                  <td style={{ maxWidth: '340px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                    {g.rationale}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
