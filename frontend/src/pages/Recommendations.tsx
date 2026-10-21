import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { Employee, RecommendationResponse, RecommendationItem } from '../types';

interface RecommendationsProps {
  initialEmployeeId?: string;
}

export const Recommendations: React.FC<RecommendationsProps> = ({ initialEmployeeId }) => {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [selectedEmpId, setSelectedEmpId] = useState<string>(initialEmployeeId || 'EMP-0001');
  const [policy, setPolicy] = useState<string>('adaptive');
  const [recData, setRecData] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Completion modal / state
  const [customScore, setCustomScore] = useState<string>('0.88');
  const [completing, setCompleting] = useState<boolean>(false);
  const [completionMessage, setCompletionMessage] = useState<string>('');

  // Override modal
  const [showOverrideModal, setShowOverrideModal] = useState<boolean>(false);
  const [overrideModuleId, setOverrideModuleId] = useState<string>('');
  const [overrideReason, setOverrideReason] = useState<string>('');

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
      loadRecommendations(selectedEmpId, policy);
    }
  }, [selectedEmpId, policy]);

  async function loadRecommendations(empId: string, pol: string) {
    setLoading(true);
    setCompletionMessage('');
    try {
      const data = await api.getRecommendations(empId, pol);
      setRecData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleCompleteModule(moduleId: string) {
    if (!selectedEmpId) return;
    setCompleting(true);
    try {
      const score = customScore ? parseFloat(customScore) : 0.88;
      const res = await api.completeModule(selectedEmpId, moduleId, score, score >= 0.75);
      setCompletionMessage(
        `✓ Module "${res.module_completed}" successfully completed with score ${(res.assessment_score * 100).toFixed(0)}%! ` +
        `Updated readiness: ${(res.updated_readiness * 100).toFixed(1)}%. Remaining gaps: ${res.remaining_gaps_count}.`
      );
      // Reload recommendation with updated profile
      loadRecommendations(selectedEmpId, policy);
    } catch (err: any) {
      alert('Error completing module: ' + err.message);
    } finally {
      setCompleting(false);
    }
  }

  async function handleOverride(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedEmpId || !overrideModuleId) return;
    try {
      await api.overrideRecommendation(selectedEmpId, overrideModuleId, overrideReason);
      setShowOverrideModal(false);
      setOverrideReason('');
      alert('Managerial override recorded in audit logs.');
      loadRecommendations(selectedEmpId, policy);
    } catch (err: any) {
      alert('Error overriding recommendation: ' + err.message);
    }
  }

  const top = recData?.recommended_module;

  return (
    <div className="recommendations-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Adaptive Training Recommendations</h1>
          <p className="page-subtitle">
            Dynamic graph-informed ranking engine balancing occupational relevance (w_r,s), cluster-transfer momentum (β), and prerequisite feasibility (γ).
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            className="btn btn-secondary"
            onClick={() => setShowOverrideModal(true)}
            disabled={!top}
          >
            🛡️ Manager Override
          </button>
        </div>
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

        <div style={{ flex: '0 0 280px' }}>
          <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>RECOMMENDATION POLICY</label>
          <select
            className="select"
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
          >
            <option value="adaptive">Policy C: Adaptive Graph Engine (Recommended)</option>
            <option value="static">Policy B: Static Relevance Ranking</option>
            <option value="random">Policy A: Random Baseline</option>
          </select>
        </div>
      </div>

      {/* Completion feedback banner */}
      {completionMessage && (
        <div style={{ padding: '1rem', background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.35)', borderRadius: 'var(--radius-md)', color: '#6ee7b7', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
          {completionMessage}
        </div>
      )}

      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>Calculating next optimal training module...</div>
      ) : top ? (
        <>
          {/* Top Recommendation Hero Card */}
          <div className="hero-rec-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
              <div>
                <span className="badge badge-purple" style={{ marginBottom: '0.5rem' }}>
                  ⭐ Primary Recommendation (Rank #1)
                </span>
                <h2 style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.25rem' }}>
                  {top.module_id}: Applied Training in {top.skill_name}
                </h2>
                <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  <span>🎯 Skill ID: <strong style={{ color: '#fff' }}>{top.target_skill_id}</strong></span>
                  <span>📁 Cluster: <strong style={{ color: '#93c5fd' }}>{top.cluster_name}</strong></span>
                  <span>⏱ Duration: <strong style={{ color: '#fff' }}>{top.duration_hours} hrs</strong></span>
                  <span>📊 Difficulty: <strong style={{ color: '#fde68a', textTransform: 'capitalize' }}>{top.difficulty_level}</strong></span>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Composite Score</div>
                <div style={{ fontSize: '2rem', fontWeight: 800, color: '#a5b4fc', fontFamily: 'var(--font-mono)' }}>
                  {top.composite_score.toFixed(3)}
                </div>
              </div>
            </div>

            {/* Explainability Breakdown */}
            <div style={{ marginTop: '1.25rem', padding: '1rem', background: 'rgba(10, 14, 23, 0.6)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#a5b4fc', textTransform: 'uppercase', marginBottom: '0.35rem' }}>
                Decision Explanation & Rationale:
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
                {top.rationale}
              </p>
              <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', fontSize: '0.775rem', color: 'var(--text-secondary)' }}>
                <span>Relevance Weight (w_r,s): <strong>{top.relevance_weight.toFixed(2)}</strong></span>
                <span>Cluster Transfer Momentum: <strong>{(top.cluster_transfer_bonus * 100).toFixed(0)}%</strong></span>
                <span>Prerequisites: <strong style={{ color: top.prerequisite_fulfilled ? 'var(--color-success)' : 'var(--color-danger)' }}>{top.prerequisite_fulfilled ? 'Fulfilled' : 'Pending'}</strong></span>
              </div>
            </div>

            {/* Interactive Assessment Submission Bar */}
            <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Simulate Assessment Score:</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={customScore}
                  onChange={e => setCustomScore(e.target.value)}
                  className="input"
                  style={{ width: '90px', padding: '0.4rem 0.6rem' }}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>(Pass threshold $\ge 0.75$)</span>
              </div>

              <button
                className="btn btn-primary"
                onClick={() => handleCompleteModule(top.module_id)}
                disabled={completing}
              >
                {completing ? 'Updating Profile...' : '✓ Complete Module & Submit Assessment'}
              </button>
            </div>
          </div>

          {/* Alternative Ranked Modules */}
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1rem' }}>
            Alternative Eligible Training Modules
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {recData.alternative_modules.map((alt, idx) => (
              <div key={alt.module_id} className="glass-panel" style={{ padding: '1.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.5rem' }}>
                  <span className="badge badge-info">Rank #{idx + 2}</span>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', fontWeight: 700, color: '#a5b4fc' }}>
                    Score: {alt.composite_score.toFixed(2)}
                  </span>
                </div>
                <h4 style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
                  {alt.module_id}: {alt.skill_name}
                </h4>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                  Cluster: {alt.cluster_name} • {alt.duration_hours} hrs
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  {alt.rationale}
                </p>
                <button
                  className="btn btn-secondary"
                  style={{ width: '100%', fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
                  onClick={() => handleCompleteModule(alt.module_id)}
                >
                  Complete This Module Instead
                </button>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🎉</div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-success)' }}>Target Readiness Achieved!</h3>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            This employee has mastered all required occupation competencies or reached the required threshold.
          </p>
        </div>
      )}

      {/* Override Modal */}
      {showOverrideModal && (
        <div className="modal-overlay" onClick={() => setShowOverrideModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>🛡️ Human Manager Override</h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
              Override the algorithmic recommendation with a pedagogical or business justification. All overrides are permanently logged.
            </p>

            <form onSubmit={handleOverride} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Select Target Module</label>
                <select
                  className="select"
                  required
                  value={overrideModuleId}
                  onChange={e => setOverrideModuleId(e.target.value)}
                >
                  <option value="">-- Choose Module --</option>
                  {recData?.alternative_modules.map(a => (
                    <option key={a.module_id} value={a.module_id}>{a.module_id}: {a.skill_name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Reason for Override (Audit Log)</label>
                <textarea
                  className="input"
                  required
                  rows={3}
                  value={overrideReason}
                  onChange={e => setOverrideReason(e.target.value)}
                  placeholder="e.g. Employee has upcoming project deadline in this specific domain..."
                />
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowOverrideModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Submit Override</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
