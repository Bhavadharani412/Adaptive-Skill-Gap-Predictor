import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { Employee, SimulationEpisodeResponse } from '../types';

export const Simulator: React.FC = () => {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [selectedEmpId, setSelectedEmpId] = useState<string>('EMP-0001');
  const [policy, setPolicy] = useState<string>('adaptive');
  const [targetReadiness, setTargetReadiness] = useState<number>(0.80);
  const [budget, setBudget] = useState<number>(50);

  // Single episode result
  const [episodeResult, setEpisodeResult] = useState<SimulationEpisodeResponse | null>(null);
  const [simulatingEpisode, setSimulatingEpisode] = useState<boolean>(false);

  // Cohort comparison state
  const [cohortSize, setCohortSize] = useState<number>(60);
  const [cohortResult, setCohortResult] = useState<any | null>(null);
  const [simulatingCohort, setSimulatingCohort] = useState<boolean>(false);

  useEffect(() => {
    api.getEmployees(undefined, 100).then(res => {
      setEmployees(res);
      if (res.length > 0) setSelectedEmpId(res[0].employee_id);
    });
  }, []);

  async function handleRunEpisode() {
    setSimulatingEpisode(true);
    try {
      const res = await api.simulateEpisode(selectedEmpId, policy, targetReadiness, budget);
      setEpisodeResult(res);
    } catch (err: any) {
      alert('Error running simulation: ' + err.message);
    } finally {
      setSimulatingEpisode(false);
    }
  }

  async function handleRunCohort() {
    setSimulatingCohort(true);
    try {
      const res = await api.compareCohort(cohortSize, targetReadiness, budget);
      setCohortResult(res);
    } catch (err: any) {
      alert('Error running cohort simulation: ' + err.message);
    } finally {
      setSimulatingCohort(false);
    }
  }

  return (
    <div className="simulator-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Training Policy Simulator</h1>
          <p className="page-subtitle">
            Simulate training episodes and run matched cohort experiments contrasting Policy A (Random), Policy B (Static), and Policy C (Adaptive).
          </p>
        </div>
      </div>

      {/* Simulator Modes Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Section 1: Single Episode Simulator */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1rem', color: '#a5b4fc' }}>
            ⚡ Single Employee Episode Simulator
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>EMPLOYEE</label>
              <select className="select" value={selectedEmpId} onChange={e => setSelectedEmpId(e.target.value)}>
                {employees.map(e => (
                  <option key={e.employee_id} value={e.employee_id}>
                    {e.employee_id}: {e.name} ({e.occupation_title})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>RECOMMENDATION POLICY</label>
              <select className="select" value={policy} onChange={e => setPolicy(e.target.value)}>
                <option value="adaptive">Policy C: Adaptive Graph Engine</option>
                <option value="static">Policy B: Static Relevance Baseline</option>
                <option value="random">Policy A: Random Baseline</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '1rem' }}>
              <div style={{ flex: 1 }}>
                <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>TARGET READINESS</label>
                <select className="select" value={targetReadiness} onChange={e => setTargetReadiness(parseFloat(e.target.value))}>
                  <option value={0.70}>70%</option>
                  <option value={0.80}>80% (Baseline Target)</option>
                  <option value={0.90}>90%</option>
                  <option value={1.00}>100% (Complete Mastery)</option>
                </select>
              </div>

              <div style={{ flex: 1 }}>
                <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>MAX BUDGET (STEPS)</label>
                <input className="input" type="number" value={budget} onChange={e => setBudget(parseInt(e.target.value) || 50)} />
              </div>
            </div>

            <button
              className="btn btn-primary"
              style={{ marginTop: '0.5rem' }}
              onClick={handleRunEpisode}
              disabled={simulatingEpisode}
            >
              {simulatingEpisode ? 'Simulating Training Episode...' : '▶ Run Simulation Episode'}
            </button>
          </div>
        </div>

        {/* Section 2: Matched Cohort Comparator */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1rem', color: '#6ee7b7' }}>
            👥 Matched Cohort Comparative Benchmark
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Runs Policies A, B, and C simultaneously on identical starting employee states, computing mean steps, learning hours, and Wilcoxon signed-rank tests.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            <div>
              <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>COHORT SAMPLE SIZE</label>
              <input
                className="input"
                type="number"
                min="20"
                max="200"
                value={cohortSize}
                onChange={e => setCohortSize(parseInt(e.target.value) || 50)}
              />
            </div>

            <button
              className="btn btn-secondary"
              style={{ marginTop: '1.25rem', borderColor: 'rgba(16, 185, 129, 0.4)', color: '#6ee7b7' }}
              onClick={handleRunCohort}
              disabled={simulatingCohort}
            >
              {simulatingCohort ? 'Running Cohort Benchmark (Please Wait)...' : '📊 Run Matched Cohort Benchmark'}
            </button>
          </div>
        </div>
      </div>

      {/* Cohort Comparative Results */}
      {cohortResult && (
        <div className="glass-panel" style={{ padding: '1.75rem', marginBottom: '2rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>
            Matched Cohort Benchmark Results (Target: {(cohortResult.target_readiness * 100).toFixed(0)}%)
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
            {Object.entries(cohortResult.policies).map(([pName, pStats]: [string, any]) => (
              <div
                key={pName}
                style={{
                  background: pName === 'adaptive' ? 'rgba(99, 102, 241, 0.15)' : 'rgba(17, 24, 39, 0.6)',
                  border: pName === 'adaptive' ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  padding: '1.25rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 700, textTransform: 'uppercase', color: pName === 'adaptive' ? '#a5b4fc' : '#f8fafc' }}>
                    {pName} Policy
                  </span>
                  <span className="badge badge-success">{(pStats.success_rate * 100).toFixed(0)}% Success</span>
                </div>
                <div style={{ fontSize: '1.75rem', fontWeight: 800, margin: '0.5rem 0', color: pName === 'adaptive' ? '#a5b4fc' : undefined }}>
                  {pStats.steps_mean} <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>mean steps</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <div>Median: <strong>{pStats.steps_median}</strong> • Std: <strong>{pStats.steps_std}</strong></div>
                  <div>Mean Learning: <strong>{pStats.mean_learning_hours} hrs</strong></div>
                  <div>Final Readiness: <strong>{(pStats.mean_final_readiness * 100).toFixed(1)}%</strong></div>
                </div>
              </div>
            ))}
          </div>

          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Paired Wilcoxon Signed-Rank Significance Tests
          </h4>
          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {Object.entries(cohortResult.paired_statistical_tests).map(([comp, tData]: [string, any]) => (
              <div key={comp} style={{ background: 'rgba(10, 14, 23, 0.5)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
                <strong style={{ textTransform: 'capitalize' }}>{comp.replace(/_/g, ' ')}:</strong>{' '}
                <span>p = {tData.p_value < 0.001 ? '< 0.001' : tData.p_value.toFixed(4)}</span> |{' '}
                <span>Cohen's d = {tData.cohens_d}</span>{' '}
                <span className="badge badge-success" style={{ marginLeft: '0.5rem' }}>{tData.significant_at_05 ? 'Significant (p < 0.05)' : 'Not Significant'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Episode Result Step-by-Step Log */}
      {episodeResult && (
        <div className="glass-panel" style={{ padding: '1.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>
                Simulation Episode Journey ({episodeResult.policy.toUpperCase()} Policy)
              </h3>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                Target: {(episodeResult.target_readiness * 100).toFixed(0)}% •
                Initial: {(episodeResult.initial_readiness * 100).toFixed(1)}% ➔
                Final: <strong style={{ color: 'var(--color-success)' }}>{(episodeResult.final_readiness * 100).toFixed(1)}%</strong> •
                Total Hours: {episodeResult.total_learning_hours} hrs
              </p>
            </div>
            <span className="badge badge-success" style={{ fontSize: '0.85rem' }}>
              ✓ Target Achieved in {episodeResult.steps_to_target} Steps
            </span>
          </div>

          <table className="table">
            <thead>
              <tr>
                <th>Step #</th>
                <th>Target Module</th>
                <th>Assessment Result</th>
                <th>Score</th>
                <th>Readiness Progress</th>
                <th>Step Rationale</th>
              </tr>
            </thead>
            <tbody>
              {episodeResult.history.map(step => (
                <tr key={step.step}>
                  <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>Step {step.step}</td>
                  <td>
                    <div style={{ fontWeight: 600 }}>{step.module_id}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Skill: {step.target_skill_id}</div>
                  </td>
                  <td>
                    {step.passed ? (
                      <span className="badge badge-success">Passed (Mastered)</span>
                    ) : (
                      <span className="badge badge-warning">Retained In Progress</span>
                    )}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{(step.score * 100).toFixed(0)}%</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '120px' }}>
                      <div className="progress-bar-bg" style={{ flex: 1 }}>
                        <div className="progress-bar-fill" style={{ width: `${Math.round(step.readiness_after_step * 100)}%` }} />
                      </div>
                      <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                        {(step.readiness_after_step * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', maxWidth: '300px' }}>
                    {step.rationale}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
