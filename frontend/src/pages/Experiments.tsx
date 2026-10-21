import React, { useEffect, useState } from 'react';
import { api } from '../services/api';

export const Experiments: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'classifier' | 'policies' | 'ablations'>('classifier');
  const [classifierData, setClassifierData] = useState<any | null>(null);
  const [policyData, setPolicyData] = useState<any | null>(null);
  const [ablationData, setAblationData] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadMetrics() {
      setLoading(true);
      try {
        const [cData, pData, aData] = await Promise.all([
          api.getClassifierMetrics().catch(() => null),
          api.getPolicyMetrics().catch(() => null),
          api.getAblationMetrics().catch(() => null)
        ]);
        setClassifierData(cData);
        setPolicyData(pData);
        setAblationData(aData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadMetrics();
  }, []);

  return (
    <div className="experiments-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Research Experiments & Publication Artifacts</h1>
          <p className="page-subtitle">
            Reproduced paper baselines, circularity disclosure analysis, GroupKFold defensible evaluations, and systematic policy ablations.
          </p>
        </div>
        <a href="/api/experiments/export-csv" download className="btn btn-primary">
          📥 Export Ablation CSV
        </a>
      </div>

      {/* Tabs */}
      <div className="tabs-nav">
        <button
          className={`tab-btn ${activeSubTab === 'classifier' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('classifier')}
        >
          1. Critical-Gap Classifiers (Phase 6 & 7)
        </button>
        <button
          className={`tab-btn ${activeSubTab === 'policies' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('policies')}
        >
          2. Policy Evaluations & Wilcoxon Tests
        </button>
        <button
          className={`tab-btn ${activeSubTab === 'ablations' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('ablations')}
        >
          3. Ablation Studies & Sensitivity
        </button>
      </div>

      {loading ? (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading experimental metrics...</div>
      ) : (
        <>
          {/* TAB 1: CLASSIFIER */}
          {activeSubTab === 'classifier' && classifierData && (
            <div>
              {/* Circularity Disclosure Alert */}
              <div style={{ padding: '1.25rem', background: 'rgba(245, 158, 11, 0.1)', border: '1px solid rgba(245, 158, 11, 0.35)', borderRadius: 'var(--radius-md)', color: '#fde68a', marginBottom: '1.5rem', fontSize: '0.875rem', lineHeight: 1.6 }}>
                <strong>⚠️ Methodological Limitation & Circularity Disclosure:</strong><br />
                {classifierData.phase_6_paper_reproduction?.circularity_disclosure}
              </div>

              {/* Phase 6 Table */}
              <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  Phase 6: Paper Reproduction Baseline (70/30 Split on Top-1/3 Target)
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  Paper Reported Reference Random Forest F1: <strong>{classifierData.phase_6_paper_reproduction?.paper_reported_reference_rf_f1}</strong>
                </p>

                <table className="table">
                  <thead>
                    <tr>
                      <th>Model Architecture</th>
                      <th>Accuracy</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1-Score</th>
                      <th>ROC-AUC</th>
                      <th>PR-AUC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(classifierData.phase_6_paper_reproduction?.models || {}).map(([mName, m]: [string, any]) => (
                      <tr key={mName}>
                        <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{mName.replace(/_/g, ' ')}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(m.accuracy * 100).toFixed(2)}%</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{m.precision.toFixed(4)}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{m.recall.toFixed(4)}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#a5b4fc' }}>{m.f1_score.toFixed(4)}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{m.roc_auc.toFixed(4)}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{m.pr_auc.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  <strong>Random Forest Feature Importances:</strong>{' '}
                  {JSON.stringify(classifierData.phase_6_paper_reproduction?.random_forest_feature_importance)}
                </div>
              </div>

              {/* Phase 7 Table */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem', color: '#6ee7b7' }}>
                  Phase 7: Scientifically Defensible Evaluation (GroupKFold + Simulated Outcome)
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  Evaluated across 25 fold-seed iterations on a non-circular role-readiness impact target. Prevents cross-employee data leakage.
                </p>

                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>F1-Score (Mean ± 95% CI)</th>
                      <th>ROC-AUC (Mean)</th>
                      <th>PR-AUC (Mean)</th>
                      <th>Brier Loss (Calibration)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(classifierData.phase_7_improved_evaluation?.metrics_with_95_ci || {}).map(([mName, mStats]: [string, any]) => (
                      <tr key={mName}>
                        <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{mName.replace(/_/g, ' ')}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: mName === 'random_forest' ? '#6ee7b7' : undefined }}>
                          {mStats.f1.mean} [{mStats.f1.ci_95_lower}, {mStats.f1.ci_95_upper}]
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{mStats.roc_auc.mean}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{mStats.pr_auc.mean}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{mStats.brier.mean}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 2: POLICIES */}
          {activeSubTab === 'policies' && policyData && (
            <div>
              <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  Policy Comparison (Matched Cohorts Reaching {(policyData.target_readiness * 100).toFixed(0)}% Readiness)
                </h3>

                <table className="table" style={{ marginBottom: '1.5rem' }}>
                  <thead>
                    <tr>
                      <th>Training Policy</th>
                      <th>Mean Steps</th>
                      <th>Median Steps</th>
                      <th>Std Dev</th>
                      <th>Success Rate</th>
                      <th>Mean Learning Hours</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(policyData.policies || {}).map(([pName, p]: [string, any]) => (
                      <tr key={pName} style={{ background: pName === 'adaptive' ? 'rgba(99, 102, 241, 0.1)' : undefined }}>
                        <td style={{ fontWeight: 700, textTransform: 'capitalize', color: pName === 'adaptive' ? '#a5b4fc' : undefined }}>
                          Policy {pName === 'random' ? 'A (Random)' : pName === 'static' ? 'B (Static Relevance)' : 'C (Adaptive Graph)'}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{p.steps_mean}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{p.steps_median}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{p.steps_std}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(p.success_rate * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{p.mean_learning_hours} hrs</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h4 style={{ fontSize: '0.95rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  Paired Wilcoxon Signed-Rank Tests
                </h4>
                <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
                  {Object.entries(policyData.paired_statistical_tests || {}).map(([comp, t]: [string, any]) => (
                    <div key={comp} style={{ background: 'rgba(10, 14, 23, 0.5)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', fontSize: '0.8rem' }}>
                      <strong style={{ textTransform: 'capitalize' }}>{comp.replace(/_/g, ' ')}:</strong>{' '}
                      <span>p = {t.p_value < 0.001 ? '< 0.001' : t.p_value.toFixed(4)}</span> |{' '}
                      <span>Cohen's d = {t.cohens_d}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Published Paper Reference Card */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                  Published Paper Reference Values Comparison
                </h4>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  Paper Reported Means: Random = 42.66 steps, Static = 21.75 steps, Adaptive = 24.24 steps.<br />
                  <em>Note:</em> {policyData.paper_references_comparison?.notes}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: ABLATIONS */}
          {activeSubTab === 'ablations' && ablationData && (
            <div>
              {/* 5 Policy Variants Table */}
              <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '1rem' }}>
                  Ablation Study: 5 Policy Variants
                </h3>

                <table className="table">
                  <thead>
                    <tr>
                      <th>Variant</th>
                      <th>Mean Steps to 80%</th>
                      <th>Median Steps</th>
                      <th>Std Dev</th>
                      <th>Mean Hours</th>
                      <th>Success Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ablationData.policy_ablations || {}).map(([vName, v]: [string, any]) => (
                      <tr key={vName}>
                        <td style={{ fontWeight: 600 }}>{vName}</td>
                        <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{v.mean_steps}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{v.median_steps}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{v.std_steps}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{v.mean_hours} hrs</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(v.success_rate * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Sensitivity Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem' }}>
                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>
                    Sensitivity: Cluster Transfer Strength ($\beta$)
                  </h4>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>$\beta$ Value</th>
                        <th>Mean Steps</th>
                        <th>Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ablationData.sensitivity_cluster_strength?.map((row: any) => (
                        <tr key={row.cluster_bonus_weight_beta}>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{row.cluster_bonus_weight_beta.toFixed(2)}</td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{row.mean_steps}</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{row.std_steps}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="glass-panel" style={{ padding: '1.5rem' }}>
                  <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>
                    Sensitivity: Readiness Target Threshold
                  </h4>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Target Threshold</th>
                        <th>Mean Steps</th>
                        <th>Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ablationData.sensitivity_readiness_target?.map((row: any) => (
                        <tr key={row.target_readiness}>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{(row.target_readiness * 100).toFixed(0)}%</td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{row.mean_steps}</td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>{row.std_steps}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
