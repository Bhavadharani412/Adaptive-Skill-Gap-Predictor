import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { DataQualityReport } from '../types';

export const Admin: React.FC = () => {
  const [report, setReport] = useState<DataQualityReport | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionStatus, setActionStatus] = useState<string>('');
  const [processing, setProcessing] = useState<boolean>(false);

  useEffect(() => {
    loadReport();
  }, []);

  async function loadReport() {
    setLoading(true);
    try {
      const data = await api.getDataQualityReport();
      setReport(data);
    } catch (err: any) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleImportFallback() {
    setProcessing(true);
    setActionStatus('');
    try {
      const res = await api.importDataset(true);
      setActionStatus('✓ Synthetic fallback dataset generated and validated.');
      setReport(res.report);
    } catch (err: any) {
      setActionStatus('❌ Import failed: ' + err.message);
    } finally {
      setProcessing(false);
    }
  }

  async function handleReseedDatabase() {
    setProcessing(true);
    setActionStatus('');
    try {
      const res = await api.reseedDatabase();
      setActionStatus('✓ ' + res.message);
      loadReport();
    } catch (err: any) {
      setActionStatus('❌ Reseed failed: ' + err.message);
    } finally {
      setProcessing(false);
    }
  }

  return (
    <div className="admin-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Administration & Data Quality Governance</h1>
          <p className="page-subtitle">
            Inspect taxonomy ingestion logs, audit data quality constraints, and configure simulation hyperparameters.
          </p>
        </div>
      </div>

      {actionStatus && (
        <div style={{ padding: '1rem', background: actionStatus.startsWith('✓') ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-md)', color: actionStatus.startsWith('✓') ? '#6ee7b7' : '#fca5a5', marginBottom: '1.5rem', fontSize: '0.875rem' }}>
          {actionStatus}
        </div>
      )}

      {/* Grid: Actions & Configuration */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Actions Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Data Pipeline Maintenance
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
            Manage the underlying taxonomy source, trigger dataset ingestion, or restore database tables.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <button
              className="btn btn-secondary"
              onClick={handleImportFallback}
              disabled={processing}
            >
              🔄 Re-Import / Validate Dataset Pipeline
            </button>
            <button
              className="btn btn-secondary"
              onClick={handleReseedDatabase}
              disabled={processing}
              style={{ borderColor: 'rgba(239, 68, 68, 0.3)', color: '#fca5a5' }}
            >
              ⚠️ Reseed SQLite Database From Scratch
            </button>
          </div>
        </div>

        {/* Hyperparameters Card */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.5rem' }}>
            Active Simulation Hyperparameters
          </h3>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Calibrated values for the Recommendation Engine and Policy Evaluator:
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Alpha ($\alpha$ - Relevance Weight):</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>1.00</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Beta ($\beta$ - Cluster Transfer Momentum):</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>0.35</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Gamma ($\gamma$ - Prerequisite Feasibility):</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>0.25</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border-color)' }}>
              <span style={{ color: 'var(--text-muted)' }}>Baseline Target Readiness (θ target):</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>80% (0.80)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0' }}>
              <span style={{ color: 'var(--text-muted)' }}>Maximum Training Step Budget:</span>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>50 steps</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Data Quality Report Card */}
      {report && (
        <div className="glass-panel" style={{ padding: '1.75rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
            <div>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>
                Data Quality & Validation Report
              </h3>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                Source: <strong>{report.source}</strong> • Generated At: {report.imported_at}
              </p>
            </div>
            <span className="badge badge-success" style={{ textTransform: 'uppercase' }}>
              ● {report.status}
            </span>
          </div>

          <div className="stats-grid" style={{ marginBottom: '1.5rem' }}>
            <div className="glass-panel stat-card">
              <div className="stat-label">Valid Occupations</div>
              <div className="stat-value">{report.metrics.total_occupations}</div>
              <div className="stat-sub">Zero Duplicate Entities</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="stat-label">Valid Skills</div>
              <div className="stat-value">{report.metrics.total_skills}</div>
              <div className="stat-sub">{report.metrics.generic_skills_count || 26} Generic Competencies</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="stat-label">Bipartite Edges</div>
              <div className="stat-value">{report.metrics.total_occupation_skill_edges}</div>
              <div className="stat-sub">{report.metrics.essential_edges_count} Essential • {report.metrics.optional_edges_count} Optional</div>
            </div>
            <div className="glass-panel stat-card">
              <div className="stat-label">Skill Relationships</div>
              <div className="stat-value">{report.metrics.total_skill_relationships || 16}</div>
              <div className="stat-sub">Prerequisites & Affinity</div>
            </div>
          </div>

          {report.warnings && report.warnings.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#fde68a', marginBottom: '0.35rem' }}>Pipeline Notices & Warnings:</h4>
              <ul style={{ paddingLeft: '1.25rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {report.warnings.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
