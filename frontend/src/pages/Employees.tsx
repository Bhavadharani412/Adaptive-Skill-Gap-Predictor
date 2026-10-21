import React, { useEffect, useState } from 'react';
import { api } from '../services/api';
import { Employee, Occupation } from '../types';

interface EmployeesProps {
  onSelectEmployeeForGaps?: (empId: string) => void;
  onSelectEmployeeForRecs?: (empId: string) => void;
}

export const Employees: React.FC<EmployeesProps> = ({
  onSelectEmployeeForGaps,
  onSelectEmployeeForRecs
}) => {
  const [employees, setEmployees] = useState<Employee[]>([]);
  const [occupations, setOccupations] = useState<Occupation[]>([]);
  const [selectedOccupation, setSelectedOccupation] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [selectedEmpDetail, setSelectedEmpDetail] = useState<Employee | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [showAddModal, setShowAddModal] = useState<boolean>(false);

  // Form state
  const [newName, setNewName] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newOccId, setNewOccId] = useState('OCC-001');
  const [newDept, setNewDept] = useState('Engineering');
  const [formError, setFormError] = useState('');

  useEffect(() => {
    loadData();
  }, [selectedOccupation]);

  async function loadData() {
    setLoading(true);
    try {
      const [empList, occList] = await Promise.all([
        api.getEmployees(selectedOccupation || undefined, 100),
        api.getOccupations()
      ]);
      setEmployees(empList);
      setOccupations(occList);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function handleSelectEmployee(empId: string) {
    try {
      const detail = await api.getEmployeeDetail(empId);
      setSelectedEmpDetail(detail);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleUpdateSkill(skillId: string, state: string, scoreStr: string) {
    if (!selectedEmpDetail) return;
    const score = scoreStr ? parseFloat(scoreStr) : undefined;
    try {
      await api.updateSkillState(selectedEmpDetail.employee_id, skillId, state, score);
      // Reload detail
      handleSelectEmployee(selectedEmpDetail.employee_id);
      loadData();
    } catch (err) {
      alert('Failed to update skill: ' + err);
    }
  }

  async function handleCreateEmployee(e: React.FormEvent) {
    e.preventDefault();
    setFormError('');
    try {
      await api.createEmployee({
        name: newName,
        email: newEmail,
        occupation_id: newOccId,
        department: newDept
      });
      setShowAddModal(false);
      setNewName('');
      setNewEmail('');
      loadData();
    } catch (err: any) {
      setFormError(err.message || 'Error creating employee');
    }
  }

  const filtered = employees.filter(emp =>
    emp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    emp.employee_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (emp.occupation_title && emp.occupation_title.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className="employees-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Employee Cohort Management</h1>
          <p className="page-subtitle">Inspect onboarding cohorts, assign occupational roles, and calibrate skill diagnostic states.</p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowAddModal(true)}>
          + Add New Employee
        </button>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-panel" style={{ padding: '1.25rem', marginBottom: '1.5rem', display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 250px' }}>
          <input
            className="input"
            type="text"
            placeholder="🔍 Search employee by name, ID or role..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div style={{ flex: '0 0 280px' }}>
          <select
            className="select"
            value={selectedOccupation}
            onChange={(e) => setSelectedOccupation(e.target.value)}
          >
            <option value="">All Occupations ({occupations.length})</option>
            {occupations.map(o => (
              <option key={o.occupation_id} value={o.occupation_id}>{o.title}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Employee List + Detail Drawer */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedEmpDetail ? '1.2fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="glass-panel table-container">
          {loading ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>Loading employee cohort...</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Employee</th>
                  <th>Role</th>
                  <th>Readiness</th>
                  <th>Gaps</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(emp => (
                  <tr
                    key={emp.employee_id}
                    style={{
                      cursor: 'pointer',
                      background: selectedEmpDetail?.employee_id === emp.employee_id ? 'rgba(99, 102, 241, 0.12)' : undefined
                    }}
                    onClick={() => handleSelectEmployee(emp.employee_id)}
                  >
                    <td>
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{emp.name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{emp.employee_id} • {emp.department}</div>
                    </td>
                    <td>
                      <span className="badge badge-purple">{emp.occupation_title || emp.occupation_id}</span>
                    </td>
                    <td style={{ minWidth: '130px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '0.2rem' }}>
                        <span>{Math.round(emp.readiness * 100)}%</span>
                        <span style={{ color: emp.readiness >= 0.8 ? 'var(--color-success)' : 'var(--text-muted)' }}>
                          {emp.readiness >= 0.8 ? 'Ready' : 'In Training'}
                        </span>
                      </div>
                      <div className="progress-bar-bg">
                        <div
                          className="progress-bar-fill"
                          style={{
                            width: `${Math.min(100, Math.round(emp.readiness * 100))}%`,
                            background: emp.readiness >= 0.8 ? 'linear-gradient(135deg, #10b981, #059669)' : undefined
                          }}
                        />
                      </div>
                    </td>
                    <td>
                      <span className="badge badge-warning">{emp.total_gaps_count} Gaps</span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.4rem' }}>
                        {onSelectEmployeeForGaps && (
                          <button
                            className="btn btn-secondary"
                            style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectEmployeeForGaps(emp.employee_id);
                            }}
                          >
                            🎯 Gaps
                          </button>
                        )}
                        {onSelectEmployeeForRecs && (
                          <button
                            className="btn btn-primary"
                            style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
                            onClick={(e) => {
                              e.stopPropagation();
                              onSelectEmployeeForRecs(emp.employee_id);
                            }}
                          >
                            💡 Train
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Selected Employee Profile Drawer */}
        {selectedEmpDetail && (
          <div className="glass-panel" style={{ padding: '1.5rem', maxHeight: '80vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
              <div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>{selectedEmpDetail.name}</h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                  {selectedEmpDetail.employee_id} • {selectedEmpDetail.email}
                </p>
                <div style={{ marginTop: '0.35rem' }}>
                  <span className="badge badge-purple">{selectedEmpDetail.occupation_title}</span>
                </div>
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
                onClick={() => setSelectedEmpDetail(null)}
              >
                ✕ Close
              </button>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ flex: 1, background: 'rgba(17, 24, 39, 0.6)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Readiness</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#a5b4fc' }}>
                  {Math.round(selectedEmpDetail.readiness * 100)}%
                </div>
              </div>
              <div style={{ flex: 1, background: 'rgba(17, 24, 39, 0.6)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Mastered</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-success)' }}>
                  {selectedEmpDetail.mastered_count}
                </div>
              </div>
              <div style={{ flex: 1, background: 'rgba(17, 24, 39, 0.6)', padding: '0.75rem', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Open Gaps</div>
                <div style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-warning)' }}>
                  {selectedEmpDetail.total_gaps_count}
                </div>
              </div>
            </div>

            <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.75rem' }}>
              Assessment State & Competency Portfolio
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {selectedEmpDetail.skills?.map(s => (
                <div
                  key={s.skill_id}
                  style={{
                    background: 'rgba(17, 24, 39, 0.5)',
                    padding: '0.65rem 0.85rem',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '0.5rem'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{s.skill_name || s.skill_id}</div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      ID: {s.skill_id} {s.assessment_score !== null && s.assessment_score !== undefined ? `• Score: ${(s.assessment_score * 100).toFixed(0)}%` : ''}
                    </div>
                  </div>

                  <select
                    className="select"
                    style={{ width: 'auto', padding: '0.25rem 0.5rem', fontSize: '0.75rem' }}
                    value={s.state}
                    onChange={(e) => handleUpdateSkill(s.skill_id, e.target.value, s.assessment_score ? String(s.assessment_score) : '0.85')}
                  >
                    <option value="not_assessed">Not Assessed</option>
                    <option value="not_mastered">Not Mastered</option>
                    <option value="in_progress">In Progress</option>
                    <option value="mastered">Mastered</option>
                  </select>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Add Employee Modal */}
      {showAddModal && (
        <div className="modal-overlay" onClick={() => setShowAddModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>Onboard New Employee</h3>
            {formError && (
              <div style={{ padding: '0.75rem', background: 'rgba(239, 68, 68, 0.2)', border: '1px solid rgba(239, 68, 68, 0.4)', borderRadius: 'var(--radius-sm)', color: '#fca5a5', fontSize: '0.85rem', marginBottom: '1rem' }}>
                {formError}
              </div>
            )}
            <form onSubmit={handleCreateEmployee} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Full Name</label>
                <input className="input" required value={newName} onChange={e => setNewName(e.target.value)} placeholder="e.g. Taylor Morgan" />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Email</label>
                <input className="input" type="email" required value={newEmail} onChange={e => setNewEmail(e.target.value)} placeholder="e.g. taylor.m@corp.asgp.internal" />
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Assigned Occupation</label>
                <select className="select" value={newOccId} onChange={e => setNewOccId(e.target.value)}>
                  {occupations.map(o => (
                    <option key={o.occupation_id} value={o.occupation_id}>{o.title} ({o.occupation_id})</option>
                  ))}
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Department</label>
                <input className="input" value={newDept} onChange={e => setNewDept(e.target.value)} placeholder="Engineering & Technology" />
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '0.5rem' }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowAddModal(false)}>Cancel</button>
                <button type="submit" className="btn btn-primary">Create Profile</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
