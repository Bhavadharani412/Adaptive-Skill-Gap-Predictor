import React from 'react';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'employees', label: 'Employees', icon: '👥' },
    { id: 'gaps', label: 'Skill Gaps', icon: '🎯' },
    { id: 'recommendations', label: 'Recommendations', icon: '💡' },
    { id: 'simulator', label: 'Simulator', icon: '⚡' },
    { id: 'experiments', label: 'Experiments', icon: '🧪' },
    { id: 'admin', label: 'Data & Admin', icon: '⚙️' }
  ];

  return (
    <nav className="navbar">
      <div className="brand" onClick={() => setActiveTab('dashboard')}>
        <span>⚡ ASGP</span>
        <span className="brand-badge">Adaptive Engine v1.0</span>
      </div>

      <div className="nav-links">
        {navItems.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
            onClick={() => setActiveTab(item.id)}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </div>
    </nav>
  );
};
