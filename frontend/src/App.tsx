import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Dashboard } from './pages/Dashboard';
import { Employees } from './pages/Employees';
import { GapAnalysis } from './pages/GapAnalysis';
import { Recommendations } from './pages/Recommendations';
import { Simulator } from './pages/Simulator';
import { Experiments } from './pages/Experiments';
import { Admin } from './pages/Admin';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<string>('EMP-0001');

  function handleNavigateToGaps(empId: string) {
    setSelectedEmployeeId(empId);
    setActiveTab('gaps');
  }

  function handleNavigateToRecs(empId: string) {
    setSelectedEmployeeId(empId);
    setActiveTab('recommendations');
  }

  return (
    <div className="app-root">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="app-container">
        {activeTab === 'dashboard' && (
          <Dashboard onNavigate={setActiveTab} />
        )}

        {activeTab === 'employees' && (
          <Employees
            onSelectEmployeeForGaps={handleNavigateToGaps}
            onSelectEmployeeForRecs={handleNavigateToRecs}
          />
        )}

        {activeTab === 'gaps' && (
          <GapAnalysis
            initialEmployeeId={selectedEmployeeId}
            onNavigateToRecommendations={handleNavigateToRecs}
          />
        )}

        {activeTab === 'recommendations' && (
          <Recommendations
            initialEmployeeId={selectedEmployeeId}
          />
        )}

        {activeTab === 'simulator' && (
          <Simulator />
        )}

        {activeTab === 'experiments' && (
          <Experiments />
        )}

        {activeTab === 'admin' && (
          <Admin />
        )}
      </main>
    </div>
  );
};

export default App;
