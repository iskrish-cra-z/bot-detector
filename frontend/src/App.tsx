import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Sidebar, TopAppBar } from './components/Navigation';
import NeuralHub from './pages/NeuralHub';
import NetworkGraph from './pages/NetworkGraph';
import ForensicDetail from './pages/ForensicDetail';
import Archive from './pages/Archive';
import Settings from './pages/Settings';
import { ScanProvider } from './context/ScanContext';
import { Toaster } from 'react-hot-toast';

const App = () => {
  return (
    <ScanProvider>
      <Toaster position="top-right" />
      <Router>
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <div className="flex-1 flex flex-col relative md:ml-64 relative z-0">
            <TopAppBar />
            <div className="flex-1 pt-16 overflow-y-auto w-full custom-scrollbar">
              <Routes>
                <Route path="/" element={<NeuralHub />} />
                <Route path="/graph" element={<NetworkGraph />} />
                <Route path="/node/:id" element={<ForensicDetail />} />
                <Route path="/archive" element={<Archive />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </div>
          </div>
        </div>
      </Router>
    </ScanProvider>
  );
};

export default App;
