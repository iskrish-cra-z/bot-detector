import React from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import clsx from 'clsx';
import { Network, LineChart, History, Settings, Bell, Terminal, Search, Loader2 } from 'lucide-react';
import { useScan } from '../context/ScanContext';

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const isActive = (path: string) => location.pathname === path;
  const { isScanning, triggerScan } = useScan();

  return (
    <nav className="fixed left-0 top-0 h-full z-50 flex flex-col bg-slate-950/60 backdrop-blur-xl w-20 md:w-64 border-r border-slate-800/50 shadow-[0_0_40px_rgba(0,245,255,0.08)]">
      <div className="p-6">
        <h1 className="text-xl font-black text-cyan-400 tracking-tighter font-headline">NEURAL_LAB</h1>
        <p className="text-[10px] font-medium tracking-widest uppercase font-headline text-slate-500 mt-1">GAT-V3 ACTIVE</p>
      </div>
      
      <div className="flex-1 mt-8 space-y-2">
        <NavLink to="/" className={clsx("flex items-center px-6 py-4 transition-all duration-300 active:scale-95 active:opacity-80", isActive("/") ? "text-cyan-400 border-l-2 border-cyan-400 bg-cyan-400/5" : "text-slate-500 hover:bg-slate-800/50 hover:text-cyan-300")}>
          <LineChart className="w-6 h-6 mr-4" />
          <span className="hidden md:block text-[10px] font-medium tracking-widest uppercase font-headline">Neural Hub</span>
        </NavLink>
        <NavLink to="/graph" className={clsx("flex items-center px-6 py-4 transition-all duration-300 active:scale-95 active:opacity-80", isActive("/graph") ? "text-cyan-400 border-l-2 border-cyan-400 bg-cyan-400/5" : "text-slate-500 hover:bg-slate-800/50 hover:text-cyan-300")}>
          <Network className="w-6 h-6 mr-4" />
          <span className="hidden md:block text-[10px] font-medium tracking-widest uppercase font-headline">Network Graph</span>
        </NavLink>
        <NavLink to="/archive" className={clsx("flex items-center px-6 py-4 transition-all duration-300 active:scale-95 active:opacity-80", isActive("/archive") ? "text-cyan-400 border-l-2 border-cyan-400 bg-cyan-400/5" : "text-slate-500 hover:bg-slate-800/50 hover:text-cyan-300")}>
          <History className="w-6 h-6 mr-4" />
          <span className="hidden md:block text-[10px] font-medium tracking-widest uppercase font-headline">Archive</span>
        </NavLink>
        <NavLink to="/settings" className={clsx("flex items-center px-6 py-4 transition-all duration-300 active:scale-95 active:opacity-80", isActive("/settings") ? "text-cyan-400 border-l-2 border-cyan-400 bg-cyan-400/5" : "text-slate-500 hover:bg-slate-800/50 hover:text-cyan-300")}>
          <Settings className="w-6 h-6 mr-4" />
          <span className="hidden md:block text-[10px] font-medium tracking-widest uppercase font-headline">Settings</span>
        </NavLink>
      </div>
      
      <div className="p-6">
        <button 
          onClick={triggerScan}
          disabled={isScanning}
          className={clsx(
            "w-full py-3 flex justify-center items-center text-[10px] font-bold tracking-[0.2em] font-headline transition-colors",
            isScanning 
              ? "bg-slate-800 border bg-cyan-400/20 text-cyan-400 cursor-not-allowed border-cyan-400/50"
              : "bg-cyan-400/10 border border-cyan-400/50 text-cyan-400 hover:bg-cyan-400 hover:text-slate-950"
          )}
        >
          {isScanning ? <Loader2 className="w-4 h-4 animate-spin mr-2"/> : null}
          {isScanning ? "SCANNING..." : "RUN SCAN"}
        </button>
      </div>
    </nav>
  );
};

export const TopAppBar: React.FC = () => {
  const navigate = useNavigate();

  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const val = e.currentTarget.value.trim();
      if (val) {
        navigate(`/node/${val}`);
        e.currentTarget.value = '';
      }
    }
  };

  return (
    <header className="fixed top-0 right-0 left-0 md:left-64 flex justify-between items-center px-6 h-16 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/40">
      <div className="flex items-center space-x-8">
        <span className="text-lg font-bold text-cyan-400 tracking-[0.2em] font-headline">FORENSIC_HUD</span>
        <nav className="hidden lg:flex space-x-6">
          <NavLink to="/" className={({isActive}) => isActive ? "text-cyan-400 font-bold border-b-2 border-cyan-400 text-xs tracking-widest font-headline pb-1" : "text-slate-400 hover:text-cyan-300 transition-colors text-xs font-bold tracking-widest font-headline pb-1"}>LIVE_FEED</NavLink>
          <NavLink to="/graph" className={({isActive}) => isActive ? "text-cyan-400 font-bold border-b-2 border-cyan-400 text-xs tracking-widest font-headline pb-1" : "text-slate-400 hover:text-cyan-300 transition-colors text-xs font-bold tracking-widest font-headline pb-1"}>THREAT_MAP</NavLink>
          <NavLink to="/archive" className={({isActive}) => isActive ? "text-cyan-400 font-bold border-b-2 border-cyan-400 text-xs tracking-widest font-headline pb-1" : "text-slate-400 hover:text-cyan-300 transition-colors text-xs font-bold tracking-widest font-headline pb-1"}>LAB_REPORTS</NavLink>
        </nav>
      </div>
      <div className="flex items-center space-x-4">
        <div className="relative hidden sm:block">
          <input 
            type="text" 
            placeholder="SEARCH_NODE..." 
            onKeyDown={handleSearch}
            className="bg-surface-container-lowest border border-outline-variant/30 text-[10px] tracking-widest font-headline px-4 py-1.5 w-48 focus:outline-none focus:border-cyan-400/50 transition-all text-cyan-400" 
          />
          <Search className="absolute right-2 top-1.5 text-slate-500 w-4 h-4 cursor-pointer" />
        </div>
        <Bell 
          onClick={() => toast("No active threat alerts.", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }})}
          className="w-5 h-5 text-slate-400 cursor-pointer hover:text-cyan-300 transition-colors" 
        />
        <Terminal 
          onClick={() => toast("Terminal logs are available in NeuralHub view.", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }})}
          className="w-5 h-5 text-slate-400 cursor-pointer hover:text-cyan-300 transition-colors" 
        />
        <div className="w-8 h-8 rounded-full border border-cyan-400/30 bg-slate-800 overflow-hidden ml-2 flex justify-center items-center font-bold text-slate-400 text-xs tracking-tighter">OP</div>
      </div>
    </header>
  );
};
