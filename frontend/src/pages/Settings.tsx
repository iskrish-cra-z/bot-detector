import React from 'react';
import { Settings as SettingsIcon, Database, Cpu, Lock } from 'lucide-react';
import toast from 'react-hot-toast';

const Settings: React.FC = () => {
  const handleSave = () => {
    toast.success("Settings applied to cluster successfully.", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }});
  };

  return (
    <main className="p-6 md:p-10 max-w-5xl mx-auto min-h-screen">
      <div className="flex items-center justify-between mb-8">
        <div>
           <h2 className="text-3xl font-black font-headline text-cyan-400 tracking-tighter uppercase flex items-center gap-3">
              <SettingsIcon className="w-8 h-8"/>
              System Configuration
           </h2>
           <p className="text-slate-500 text-xs font-bold tracking-widest font-headline mt-2">CONFIGURE GAT INFERENCE ENGINE</p>
        </div>
        <button 
           onClick={handleSave}
           className="px-6 py-3 bg-cyan-400/10 border border-cyan-400/50 text-[10px] font-bold tracking-widest font-headline text-cyan-400 hover:bg-cyan-400 hover:text-slate-950 transition-colors"
        >
          SAVE_CHANGES
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <Cpu className="w-5 h-5 text-cyan-400"/>
            <h3 className="font-headline font-bold text-sm tracking-widest uppercase text-on-surface">Compute Target</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex flex-col gap-2">
               <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Device Priority</label>
               <select className="bg-surface-container border border-outline-variant/30 text-cyan-400 text-sm font-headline px-3 py-2 outline-none focus:border-cyan-400/50 transition-colors">
                  <option value="cuda:0">NVIDIA CUDA (GPU_0)</option>
                  <option value="cuda:1">NVIDIA CUDA (GPU_1)</option>
                  <option value="mps">APPLE M-SERIES (MPS)</option>
                  <option value="cpu">SYSTEM CPU (Fallback)</option>
               </select>
            </div>
            
            <div className="flex flex-col gap-2">
               <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">GAT Attention Heads</label>
               <input type="number" defaultValue={8} className="bg-surface-container border border-outline-variant/30 text-cyan-400 text-sm font-headline px-3 py-2 outline-none focus:border-cyan-400/50 transition-colors" />
            </div>
          </div>
        </div>

        <div className="glass-panel p-6">
          <div className="flex items-center gap-3 mb-6">
            <Database className="w-5 h-5 text-cyan-400"/>
            <h3 className="font-headline font-bold text-sm tracking-widest uppercase text-on-surface">Data Ingestion</h3>
          </div>
          
          <div className="space-y-4">
            <div className="flex flex-col gap-2">
               <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Dataset Topology</label>
               <select className="bg-surface-container border border-outline-variant/30 text-cyan-400 text-sm font-headline px-3 py-2 outline-none focus:border-cyan-400/50 transition-colors">
                  <option value="mgtab">MGTAB (Multi-relational)</option>
                  <option value="cresci-15">Cresci-2015</option>
                  <option value="twibot-20">TwiBot-20</option>
               </select>
            </div>
            
            <div className="flex flex-col gap-2">
               <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Edge Dropout Rate</label>
               <input type="number" step="0.1" defaultValue={0.2} className="bg-surface-container border border-outline-variant/30 text-cyan-400 text-sm font-headline px-3 py-2 outline-none focus:border-cyan-400/50 transition-colors" />
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 md:col-span-2 border border-[#FF3E00]/30 shadow-[0_0_15px_rgba(255,62,0,0.05)]">
           <div className="flex items-center gap-3 mb-6">
             <Lock className="w-5 h-5 text-[#FF3E00]"/>
             <h3 className="font-headline font-bold text-sm tracking-widest uppercase text-[#FF3E00]">Security Protocol</h3>
           </div>
           
           <div className="flex items-center justify-between">
              <div>
                 <h4 className="font-headline font-bold text-slate-200">Auto-Flag Threshold</h4>
                 <p className="font-body text-xs text-slate-400 mt-1">Automatically send flag requests to Twitter API if probability exceeds threshold.</p>
              </div>
              <div className="flex items-center gap-4">
                 <span className="text-[#FF3E00] font-headline font-bold">95%</span>
                 <input type="range" min="80" max="99" defaultValue="95" className="w-32 accent-[#FF3E00]" />
              </div>
           </div>
        </div>
      </div>
    </main>
  );
};

export default Settings;
