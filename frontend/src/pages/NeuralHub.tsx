import React, { useEffect, useState, useRef } from 'react';
import { fetchStats, fetchDetections } from '../api/client';
import { Link } from 'react-router-dom';
import { useScan } from '../context/ScanContext';

const NeuralHub: React.FC = () => {
  const [stats, setStats] = useState<any>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const { logs, isScanning } = useScan();

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchStats().then(setStats).catch(console.error);
    fetchDetections().then(setDetections).catch(console.error);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <main className="p-6 md:p-10 neural-grid h-full relative">
      <div className="max-w-7xl mx-auto space-y-6 relative z-10 flex flex-col h-full pb-20">
        
        {/* Key Stats Bar */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 shrink-0">
          <div className="glass-panel p-5 cyan-glow">
            <p className="text-[10px] font-bold text-slate-500 tracking-widest uppercase font-headline">Total Accounts Scanned</p>
            <div className="flex items-baseline gap-2 mt-2">
              <h2 className="text-4xl font-black text-cyan-400 font-headline tabular-nums">
                {stats ? stats.total_scanned.toLocaleString() : '---'}
              </h2>
            </div>
            <div className="h-1 bg-slate-800 mt-4 overflow-hidden">
              <div className="h-full bg-cyan-400 w-full shadow-[0_0_10px_#00F5FF]"></div>
            </div>
          </div>
          
          <div className="glass-panel p-5 vermillion-glow">
            <p className="text-[10px] font-bold text-slate-500 tracking-widest uppercase font-headline">Bot Probability</p>
            <div className="flex items-baseline gap-2 mt-2">
              <h2 className="text-4xl font-black text-[#FF3E00] font-headline tabular-nums">
                {stats ? `${(stats.bot_probability_avg * 100).toFixed(1)}%` : '---'}
              </h2>
              <span className="text-xs text-[#FF3E00]/60 font-headline">CRITICAL</span>
            </div>
            <div className="flex gap-1 mt-4">
              <div className="h-1 flex-1 bg-[#FF3E00]"></div>
              <div className="h-1 flex-1 bg-[#FF3E00]"></div>
              <div className="h-1 flex-1 bg-[#FF3E00]"></div>
              <div className="h-1 flex-1 bg-slate-800"></div>
              <div className="h-1 flex-1 bg-slate-800"></div>
            </div>
          </div>
          
          <div className="glass-panel p-5">
            <p className="text-[10px] font-bold text-slate-500 tracking-widest uppercase font-headline">GAT Model Accuracy</p>
            <div className="flex items-baseline gap-2 mt-2">
              <h2 className="text-4xl font-black text-on-background font-headline tabular-nums">
                {stats ? `${(stats.accuracy * 100).toFixed(1)}%` : '---'}
              </h2>
              <span className="text-xs text-slate-500 font-headline">V3_ENGINE</span>
            </div>
            <div className="flex items-center gap-2 mt-4 text-[10px] text-cyan-400/80 font-headline">
              <span className="material-symbols-outlined text-xs">verified</span>
              STABLE_DIFFUSION_ANALYSIS_ACTIVE
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
          
          {/* Recent Detections Table */}
          <div className="lg:col-span-8 glass-panel overflow-hidden flex flex-col">
            <div className="p-4 border-b border-slate-800/50 flex justify-between items-center bg-slate-900/40">
              <h3 className="text-xs font-bold tracking-widest uppercase font-headline">Recent Detections</h3>
              <div className="flex gap-2">
                <span className="px-2 py-0.5 bg-slate-800 text-[8px] font-bold font-headline rounded text-cyan-400">
                  {isScanning ? 'SCANNING...' : 'AUTO_REFRESH: 5s'}
                </span>
              </div>
            </div>
            <div className="overflow-x-auto flex-1 custom-scrollbar">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="text-[9px] font-bold text-slate-500 uppercase tracking-widest bg-slate-900/50">
                    <th className="px-6 py-4">Account_Node</th>
                    <th className="px-6 py-4">Bot Score</th>
                    <th className="px-6 py-4">GAT Reason</th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="text-xs font-medium">
                  {detections.map((det) => (
                    <tr key={det.id} className="hover:bg-slate-800/30 transition-colors border-b border-slate-800/20">
                      <td className="px-6 py-4 flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-sm ${det.bot_score > 50 ? 'bg-[#FF3E00]/20 border border-[#FF3E00]/50' : 'bg-cyan-400/20 border border-cyan-400/50'}`}></div>
                        <div>
                          <p className="text-on-background">{det.username}</p>
                          <p className="text-[9px] text-slate-500">{det.id}</p>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`font-headline font-bold ${det.bot_score > 50 ? 'text-[#FF3E00]' : 'text-cyan-400'}`}>
                          {det.bot_score}/100
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className="text-slate-400 font-headline text-[10px] uppercase">{det.reason}</span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex justify-end gap-2">
                          <Link to={`/node/${det.id}`} className="px-3 py-1 bg-slate-800 text-cyan-400 text-[9px] font-bold uppercase hover:bg-slate-700">INSPECT</Link>
                          {det.bot_score > 50 ? (
                            <button className="px-3 py-1 bg-[#FF3E00]/10 border border-[#FF3E00]/30 text-[#FF3E00] text-[9px] font-bold uppercase hover:bg-[#FF3E00]/20">FLAG</button>
                          ) : (
                            <button className="px-3 py-1 bg-cyan-400/10 border border-cyan-400/30 text-cyan-400 text-[9px] font-bold uppercase hover:bg-cyan-400/20">VOUCH</button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Threat Map Preview & Logs */}
          <div className="lg:col-span-4 space-y-6 flex flex-col min-h-0">
            <div className="glass-panel p-4 h-64 relative overflow-hidden group border border-outline-variant/20 flex flex-col items-center justify-center shrink-0">
               <h3 className="absolute top-4 left-4 text-xs font-bold tracking-widest uppercase font-headline text-slate-400 z-20">GAT Architecture Map</h3>
               <div className="mt-8 relative w-48 h-48 flex items-center justify-center">
                  <div className={`absolute inset-0 border border-cyan-400/10 rounded-full ${isScanning ? 'animate-ping duration-1000' : 'animate-pulse'}`}></div>
                  <div className="absolute top-1/2 left-0 w-full h-px bg-cyan-400/10"></div>
                  <div className="absolute left-1/2 top-0 w-px h-full bg-cyan-400/10"></div>
                  <div className="absolute top-10 right-10 w-3 h-3 bg-[#FF3E00] rounded-full shadow-[0_0_15px_#FF3E00]"></div>
                  <div className="absolute bottom-20 left-12 w-2 h-2 bg-[#FF3E00] rounded-full shadow-[0_0_10px_#FF3E00]"></div>
                  <div className="absolute top-1/3 left-1/4 w-4 h-4 bg-cyan-400 rounded-full shadow-[0_0_20px_#00F5FF]"></div>
               </div>
            </div>

            <div className="glass-panel p-4 flex flex-col border border-outline-variant/20 flex-1 min-h-0">
              <div className="flex items-center gap-4 mb-4 border-b border-slate-800 pb-2 shrink-0">
                <div className={`w-2 h-2 ${isScanning ? 'bg-[#FF3E00]' : 'bg-cyan-400'} rounded-full animate-pulse`}></div>
                <h3 className={`text-xs font-bold tracking-widest uppercase font-headline ${isScanning ? 'text-[#FF3E00]' : 'text-cyan-400'}`}>
                   {isScanning ? 'SYSTEM_LOGS_STREAM (SCANNING)' : 'SYSTEM_LOGS_STREAM'}
                </h3>
              </div>
              <div ref={scrollRef} className="font-headline text-[10px] space-y-2 text-slate-500 uppercase tracking-tight overflow-y-auto custom-scrollbar flex-1 relative">
                {logs.map((log, idx) => (
                   <p key={idx} className={`${log.includes('SUCCESS') || log.includes('INFO') ? 'text-cyan-400' : ''} ${log.includes('ALERT') ? 'text-[#FF3E00]' : ''}`}>
                      <span className="text-cyan-400/40 opacity-50 mr-2">[HUD]</span>
                      {log}
                   </p>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Background Decorators */}
      <div className="absolute top-[10%] right-[10%] w-[30%] h-[30%] bg-cyan-400/5 blur-[120px] rounded-full pointer-events-none z-0"></div>
      <div className="absolute bottom-[20%] left-[10%] w-[25%] h-[25%] bg-[#FF3E00]/5 blur-[100px] rounded-full pointer-events-none z-0"></div>
    </main>
  );
};

export default NeuralHub;
