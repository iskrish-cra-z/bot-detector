import React from 'react';
import { History, Download, Filter } from 'lucide-react';

const Archive: React.FC = () => {
  // Mock historical data
  const historicalScans = [
    { id: 'SCAN-8429', date: '2026-03-30', nodes: 10199, flagged: 213, status: 'COMPLETED' },
    { id: 'SCAN-8428', date: '2026-03-29', nodes: 10199, flagged: 189, status: 'COMPLETED' },
    { id: 'SCAN-8427', date: '2026-03-28', nodes: 8432, flagged: 102, status: 'ABORTED' },
  ];

  return (
    <main className="p-6 md:p-10 max-w-7xl mx-auto min-h-screen">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-3xl font-black font-headline text-cyan-400 tracking-tighter uppercase flex items-center gap-3">
            <History className="w-8 h-8"/>
            Historical Archive
          </h2>
          <p className="text-slate-500 text-xs font-bold tracking-widest font-headline mt-2">PREVIOUS GAT PIPELINE EXECUTIONS</p>
        </div>
        <div className="flex gap-4">
          <button className="flex items-center gap-2 px-4 py-2 border border-outline-variant/30 text-[10px] font-bold tracking-widest font-headline text-slate-400 hover:text-cyan-400 hover:border-cyan-400/50 transition-colors">
            <Filter className="w-4 h-4"/>
            FILTER_LOGS
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-cyan-400/10 border border-cyan-400/50 text-[10px] font-bold tracking-widest font-headline text-cyan-400 hover:bg-cyan-400 hover:text-slate-950 transition-colors">
            <Download className="w-4 h-4"/>
            EXPORT_CSV
          </button>
        </div>
      </div>

      <div className="glass-panel overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="text-[9px] font-bold text-slate-500 uppercase tracking-widest bg-slate-900/50 border-b border-slate-800">
              <th className="px-6 py-4">Job_ID</th>
              <th className="px-6 py-4">Execution_Date</th>
              <th className="px-6 py-4">Nodes_Processed</th>
              <th className="px-6 py-4">Threats_Flagged</th>
              <th className="px-6 py-4 font-right">Status</th>
            </tr>
          </thead>
          <tbody className="text-xs font-medium">
            {historicalScans.map((scan) => (
              <tr key={scan.id} className="hover:bg-slate-800/30 transition-colors border-b border-slate-800/20">
                <td className="px-6 py-4 font-headline text-cyan-400 font-bold tracking-widest">{scan.id}</td>
                <td className="px-6 py-4 text-slate-300 font-headline">{scan.date}</td>
                <td className="px-6 py-4 text-slate-400 font-mono tabular-nums">{scan.nodes.toLocaleString()}</td>
                <td className="px-6 py-4 text-[#FF3E00] font-headline font-bold">{scan.flagged}</td>
                <td className="px-6 py-4">
                  <span className={`px-2 py-1 text-[9px] font-bold tracking-widest font-headline ${scan.status === 'COMPLETED' ? 'text-cyan-400 bg-cyan-400/10 border border-cyan-400/30' : 'text-[#FF3E00] bg-[#FF3E00]/10 border border-[#FF3E00]/30'}`}>
                    {scan.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
};

export default Archive;
