import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchEvidence } from '../api/client';
import { Send, CheckCircle, RefreshCcw, Download, Eye } from 'lucide-react';
import toast from 'react-hot-toast';

const ForensicDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [evidence, setEvidence] = useState<any[]>([]);

  useEffect(() => {
    fetchEvidence().then(setEvidence).catch(console.error);
  }, []);

  return (
    <main className="p-6 md:p-10 max-w-[1600px] mx-auto min-h-screen">
      {/* Hero Section */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-end mb-12">
        <div className="lg:col-span-8 flex flex-col md:flex-row items-center md:items-end space-y-6 md:space-y-0 md:space-x-8">
          <div className="relative">
            <div className="w-48 h-48 lg:w-56 lg:h-56 bg-surface border-2 border-[#FF3E00] flex justify-center items-center rounded-sm glowing-error vermillion-glow text-[#FF3E00] text-5xl font-black">
              {id?.slice(0, 3)}
            </div>
            <div className="absolute -bottom-4 -right-4 bg-[#FF3E00] text-[#0A0B10] px-4 py-2 font-black font-headline tracking-tighter text-2xl skew-x-[-12deg] shadow-lg">
              VERDICT: BOT
            </div>
          </div>
          
          <div className="text-center md:text-left">
            <div className="flex items-center justify-center md:justify-start space-x-3 mb-2">
              <span className="text-[#FF3E00] font-headline font-bold text-[10px] tracking-[0.3em] uppercase">Status: Flagged Critical</span>
              <div className="h-1 w-12 bg-[#FF3E00]"></div>
            </div>
            <h2 className="text-5xl lg:text-7xl font-black font-headline tracking-tighter mb-2 text-on-background">@{id}_PULSE</h2>
            <p className="font-headline text-slate-400 tracking-widest text-sm mb-6">UID: {id}</p>
            <div className="flex flex-wrap gap-3">
              <span className="px-3 py-1 bg-surface-container-high border border-outline-variant/30 text-[10px] font-bold tracking-widest font-headline text-[#FF3E00] uppercase">High Entropy</span>
              <span className="px-3 py-1 bg-surface-container-high border border-outline-variant/30 text-[10px] font-bold tracking-widest font-headline text-slate-400 uppercase">Automated API</span>
              <span className="px-3 py-1 bg-surface-container-high border border-outline-variant/30 text-[10px] font-bold tracking-widest font-headline text-slate-400 uppercase">Node Cluster 04</span>
            </div>
          </div>
        </div>
        
        <div className="lg:col-span-4 flex flex-col space-y-3">
          <button 
            onClick={() => toast("API Payload dispatched to Twitter Trust & Safety", { style: { background: '#0A0B10', color: '#FF3E00', border: '1px solid #FF3E00' }})}
            className="group flex items-center justify-between px-6 py-4 bg-[#FF3E00] text-[#0A0B10] font-headline font-black tracking-widest hover:brightness-110 transition-all"
          >
            <span>REPORT TO TWITTER</span>
            <Send className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
          <div className="grid grid-cols-2 gap-3">
            <button 
              onClick={() => toast.success("Added to Global Safe List", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }})}
              className="flex justify-center items-center gap-2 px-4 py-3 border border-outline-variant/50 text-slate-300 font-headline font-bold text-[10px] tracking-[0.2em] hover:bg-surface-container-high transition-colors"
            >
               <CheckCircle className="w-4 h-4" /> ADD TO SAFE LIST
            </button>
            <button 
               onClick={() => toast("Recalculating node gradients...", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }})}
               className="flex justify-center items-center gap-2 px-4 py-3 border border-cyan-400/50 text-cyan-400 font-headline font-bold text-[10px] tracking-[0.2em] hover:bg-cyan-400/10 transition-colors"
            >
               <RefreshCcw className="w-4 h-4"/> RECALIBRATE
            </button>
          </div>
        </div>
      </section>

      {/* Radar Chart & Metrics */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        <div className="lg:col-span-2 glass-panel p-8 min-h-[400px] flex flex-col">
          <div className="flex justify-between items-start mb-8">
            <div>
              <h3 className="font-headline font-bold text-xl text-cyan-400 tracking-tight uppercase">Feature Vector Analysis</h3>
              <p className="text-xs text-slate-500 font-headline tracking-widest mt-1">CROSS-REFERENCE: GAT_ENGINE_V3</p>
            </div>
            <div className="text-right">
              <span className="text-4xl font-black font-headline text-[#FF3E00] leading-none">98.4%</span>
              <p className="text-[10px] text-[#FF3E00] font-bold tracking-widest font-headline mt-1">PROBABILITY SCORE</p>
            </div>
          </div>
          
          <div className="flex-1 relative radar-grid flex items-center justify-center border border-outline-variant/10 min-h-[300px]">
            <svg className="w-full max-w-sm h-auto drop-shadow-[0_0_20px_rgba(255,62,0,0.3)]" viewBox="0 0 400 400">
              <circle cx="200" cy="200" r="150" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <circle cx="200" cy="200" r="100" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <circle cx="200" cy="200" r="50" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              
              <line x1="200" y1="50" x2="200" y2="350" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
              <line x1="50" y1="200" x2="350" y2="200" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
              
              <polygon points="200,80 340,200 200,320 60,200" fill="rgba(255, 62, 0, 0.4)" stroke="#FF3E00" strokeWidth="2" />
              <polygon points="200,160 240,200 200,240 160,200" fill="rgba(0, 245, 255, 0.2)" stroke="#00F5FF" strokeWidth="1" strokeDasharray="4" />
              
              <text x="200" y="40" fill="#849495" fontSize="10" fontFamily="Space Grotesk" fontWeight="bold" letterSpacing="1" textAnchor="middle">TWEET VOLUME</text>
              <text x="360" y="205" fill="#849495" fontSize="10" fontFamily="Space Grotesk" fontWeight="bold" letterSpacing="1" textAnchor="start">MENTION FREQ</text>
              <text x="200" y="370" fill="#849495" fontSize="10" fontFamily="Space Grotesk" fontWeight="bold" letterSpacing="1" textAnchor="middle">BEHAVIOR SCORE</text>
              <text x="40" y="205" fill="#849495" fontSize="10" fontFamily="Space Grotesk" fontWeight="bold" letterSpacing="1" textAnchor="end">GROWTH RATE</text>
            </svg>
          </div>
        </div>

        <div className="flex flex-col space-y-6">
          <div className="glass-panel p-6 border-l-4 border-l-[#FF3E00]">
            <p className="text-[10px] font-bold tracking-[0.2em] font-headline text-slate-500 mb-2">NETWORK CENTRALITY</p>
            <div className="flex items-end justify-between">
              <h4 className="text-3xl font-black font-headline text-on-background">0.941</h4>
              <span className="text-[#FF3E00] flex items-center text-xs font-bold font-headline">22%</span>
            </div>
          </div>
          
          <div className="glass-panel p-6 border-l-4 border-l-cyan-400/30">
             <p className="text-[10px] font-bold tracking-[0.2em] font-headline text-slate-500 mb-2">HUMAN PROXIMITY</p>
             <div className="flex items-end justify-between">
                <h4 className="text-3xl font-black font-headline text-on-background">0.122</h4>
                <span className="text-slate-500 flex items-center text-xs font-bold font-headline">84%</span>
             </div>
          </div>
        </div>
      </section>

      {/* GAT Evidence Log */}
      <section className="glass-panel p-8">
        <div className="flex items-center justify-between mb-8">
           <div>
              <h3 className="font-headline font-bold text-xl text-cyan-400 tracking-tight uppercase">GAT Attention Evidence Log</h3>
              <p className="text-xs text-slate-500 font-headline tracking-widest mt-1">CROSS-CONNECTION FEATURE WEIGHTS</p>
           </div>
           <button 
              onClick={() => toast.success("Logs exported to /results_dump.csv", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }})}
              className="flex items-center space-x-2 text-slate-400 hover:text-cyan-400 transition-colors text-[10px] font-bold tracking-[0.2em] font-headline border border-outline-variant/30 px-4 py-2"
           >
              <Download className="w-4 h-4" />
              <span>EXPORT_LOG</span>
           </button>
        </div>
        
        <div className="overflow-x-auto border border-outline-variant/10">
          <table className="w-full text-left font-headline">
             <thead>
                <tr className="bg-surface-container-lowest border-b border-outline-variant/20">
                   <th className="px-6 py-4 text-[10px] font-black tracking-widest text-slate-400 uppercase">Linked Node</th>
                   <th className="px-6 py-4 text-[10px] font-black tracking-widest text-slate-400 uppercase">Attention Weight</th>
                   <th className="px-6 py-4 text-[10px] font-black tracking-widest text-slate-400 uppercase">Inference</th>
                   <th className="px-6 py-4 text-right text-[10px] font-black tracking-widest text-slate-400 uppercase">Action</th>
                </tr>
             </thead>
             <tbody className="divide-y divide-outline-variant/10">
               {evidence.map((ev, idx) => (
                 <tr key={idx} className="hover:bg-cyan-400/5 transition-colors group">
                    <td className="px-6 py-4 font-bold text-sm text-on-background">{ev.node_id}</td>
                    <td className="px-6 py-4 font-mono text-sm text-[#FF3E00] drop-shadow-[0_0_8px_rgba(255,62,0,0.5)]">{ev.attention}</td>
                    <td className="px-6 py-4 text-xs text-slate-400 max-w-sm">{ev.inference}</td>
                    <td className="px-6 py-4 text-right">
                       <Eye 
                          onClick={() => navigate(`/node/${ev.node_id}`)}
                          className="w-5 h-5 inline text-slate-600 hover:text-cyan-400 cursor-pointer transition-colors" 
                       />
                    </td>
                 </tr>
               ))}
             </tbody>
          </table>
        </div>
      </section>
    </main>
  );
};

export default ForensicDetail;
