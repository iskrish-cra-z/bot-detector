import React, { useEffect, useState, useRef, useCallback } from 'react';
import { fetchGraphSample } from '../api/client';
import { Link } from 'react-router-dom';
import ForceGraph2D from 'react-force-graph-2d';
import { Network, Plus, Minus, Focus, X, AlertTriangle } from 'lucide-react';

const NetworkGraph: React.FC = () => {
  const [graphData, setGraphData] = useState<{nodes: any[], links: any[]}>({nodes: [], links: []});
  const [selectedNode, setSelectedNode] = useState<any | null>(null);
  const fgRef = useRef<any>();

  useEffect(() => {
    fetchGraphSample()
      .then(setGraphData)
      .catch(console.error);
  }, []);

  const handleNodeClick = useCallback((node: any) => {
    setSelectedNode(node);
    // Smooth Camera Follow
    fgRef.current?.centerAt(node.x, node.y, 1000);
    fgRef.current?.zoom(6, 1500);
  }, []);

  const closeInspector = () => setSelectedNode(null);

  const resetView = () => {
    fgRef.current?.centerAt(0, 0, 1000);
    fgRef.current?.zoom(1.5, 1000);
  };

  return (
    <main className="w-full h-full relative overflow-hidden bg-background">
      {/* BACKGROUND GRAPH CANVAS & HEATMAP */}
      <div className="absolute inset-0 z-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-cyan-900/10 via-background to-background animate-pulse duration-[10000ms]">
         <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel={(node: any) => {
              const baseConf = Math.round((node.val ? Math.min(node.val * 10, 100) : 85));
              if (node.node_type === 'tweet') return `TWEET_STR_PAYLOAD [Hash_ID: ${node.id}]`;
              if (node.node_type === 'hashtag') return `#TOPIC_CLUSTER [Hash_ID: ${node.id}]`;
              return `${node.label} (Threat Conf: ${baseConf}%)`;
            }}
            nodeColor={(node: any) => (node.is_bot ? '#FF3E00' : '#00F5FF')}
            nodeRelSize={4}
            nodeVal={(node) => node.val}
            linkColor={() => 'rgba(0, 245, 255, 0.15)'}
            linkWidth={1}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={1.5}
            linkDirectionalParticleSpeed={0.005}
            linkDirectionalParticleColor={(link: any) => 'rgba(0, 245, 255, 0.8)'}
            onNodeClick={handleNodeClick}
            backgroundColor="transparent"
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              const r = Math.sqrt(Math.max(0, node.val || 1)) + 2;
              
              // Glow Intensity = simulated confidence based on degree/value
              const confidence = node.val ? Math.min(node.val * 10, 100) / 100 : 0.85;
              const glowRadius = confidence * 25 + 5;
              
              ctx.beginPath();
              
              // Determine Color
              let nodeColor = '#0A0B10';
              if (node.node_type === 'user') {
                 nodeColor = node.is_bot ? '#FF3E00' : '#00F5FF';
              } else if (node.node_type === 'tweet') {
                 nodeColor = '#A855F7'; // Neon Purple for semantic data
              } else if (node.node_type === 'hashtag') {
                 nodeColor = '#10B981'; // Neon Emerald for topics
              }

              // Heterogeneous Geometries
              if (node.node_type === 'user') {
                 // Circle for User
                 ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
                 ctx.fillStyle = nodeColor;
              } else if (node.node_type === 'tweet') {
                 // Square for Tweet
                 ctx.rect(node.x - r, node.y - r, r*2, r*2);
                 ctx.fillStyle = nodeColor;
              } else if (node.node_type === 'hashtag') {
                 // Diamond for Hashtag
                 ctx.moveTo(node.x, node.y - r*1.2);
                 ctx.lineTo(node.x + r*1.2, node.y);
                 ctx.lineTo(node.x, node.y + r*1.2);
                 ctx.lineTo(node.x - r*1.2, node.y);
                 ctx.fillStyle = nodeColor;
              } else {
                 ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
                 ctx.fillStyle = '#00F5FF';
              }
              
              // Neon Glow
              ctx.shadowBlur = glowRadius;
              ctx.shadowColor = nodeColor;
              ctx.fill();
              
              ctx.shadowBlur = 0; // reset
            }}
          />
      </div>

      {/* OVERLAY CONTROLS */}
      <div className="absolute left-8 top-1/2 -translate-y-1/2 z-10 flex flex-col gap-4 pointer-events-auto">
        <div className="bg-surface-container-low/60 backdrop-blur-xl border border-outline-variant/30 p-2 flex flex-col gap-3 rounded-lg shadow-2xl">
          <button className="w-10 h-10 flex items-center justify-center text-cyan-400 hover:bg-cyan-400/10 transition-colors rounded" onClick={() => fgRef.current?.zoom(fgRef.current.zoom() * 1.5, 400)}>
            <Plus className="w-5 h-5"/>
          </button>
          <button className="w-10 h-10 flex items-center justify-center text-cyan-400 hover:bg-cyan-400/10 transition-colors rounded" onClick={() => fgRef.current?.zoom(fgRef.current.zoom() / 1.5, 400)}>
            <Minus className="w-5 h-5"/>
          </button>
          <div className="h-px bg-outline-variant/30 mx-2"></div>
          <button className="w-10 h-10 flex items-center justify-center text-cyan-400 hover:bg-cyan-400/10 transition-colors rounded" onClick={resetView}>
            <Focus className="w-4 h-4"/>
          </button>
        </div>
        
        <div className="bg-surface-container-low/60 backdrop-blur-xl border border-outline-variant/30 p-3 rounded-lg flex flex-col gap-4 shadow-2xl">
          <p className="text-[9px] font-headline font-bold text-slate-500 uppercase tracking-widest mb-1">Filter Scale</p>
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between text-[10px] font-headline text-cyan-400">
              <span>HUMAN</span>
              <span className="text-[#FF3E00]">BOT</span>
            </div>
            <div className="w-32 h-1 bg-surface-container-highest relative rounded-full">
              <div className="absolute inset-y-0 left-0 w-full bg-gradient-to-r from-cyan-400 to-[#FF3E00] rounded-full"></div>
            </div>
          </div>
        </div>
      </div>

      {/* NODE INSPECTOR SLIDEOUT */}
      <aside className={`absolute right-0 top-0 h-full w-80 bg-slate-950/80 backdrop-blur-2xl border-l border-slate-800/50 z-20 transition-transform duration-500 flex flex-col ${selectedNode ? 'translate-x-0' : 'translate-x-full'}`}>
        <div className="p-6 h-full flex flex-col overflow-y-auto">
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full animate-pulse ${selectedNode?.is_bot ? 'bg-[#FF3E00]' : 'bg-cyan-400'}`}></div>
              <h2 className="font-headline font-bold text-lg tracking-tight uppercase text-on-surface">Node Details</h2>
            </div>
            <X className="text-slate-500 cursor-pointer hover:text-white" onClick={closeInspector} />
          </div>

          {selectedNode && (
            <>
              <div className="flex flex-col items-center mb-8">
                <div className="relative mb-4">
                  <div className={`absolute -inset-2 rounded-full opacity-30 blur-md ${selectedNode.is_bot ? 'bg-gradient-to-tr from-[#FF3E00] to-transparent' : 'bg-gradient-to-tr from-cyan-400 to-transparent'}`}></div>
                  <div className={`w-24 h-24 rounded-full border-2 relative z-10 flex justify-center items-center font-black text-2xl bg-surface-container ${selectedNode.is_bot ? 'border-[#FF3E00] text-[#FF3E00]' : 'border-cyan-400 text-cyan-400'}`}>
                    {selectedNode.label.slice(0,2)}
                  </div>
                </div>
                <h3 className="font-headline font-black text-xl text-on-background tracking-tighter">{selectedNode.label}</h3>
                <p className={`text-[10px] font-headline font-bold tracking-[0.2em] mt-1 uppercase ${selectedNode.is_bot ? 'text-[#FF3E00]' : 'text-cyan-400'}`}>
                  {selectedNode.is_bot ? 'BOT PROBABILITY: HIGH' : 'HUMAN PROBABILITY: HIGH'}
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2 mb-8">
                <div className="bg-surface-container-low p-3 rounded-lg border border-outline-variant/10">
                  <p className="text-[9px] font-headline text-slate-500 uppercase tracking-widest mb-1">Degree</p>
                  <p className="font-headline font-bold text-cyan-400">{selectedNode.val}</p>
                </div>
                <div className="bg-surface-container-low p-3 rounded-lg border border-outline-variant/10">
                  <p className="text-[9px] font-headline text-slate-500 uppercase tracking-widest mb-1">UID</p>
                  <p className="font-headline font-bold text-cyan-400">{selectedNode.id}</p>
                </div>
              </div>

              <div className="flex-1 space-y-4">
                <h4 className="text-[10px] font-headline text-slate-500 uppercase tracking-widest">Topology Flag</h4>
                <div className="flex items-start gap-3 bg-surface-container-low p-3 rounded border border-outline-variant/10">
                  <AlertTriangle className={`w-4 h-4 mt-0.5 ${selectedNode.is_bot ? 'text-[#FF3E00]' : 'text-cyan-400'}`} />
                  <div>
                    <p className="text-[11px] font-headline font-bold text-white uppercase">{selectedNode.is_bot ? 'ECHO_CHAMBER_NODE' : selectedNode.node_type === 'tweet' ? 'SEMANTIC_PAYLOAD' : selectedNode.node_type === 'hashtag' ? 'TOPIC_CLUSTER_ROOT' : 'Standard Traffic'}</p>
                    <p className="text-[10px] text-slate-400 font-body leading-tight mt-1">Calculated via Heterogeneous Graph Transformer (HGT).</p>
                  </div>
                </div>
              </div>

              <Link to={`/node/${selectedNode.id}`} className="w-full mt-4 bg-transparent border border-cyan-400 text-cyan-400 py-3 rounded-DEFAULT font-headline font-bold text-xs tracking-widest uppercase hover:bg-cyan-400/10 transition-all text-center block">
                FORENSIC_DEEP_DIVE
              </Link>
            </>
          )}
        </div>
      </aside>

      {/* SCAN STATUS OVERLAY */}
      <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10 pointer-events-none">
        <div className="bg-slate-950/80 backdrop-blur-xl border border-cyan-400/20 px-6 py-3 rounded-full flex items-center gap-6 shadow-[0_0_30px_rgba(0,245,255,0.15)]">
          <div className="flex flex-col">
            <span className="text-[8px] font-headline text-slate-500 uppercase tracking-widest">Rendered Nodes</span>
            <span className="text-sm font-headline font-black text-cyan-400">{graphData.nodes.length} / 10K</span>
          </div>
          <div className="h-8 w-px bg-slate-800"></div>
          <div className="flex items-center gap-2">
             <div className="w-3 h-3 bg-cyan-400 rounded-sm"></div>
             <span className="text-[10px] font-headline font-bold text-white uppercase tracking-tighter">SIGMA_PHYSICS_ENABLED</span>
          </div>
        </div>
      </div>
    </main>
  );
};

export default NetworkGraph;
