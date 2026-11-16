import React, { createContext, useContext, useEffect, useState, useRef } from 'react';
import toast from 'react-hot-toast';

interface ScanContextType {
  logs: string[];
  isScanning: boolean;
  triggerScan: () => void;
}

const ScanContext = createContext<ScanContextType>({
  logs: [],
  isScanning: false,
  triggerScan: () => {}
});

export const useScan = () => useContext(ScanContext);

export const ScanProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [logs, setLogs] = useState<string[]>([
    "[SYSTEM] Neural_Hub initialized... awaiting trigger",
  ]);
  const [isScanning, setIsScanning] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let reconnectTimer: any;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`;

    const connect = () => {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'log' || data.type === 'status') {
            setLogs(prev => [...prev, data.message]);
            
            if (data.message.includes('[SUCCESS]')) {
               setIsScanning(false);
               toast.success("GAT Scan completed successfully!", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }});
            }
          }
        } catch (e) {
          setLogs(prev => [...prev, event.data]);
        }
      };

      ws.onclose = () => {
         // Auto-reconnect after 3 seconds if disconnected
         reconnectTimer = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const triggerScan = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      setIsScanning(true);
      setLogs([]); // Clear previous logs
      wsRef.current.send("START_SCAN");
      toast("Initiating Neural Scan...", { style: { background: '#0A0B10', color: '#00F5FF', border: '1px solid #00F5FF' }});
    } else {
      toast.error("Scanner offline. Reconnecting...", { style: { background: '#0A0B10', color: '#FF3E00', border: '1px solid #FF3E00' }});
    }
  };

  return (
    <ScanContext.Provider value={{ logs, isScanning, triggerScan }}>
      {children}
    </ScanContext.Provider>
  );
};
