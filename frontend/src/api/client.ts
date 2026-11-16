import axios from 'axios';

// Vite proxies /api to the backend
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const fetchStats = async () => {
  const { data } = await apiClient.get('/stats');
  return data;
};

export const fetchDetections = async () => {
  const { data } = await apiClient.get('/detections');
  return data;
};

export const fetchGraphSample = async () => {
  const { data } = await apiClient.get('/graph');
  return data;
};

export const fetchEvidence = async () => {
  const { data } = await apiClient.get('/evidence');
  return data;
};

export default apiClient;
