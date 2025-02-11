import { useState } from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

export default function FixedEmbeddingButton({ selectedModel }) {
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [metrics, setMetrics] = useState(null);
    const [modelPath, setModelPath] = useState(null);
    const [params, setParams] = useState({
        ep_num: 3,
        train_rate: 0.8,
        lr: 0.0001,
    });
    const [progress, setProgress] = useState({
        currentEpoch: 0,
        totalEpochs: 0,
        currentBatch: 0,
        totalBatches: 0
    });

    const handleParamChange = (e) => {
        const { name, value } = e.target;
        let parsedValue = value;
        
        // Handle numeric inputs
        if (name === 'ep_num') {
            parsedValue = Math.min(Math.max(parseInt(value) || 0, 0), 10); // Range 0-10
        } else if (name === 'train_rate') {
            parsedValue = Math.min(Math.max(parseFloat(value) || 0, 0), 1); // Range 0-1
        } else if (name === 'lr') {
            parsedValue = Math.max(parseFloat(value) || 0, 0); // Must be positive
        }

        setParams(prev => ({
            ...prev,
            [name]: parsedValue
        }));
    };

    const handleTraining = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:8000/api/train-fixed-embeddings/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: selectedModel,
                    ep_num: params.ep_num,
                    train_rate: params.train_rate,
                    lr: params.lr
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const data = JSON.parse(line);
                        console.log('Received update:', data);
                        
                        if (data.progress) {
                            setProgress(prev => ({
                                ...prev,
                                ...data.progress
                            }));
                        } else if (data.metrics) {
                            setMetrics(data.metrics);
                            setModelPath(data.model_path);
                        } else if (data.error) {
                            throw new Error(data.error);
                        }
                    } catch (parseError) {
                        console.error('Error parsing update:', parseError);
                        console.log('Raw line:', line);
                        continue;
                    }
                }
            }

            setMessage('Training completed successfully!');
        } catch (error) {
            console.error('Training error:', error);
            setMessage('Training failed: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const handleDownload = async (path, filename) => {
        try {
            const response = await fetch(`http://localhost:8000/api/download-file/?path=${encodeURIComponent(path)}`);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Download error:', error);
            setMessage('Failed to download file');
        }
    };

    return (
        <>
            {/* Parameters Form */}
            <form onSubmit={handleTraining} className="mb-6 space-y-4">
                <div className="grid grid-cols-3 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Number of Epochs
                        </label>
                        <input
                            type="number"
                            name="ep_num"
                            value={params.ep_num}
                            onChange={handleParamChange}
                            min="1"
                            max="10"
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Training Rate
                        </label>
                        <input
                            type="number"
                            name="train_rate"
                            value={params.train_rate}
                            onChange={handleParamChange}
                            step="0.1"
                            min="0"
                            max="1"
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Learning Rate
                        </label>
                        <input
                            type="number"
                            name="lr"
                            value={params.lr}
                            onChange={handleParamChange}
                            step="0.0001"
                            min="0"
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                        />
                    </div>
                </div>

                <button
                    type="submit"
                    disabled={isLoading}
                    className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors
                        ${isLoading 
                            ? 'bg-gray-400 cursor-not-allowed' 
                            : 'bg-blue-500 hover:bg-blue-600'
                        }`}
                >
                    {isLoading ? 'Training in Progress...' : 'Start Training'}
                </button>
            </form>

            {/* Progress Bars */}
            {isLoading && (
                <div className="mb-6 space-y-4">
                    {/* Epoch Progress */}
                    <div>
                        <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>Epoch Progress</span>
                            <span>{progress.currentEpoch} / {progress.totalEpochs}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${(progress.currentEpoch / progress.totalEpochs) * 100}%` }}
                            ></div>
                        </div>
                    </div>

                    {/* Batch Progress */}
                    <div>
                        <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>Batch Progress</span>
                            <span>{progress.currentBatch} / {progress.totalBatches}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                                className="bg-green-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${(progress.currentBatch / progress.totalBatches) * 100}%` }}
                            ></div>
                        </div>
                    </div>
                </div>
            )}

            {/* Results Section */}
            {metrics && (
                <div className="space-y-6">
                    {/* Training Loss Chart */}
                    <div className="h-64 bg-white p-4 rounded-lg shadow">
                        <Line
                            data={{
                                labels: metrics.epochs,
                                datasets: [
                                    {
                                        label: 'Training Loss',
                                        data: metrics.train_loss,
                                        borderColor: 'rgb(59, 130, 246)', // blue-500
                                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                                        tension: 0.1
                                    },
                                    {
                                        label: 'Testing Loss',
                                        data: metrics.test_loss,
                                        borderColor: 'rgb(34, 197, 94)', // green-500
                                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                                        tension: 0.1
                                    }
                                ]
                            }}
                            options={{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {
                                    legend: {
                                        position: 'top',
                                    },
                                    title: {
                                        display: true,
                                        text: 'Training Progress'
                                    }
                                },
                                scales: {
                                    y: {
                                        beginAtZero: true,
                                        grid: {
                                            color: 'rgba(0, 0, 0, 0.1)',
                                        }
                                    },
                                    x: {
                                        grid: {
                                            color: 'rgba(0, 0, 0, 0.1)',
                                        }
                                    }
                                }
                            }}
                        />
                    </div>

                    {/* Metrics Tables */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-white rounded-lg shadow">
                            <h4 className="font-semibold mb-3 text-gray-900">Training Metrics</h4>
                            <div className="space-y-2">
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Accuracy:</span>
                                    <span className="font-medium">{metrics.final_train.accuracy.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Precision:</span>
                                    <span className="font-medium">{metrics.final_train.precision.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Recall:</span>
                                    <span className="font-medium">{metrics.final_train.recall.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">F1 Score:</span>
                                    <span className="font-medium">{metrics.final_train.f1.toFixed(4)}</span>
                                </p>
                            </div>
                        </div>
                        <div className="p-4 bg-white rounded-lg shadow">
                            <h4 className="font-semibold mb-3 text-gray-900">Testing Metrics</h4>
                            <div className="space-y-2">
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Accuracy:</span>
                                    <span className="font-medium">{metrics.final_test.accuracy.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Precision:</span>
                                    <span className="font-medium">{metrics.final_test.precision.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">Recall:</span>
                                    <span className="font-medium">{metrics.final_test.recall.toFixed(4)}</span>
                                </p>
                                <p className="flex justify-between">
                                    <span className="text-gray-600">F1 Score:</span>
                                    <span className="font-medium">{metrics.final_test.f1.toFixed(4)}</span>
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Download Button */}
                    {modelPath && (
                        <button
                            onClick={() => handleDownload(modelPath, 'fixed_embeddings_model.pth')}
                            className="w-full py-3 px-4 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors"
                        >
                            Download Model
                        </button>
                    )}
                </div>
            )}

            {message && (
                <div className={`mt-4 p-4 rounded-lg ${
                    message.includes('success') 
                        ? 'bg-green-50 text-green-800 border border-green-200' 
                        : 'bg-red-50 text-red-800 border border-red-200'
                }`}>
                    {message}
                </div>
            )}
        </>
    );
} 