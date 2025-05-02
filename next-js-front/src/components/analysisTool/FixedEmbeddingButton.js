'use client';
import { useState } from 'react';
import { API_BASE_URL } from '../../config/urls';

export default function FixedEmbeddingButton({ selectedModel, currentDirs }) {
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [trainingOutput, setTrainingOutput] = useState([]);

    const handleTraining = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setTrainingOutput([]);

        try {
            console.log("aaaaa", currentDirs.input_directory, selectedModel);
            const response = await fetch(`${API_BASE_URL}/backend/api/train-model/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    input_directory: currentDirs.input_directory,
                    model: selectedModel
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
                    setTrainingOutput(prev => [...prev, line]);
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

    return (
        <div className="mt-6 space-y-4 w-full">
            <button
                onClick={handleTraining}
                disabled={isLoading}
                className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors
                    ${isLoading 
                        ? 'bg-gray-400 cursor-not-allowed' 
                        : 'bg-blue-500 hover:bg-blue-600'
                    }`}
            >
                {isLoading ? `Training ${selectedModel}...` : `Train ${selectedModel} Model`}
            </button>

            {/* Training Output Display */}
            {trainingOutput.length > 0 && (
                <div className="mt-4 p-4 bg-black rounded-lg max-w-3xl overflow-x-hidden">
                    <div className="overflow-hidden">
                        <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap break-words overflow-x-auto max-h-[400px] overflow-y-auto">
                            {trainingOutput.join('\n')}
                        </pre>
                    </div>
                </div>
            )}

            {message && (
                <div className={`mt-4 p-4 rounded-lg w-full overflow-hidden ${
                    message.includes('success') 
                        ? 'bg-green-50 text-green-800 border border-green-200' 
                        : 'bg-red-50 text-red-800 border border-red-200'
                }`}>
                    <div className="whitespace-pre-wrap break-words">
                        {message}
                    </div>
                </div>
            )}
        </div>
    );
} 