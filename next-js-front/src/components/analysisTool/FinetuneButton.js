'use client';
import { useState } from 'react';
import { API_BASE_URL } from '../../config/urls';

export default function FinetuneButton({ selectedModel, currentDirs }) {
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [testOutput, setTestOutput] = useState([]);

    const handleTest = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setTestOutput([]);

        try {
            const response = await fetch(`${API_BASE_URL}/backend/api/test-model/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: selectedModel,
                    input_directory: currentDirs.input_directory
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
                    setTestOutput(prev => [...prev, line]);
                }
            }

            setMessage('Testing completed successfully!');
        } catch (error) {
            console.error('Testing error:', error);
            setMessage('Testing failed: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="mt-6 space-y-4 w-full">
            <button
                onClick={handleTest}
                disabled={isLoading}
                className={`w-full py-3 px-4 rounded-md text-white font-medium transition-colors
                    ${isLoading 
                        ? 'bg-gray-400 cursor-not-allowed' 
                        : 'bg-blue-500 hover:bg-blue-600'
                    }`}
            >
                {isLoading ? `Testing ${selectedModel}...` : `Test ${selectedModel} Model`}
            </button>

            {/* Testing Output Display */}
            {testOutput.length > 0 && (
                <div className="mt-4 p-4 bg-black rounded-lg max-w-3xl overflow-x-hidden">
                    <div className="overflow-hidden">
                        <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap break-words overflow-x-auto max-h-[400px] overflow-y-auto">
                            {testOutput.join('\n')}
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