'use client';
import { useState } from 'react';
import { API_BASE_URL } from '../../config/urls';

export default function PredictButton({ selectedModel, currentDirs }) {
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState('');
    const [predictOutput, setPredictOutput] = useState([]);
    const [file, setFile] = useState(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile && selectedFile.name.endsWith('.txt')) {
            setFile(selectedFile);
            setMessage('');
        } else {
            setMessage('Please select a .txt file');
        }
    };

    const handlePredict = async (e) => {
        e.preventDefault();
        if (!file) {
            setMessage('Please select a file first');
            return;
        }

        setIsLoading(true);
        setPredictOutput([]);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', selectedModel);
        formData.append('input_directory', currentDirs.input_directory);

        try {
            const response = await fetch(`${API_BASE_URL}/backend/api/predict-model/`, {
                method: 'POST',
                body: formData,
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
                    setPredictOutput(prev => [...prev, line]);
                }
            }

            setMessage('Prediction completed successfully!');
            setFile(null);
            e.target.reset();
        } catch (error) {
            console.error('Prediction error:', error);
            setMessage('Prediction failed: ' + error.message);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="mt-6 space-y-4 w-full">
            <form onSubmit={handlePredict} className="space-y-4">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                    <div className="text-center">
                        <input
                            type="file"
                            accept=".txt"
                            onChange={handleFileChange}
                            className="hidden"
                            id="txt-upload"
                            disabled={isLoading}
                        />
                        <label
                            htmlFor="txt-upload"
                            className="cursor-pointer flex flex-col items-center space-y-2"
                        >
                            <span className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                                Choose TXT File
                            </span>
                            <span className="text-sm text-gray-500">
                                {file ? file.name : 'No file chosen'}
                            </span>
                        </label>
                    </div>
                </div>

                {file && (
                    <div className="text-sm text-gray-600">
                        <div className="font-medium mb-1">Selected file:</div>
                        <div className="pl-2">
                            {file.name}
                        </div>
                    </div>
                )}

                <button
                    type="submit"
                    disabled={isLoading || !file}
                    className={`w-full py-3 px-4 rounded-lg text-white font-medium transition-colors
                        ${(isLoading || !file)
                            ? 'bg-gray-400 cursor-not-allowed' 
                            : 'bg-blue-600 hover:bg-blue-700'
                        }`}
                >
                    {isLoading ? `Running Prediction...` : `Run Prediction`}
                </button>
            </form>

            {/* Prediction Output Display */}
            {predictOutput.length > 0 && (
                <div className="mt-4 p-4 bg-black rounded-lg max-w-3xl overflow-x-hidden">
                    <div className="overflow-hidden">
                        <pre className="text-green-400 font-mono text-sm whitespace-pre-wrap break-words overflow-x-auto max-h-[400px] overflow-y-auto">
                            {predictOutput.join('\n')}
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