'use client';
import React from 'react';
import { useState } from 'react';
import ModelTrainingCard from './ModelTrainingCard';

export default function CSVUploadCard() {
    const [files, setFiles] = useState([]);
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentDirs, setCurrentDirs] = useState(null);
    const [selectedModel, setSelectedModel] = useState('');

    const modelOptions = [
        'UCE',
        'tGPT',
        'scGPT',
        'scFoundation',
        'scBERT',
        'Openbiomed(cellLM)',
        'CellPLM',
        'GeneFormer'
    ];

    const handleFileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter(file => file.type === 'text/csv');
        
        if (validFiles.length !== selectedFiles.length) {
            setMessage('Please select only CSV files');
        } else if (validFiles.length > 0) {
            setFiles(validFiles);
            setMessage('');
        }
    };

    const clearCache = async () => {
        if (!currentDirs) {
            setMessage('No cache to clear');
            return;
        }

        try {
            const response = await fetch('http://localhost:8000/api/clear-cache/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(currentDirs),
            });

            const data = await response.json();

            if (response.ok) {
                setMessage('Cache cleared successfully');
                setCurrentDirs(null);
            } else {
                setMessage(data.error || 'Failed to clear cache');
            }
        } catch (error) {
            setMessage('Error clearing cache');
            console.error('Clear cache error:', error);
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        
        if (files.length === 0) {
            setMessage('Please select files first');
            return;
        }

        setIsLoading(true);
        const formData = new FormData();
        files.forEach((file) => {
            formData.append('files', file);
        });
        formData.append('model', selectedModel);

        try {
            const response = await fetch('http://localhost:8000/api/upload-csv/', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                setMessage(
                    `Files processed successfully!\n` +
                    `Input directory: ${data.input_directory}\n` +
                    `Output directory: ${data.output_directory}\n` +
                    `Files processed: ${data.files_processed}`
                );
                setCurrentDirs({
                    input_directory: data.input_directory,
                    output_directory: data.output_directory
                });
                setFiles([]);
                event.target.reset();
            } else {
                setMessage(data.error || 'Upload failed');
            }
        } catch (error) {
            setMessage('Error uploading and processing files');
            console.error('Upload error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full">
            {/* Model Selection Section - Always visible */}
            <div className="bg-white rounded-lg shadow-md overflow-hidden mb-6 flex-shrink-0">
                <div className="p-6 bg-gradient-to-r from-blue-50 to-white">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">
                        Select a Model
                    </h3>
                    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                        {modelOptions.map((model) => (
                            <button
                                key={model}
                                onClick={() => setSelectedModel(model)}
                                className={`p-2.5 rounded-md border transition-all duration-200 
                                    ${selectedModel === model 
                                        ? 'border-blue-500 bg-blue-50 shadow-sm' 
                                        : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/50'}
                                    text-center`}
                            >
                                <span className="text-sm font-medium text-gray-700">
                                    {model}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* File Upload Section and Success Message */}
            <div className="flex-1 flex flex-col">
                {selectedModel && (
                    <>
                        <h2 className="text-2xl font-bold mb-4 text-gray-800 flex-shrink-0">Upload All Input CSV Files</h2>
                        
                        {/* {!currentDirs && ( */}
                            <form onSubmit={handleSubmit} className="flex flex-col space-y-4 flex-1">
                                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                                    <div className="text-center">
                                        <input
                                            type="file"
                                            accept=".csv"
                                            onChange={handleFileChange}
                                            multiple
                                            className="hidden"
                                            id="csv-upload"
                                        />
                                        <label
                                            htmlFor="csv-upload"
                                            className="cursor-pointer flex flex-col items-center space-y-2"
                                        >
                                            <span className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                                                Choose Files
                                            </span>
                                            <span className="text-sm text-gray-500">
                                                {files.length > 0 
                                                    ? `${files.length} files selected` 
                                                    : 'No files chosen'}
                                            </span>
                                        </label>
                                    </div>
                                </div>

                                {files.length > 0 && (
                                    <div className="text-sm text-gray-600 max-h-24 overflow-y-auto">
                                        <div className="font-medium mb-1">Selected files:</div>
                                        {files.map((file, index) => (
                                            <div key={index} className="pl-2">
                                                {file.name}
                                            </div>
                                        ))}
                                    </div>
                                )}

                                <div className="flex gap-4">
                                    <button
                                        type="submit"
                                        disabled={files.length === 0 || isLoading}
                                        className={`flex-1 py-3 rounded-lg text-white font-medium
                                                  ${files.length === 0 || isLoading 
                                                    ? 'bg-gray-400 cursor-not-allowed'
                                                    : 'bg-blue-600 hover:bg-blue-700'}`}
                                    >
                                        {isLoading ? 'Uploading and Preprocessing...' : 'Upload and Preprocess CSV Files'}
                                    </button>

                                    {currentDirs && (
                                        <button
                                            type="button"
                                            onClick={clearCache}
                                            className="px-4 py-3 rounded-lg text-white font-medium bg-red-600 hover:bg-red-700"
                                        >
                                            Clear Cache
                                        </button>
                                    )}
                                </div>
                            </form>
                        {/* )} */}

                        {message && (
                            <div className={`mt-4 p-3 rounded-lg whitespace-pre-line flex-shrink-0 ${
                                message.includes('success') || message.includes('cleared')
                                    ? 'bg-green-100 text-green-700'
                                    : 'bg-red-100 text-red-700'
                            }`}>
                                {message}
                            </div>
                        )}
                    </>
                )}
                
                {/* Training options only shown after successful upload */}
                {currentDirs && <ModelTrainingCard selectedModel={selectedModel} />}
            </div>
        </div>
    );
} 