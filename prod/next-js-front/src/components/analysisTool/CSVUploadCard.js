'use client';
import { useState, useRef, useEffect } from 'react';
import ModelTrainingCard from './ModelTrainingCard';
console.log('process.env.NEXT_PUBLIC_API_URL:', process.env.NEXT_PUBLIC_API_URL);
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://scdrugmap.com';
export default function CSVUploadCard() {
    const [files, setFiles] = useState([]);
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentDirs, setCurrentDirs] = useState(null);
    const [selectedModel, setSelectedModel] = useState('');
    const [isFormatExpanded, setIsFormatExpanded] = useState(true);
    const messageEndRef = useRef(null);


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
            const response = await fetch(`${API_BASE_URL}/backend/api/clear-cache/`, {
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

    // useEffect(() => {
    //     return () => {
    //         if (currentDirs) {
    //             clearCache();
    //         }
    //     };
    // }, [currentDirs]);

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
            const response = await fetch(`${API_BASE_URL}/backend/api/upload-csv/`, {
                method: 'POST',
                body: formData,
            });

            setMessage('Start processing files... \n');

            if (response.ok) {

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let fullLog = '';
                let finalJson = null;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                
                    const chunk = decoder.decode(value);
                    fullLog += chunk;
                
                    // 实时显示日志
                    setMessage(prev => prev + chunk);
                }

                const lines = fullLog.trim().split('\n');
                const lastLine = lines[lines.length - 1];

                try {
                    finalJson = JSON.parse(lastLine);  // 最后一行是 print 出来的 JSON
                } catch (e) {
                    console.error('Failed to parse final result JSON:', e);
                }
                
                if (finalJson) {
                    setMessage(
                        `Files processed successfully!\n` +
                        `Input directory: ${finalJson.input_directory}\n` +
                        `Output directory: ${finalJson.output_directory}\n` +
                        `Files processed: ${finalJson.files_processed}`
                    );
                    setCurrentDirs({
                        input_directory: finalJson.input_directory,
                        output_directory: finalJson.output_directory
                    });
                    setFiles([]);
                }
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

    useEffect(() => {
        if (messageEndRef.current) {
            messageEndRef.current.scrollTop = messageEndRef.current.scrollHeight;
        }
    }, [message]);

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
                {/* {selectedModel && ( */}
                <h2 className="text-2xl font-bold mb-4 text-gray-800 flex-shrink-0">Upload All Input CSV Files</h2>
                   

                {/* Example Input File Format collapsible section */}
                <div className="mb-4">
                <button
                    type="button"
                    onClick={() => setIsFormatExpanded(!isFormatExpanded)}
                    aria-expanded={isFormatExpanded}
                    className="w-full flex items-center gap-2 py-2 px-3 rounded-md border bg-white hover:bg-gray-50 text-left transition"
                >
                    {/* Triangle icon (rotates on expand) */}
                    <svg
                    className={`h-4 w-4 transform transition-transform duration-200 ${isFormatExpanded ? 'rotate-90' : ''}`}
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    aria-hidden="true"
                    >
                    <path d="M7.293 14.707a1 1 0 0 1 0-1.414L10.586 10 7.293 6.707A1 1 0 1 1 8.707 5.293l4 4a1 1 0 0 1 0 1.414l-4 4a1 1 0 0 1-1.414 0z"/>
                    </svg>
                    <span className="font-medium text-gray-800">Example Input File Format</span>
                </button>

                {isFormatExpanded && (
                    <div className="mt-2 rounded-md border bg-gray-50 p-4 text-sm text-gray-700 space-y-2">
                    <h4 className="font-semibold">Description</h4>
                    <p>
                        This file is a <strong>tab-delimited text file (.tsv format)</strong> containing a gene
                        expression count matrix for single cells.
                    </p>

                    <p className="font-semibold">Structure:</p>
                    <ul className="list-disc pl-5 space-y-1">
                        <li>Each row represents a single cell.</li>
                        <li>Each column after the first two contains the raw count for a specific gene in that cell.</li>
                    </ul>

                    <p className="font-semibold">Columns:</p>
                    <ul className="list-disc pl-5 space-y-1">
                        <li>
                        <code>Cell_barcode</code> — Unique identifier for the cell (e.g., <code>C70R_C70R.bcDWVD</code>).
                        </li>
                        <li>
                        <code>Condition</code> — Experimental condition for that cell (e.g., <code>resistant</code>, <code>sensitive</code>).
                        </li>
                        <li>
                        Gene columns — Each remaining column corresponds to one gene symbol (e.g., <code>AAAS</code>, <code>AACS</code>, <code>AAED1</code>, …), with integer counts.
                        </li>
                    </ul>

                    {/* Download button (served from /public) */}
                    <a
                        href="/GSE104987_mal_countsMatrix.csv"
                        download
                        className="inline-block mt-3 px-4 py-2 rounded-md bg-blue-400 hover:bg-blue-500 text-white font-medium"
                    >
                        Download Example Input File
                    </a>
                    </div>
                )}
                </div>



                {/* {!currentDirs && ( */}
                    <form onSubmit={handleSubmit} className="flex flex-col space-y-4 flex-1">
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                            <div className="text-center">
                                <input
                                    type="file"
                                    accept=".csv,.tsv,.txt"
                                    onChange={handleFileChange}
                                    multiple
                                    className="hidden"
                                    id="csv-upload"
                                />
                                <label
                                    htmlFor="csv-upload"
                                    className="cursor-pointer flex flex-col items-center space-y-2"
                                >
                                    <span className="px-4 py-2 bg-blue-400 text-white rounded-full hover:bg-blue-500 transition-colors">
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
                                {isLoading ? 'Uploading and Preprocessing... (This may take a long time)' : 'Upload and Preprocess CSV Files'}
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
                    <div
                        ref={messageEndRef}
                        className={`mt-4 p-3 rounded-lg whitespace-pre-line flex-shrink-0 overflow-y-auto bg-white border text-sm ${
                            message.includes('success') || message.includes('cleared')
                                ? 'border-green-400 text-green-700'
                                : message.includes('Start processing files...')
                                    ? 'border-yellow-400 text-yellow-700'
                                    : 'border-red-400 text-red-700'
                        }`}
                        style={{
                            maxHeight: '200px',
                            maxWidth: '100%',
                            wordWrap: 'break-word',
                            overflowWrap: 'break-word',
                            fontSize: '0.85rem',
                            backgroundColor: '#f9fafb'
                        }}
                    >
                        <pre style={{ whiteSpace: 'pre-wrap' }}>{message}</pre>
                    </div>
                )}
                {/* )} */}
                
                {/* Training options only shown after successful upload */}
                {currentDirs && <ModelTrainingCard selectedModel={selectedModel} currentDirs={currentDirs}/>}
            </div>
        </div>
    );
} 