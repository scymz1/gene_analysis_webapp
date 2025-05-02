'use client';
import { useState } from 'react';
import { API_BASE_URL } from '../../config/urls';

export default function DataPage() {
    const [file, setFile] = useState(null);
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [visualizationUrl, setVisualizationUrl] = useState('');
    const [helixVisualizationUrl, setHelixVisualizationUrl] = useState('');
	//  const [outputDir, setOutputDir] = useState('');

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        if (selectedFile && (selectedFile.name.endsWith('.db') || selectedFile.type === 'text/plain')) {
            setFile(selectedFile);
            setMessage('');
        } else {
            setFile(null);
            setMessage('Please select a valid .db file');
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        
        if (!file) {
            setMessage('Please select a file first');
            return;
        }

        setIsLoading(true);
        setMessage('Uploading file...');

        try {
            // First, upload the file to create a temporary directory
            const formData = new FormData();
            formData.append('files', file);
            formData.append('model', 'visualization_only'); // Special flag for backend
            formData.append('use_shape', 'false');

            const uploadResponse = await fetch(`${API_BASE_URL}/backend/api/upload-fasta/`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                },
                body: formData,
                credentials: 'include',
            });

            if (!uploadResponse.ok) {
                throw new Error(`HTTP error! status: ${uploadResponse.status}`);
            }

            const uploadData = await uploadResponse.json();
            setOutputDir(uploadData.output_directory);
            
            // Now generate the visualization
            setMessage('Generating visualization...');
            
            const visualizationResponse = await fetch(`${API_BASE_URL}/backend/api/generate-visualization/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    output_dir: uploadData.output_directory
                }),
            });

            if (!visualizationResponse.ok) {
                throw new Error(`HTTP error! status: ${visualizationResponse.status}`);
            }

            const visualizationData = await visualizationResponse.json();
            
            setMessage('Visualization generated successfully!');
            setVisualizationUrl(`${API_BASE_URL}${visualizationData.image_url}`);
            if (visualizationData.helix_image_url) {
                setHelixVisualizationUrl(`${API_BASE_URL}${visualizationData.helix_image_url}`);
            }
            
        } catch (error) {
            console.error('Error:', error);
            setMessage(`Error: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const clearResults = () => {
        setFile(null);
        setMessage('');
        setVisualizationUrl('');
        setHelixVisualizationUrl('');
        setOutputDir('');
    };

    return (
        <div className="p-10 min-h-screen dark:bg-gray-900">
            <div className="max-w-7xl mx-auto">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
                    RNA Structure Visualization
                </h1>

                {/* Sample Images Card */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden mb-8">
                    <div className="p-6">
                        <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
                            Sample Visualizations
                        </h2>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                            Here are some example RNA structure visualizations that can be generated using our tools.
                        </p>
                        <div className="grid grid-cols-3 gap-6">
                            <div className="relative group h-[300px]">
                                <img 
                                    src={`${API_BASE_URL}/backend/api/download-result/?output_dir=AnalysisTools/visualization/sample_images&file=rna_structure%20(1).png`}
                                    alt="Sample RNA Structure 1"
                                    className="w-full h-full object-contain rounded-lg shadow-md transition-transform duration-300 group-hover:scale-105"
                                />
                            </div>
                            <div className="relative group h-[300px]">
                                <img 
                                    src={`${API_BASE_URL}/backend/api/download-result/?output_dir=AnalysisTools/visualization/sample_images&file=rna_structure%20(2).png`}
                                    alt="Sample RNA Structure 2"
                                    className="w-full h-full object-contain rounded-lg shadow-md transition-transform duration-300 group-hover:scale-105"
                                />
                            </div>
                            <div className="relative group h-[300px]">
                                <img 
                                    src={`${API_BASE_URL}/backend/api/download-result/?output_dir=AnalysisTools/visualization/sample_images&file=rna_structure%20(3).png`}
                                    alt="Sample RNA Structure 3"
                                    className="w-full h-full object-contain rounded-lg shadow-md transition-transform duration-300 group-hover:scale-105"
                                />
                            </div>
                        </div>
                    </div>
                </div>

                {/* Upload Form Card */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
                    <div className="p-6">
                        <h2 className="text-xl font-semibold text-gray-800 dark:text-white mb-4">
                            Upload Your Result
                        </h2>
                        
                        <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg mb-6">
                            <p className="text-sm text-blue-800 dark:text-blue-200">
                                Upload a result.db file to generate a visualization of the RNA secondary structure.
                            </p>
                        </div>
                        
                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Result DB File
                                </label>
                                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50 dark:bg-gray-700/50 hover:border-blue-500 dark:hover:border-blue-400 transition-colors">
                                    <div className="text-center">
                                        <input
                                            type="file"
                                            accept=".db,.txt"
                                            onChange={handleFileChange}
                                            className="hidden"
                                            id="result-db-upload"
                                        />
                                        <label
                                            htmlFor="result-db-upload"
                                            className="cursor-pointer flex flex-col items-center space-y-2"
                                        >
                                            <span className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                                                Choose File
                                            </span>
                                            <span className="text-sm text-gray-500 dark:text-gray-400">
                                                {file ? file.name : 'No file chosen'}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            </div>
                            
                            <div className="flex gap-4">
                                <button
                                    type="submit"
                                    disabled={isLoading || !file}
                                    className={`flex-1 py-3 rounded-lg text-white font-medium transition-colors
                                        ${(isLoading || !file)
                                            ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed'
                                            : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'}`}
                                >
                                    {isLoading ? 'Processing...' : 
                                     !file ? 'Please select a file first' : 
                                     'Generate Visualization'}
                                </button>
                                
                                {visualizationUrl && (
                                    <button
                                        type="button"
                                        onClick={clearResults}
                                        className="px-4 py-3 rounded-lg text-white font-medium bg-red-600 hover:bg-red-700 dark:bg-red-500 dark:hover:bg-red-600 transition-colors"
                                    >
                                        Clear Results
                                    </button>
                                )}
                            </div>
                        </form>
                        
                        {message && (
                            <div className={`mt-4 p-3 rounded-lg break-words whitespace-pre-wrap max-w-full overflow-x-auto
                                ${message.includes('success')
                                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
                                    : message.includes('Error')
                                        ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'
                                        : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'}`}
                            >
                                <div className="break-all">
                                    {message}
                                </div>
                            </div>
                        )}
                        
                        {visualizationUrl && (
                            <div className="mt-6">
                                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                                    RNA Structure Visualization
                                </h3>
                                <div className="flex justify-center">
                                    <img 
                                        src={visualizationUrl} 
                                        alt="RNA Structure Visualization" 
                                        className="max-w-full h-auto border rounded-lg shadow-md dark:border-gray-700"
                                    />
                                </div>
                                <div className="mt-4 flex justify-center">
                                    <a
                                        href={visualizationUrl}
                                        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                        download
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        Download Image
                                    </a>
                                </div>
                            </div>
                        )}

                        {helixVisualizationUrl && (
                            <div className="mt-8">
                                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                                    Double Helix Visualization
                                </h3>
                                <div className="flex justify-center">
                                    <img 
                                        src={helixVisualizationUrl} 
                                        alt="Double Helix Visualization" 
                                        className="max-w-full h-auto border rounded-lg shadow-md dark:border-gray-700"
                                    />
                                </div>
                                <div className="mt-4 flex justify-center">
                                    <a
                                        href={helixVisualizationUrl}
                                        className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                                        download
                                        target="_blank"
                                        rel="noopener noreferrer"
                                    >
                                        Download Helix Image
                                    </a>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
