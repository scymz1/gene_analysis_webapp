'use client';
import { useState } from 'react';
// import ModelTrainingCard from './ModelTrainingCard';
import { API_BASE_URL } from '../../config/urls';   

export default function CSVUploadCard({shape}) {
    const [files, setFiles] = useState([]);
    const [shapeFiles, setShapeFiles] = useState([]);
    const [fastqR1Files, setFastqR1Files] = useState([]);
    const [fastqR2Files, setFastqR2Files] = useState([]);
    const [message, setMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentDirs, setCurrentDirs] = useState(null);
    const [selectedModel, setSelectedModel] = useState('');
    const [inputMode, setInputMode] = useState('file'); // 'file' or 'text'
    const [sequenceText, setSequenceText] = useState('');
    const [shapeText, setShapeText] = useState('');
    const [visualizationUrl, setVisualizationUrl] = useState('');
    const [helixVisualizationUrl, setHelixVisualizationUrl] = useState('');
    const [isGeneratingVisualization, setIsGeneratingVisualization] = useState(false);
    
    // Generate unique IDs for this instance
    const fastaInputId = shape ? 'fasta-upload-shape' : 'fasta-upload-noshape';
    const shapeInputId = 'shape-upload';
    const fastqR1InputId = 'fastq-r1-upload';
    const fastqR2InputId = 'fastq-r2-upload';

    const modelOptions = shape ? [
        'seismic',
        'LinearSampling',
        'RNAstructure',
    ] : [
        'Ufold',
        'CNNfold',
        'mxfold2',
        'KnotFold',
        'LinearSampling',
        'redfold',
        'rnaFold',
        'sincFold',
    ];

    const handleFileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter(file => file.type === 'text/plain' || file.name.endsWith('.fasta'));
        
        if (validFiles.length !== selectedFiles.length) {
            setMessage('Please select only FASTA files');
        } else if (validFiles.length > 0) {
            setFiles(validFiles);
            setMessage('');
        }
    };

    const handleShapeFileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter(file => file.type === 'text/plain' || file.name.endsWith('.shape'));
        
        if (validFiles.length !== selectedFiles.length) {
            setMessage('Please select only SHAPE files');
        } else if (validFiles.length > 0) {
            setShapeFiles(validFiles);
            setMessage('');
        }
    };

    const handleFastqR1FileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter(file => 
            file.name.endsWith('.fastq') || 
            file.name.endsWith('.fq') || 
            file.name.endsWith('.fastq.gz') || 
            file.name.endsWith('.fq.gz')
        );
        
        if (validFiles.length !== selectedFiles.length) {
            setMessage('Please select only FASTQ files');
        } else if (validFiles.length > 0) {
            setFastqR1Files(validFiles);
            setMessage('');
        }
    };

    const handleFastqR2FileChange = (event) => {
        const selectedFiles = Array.from(event.target.files);
        const validFiles = selectedFiles.filter(file => 
            file.name.endsWith('.fastq') || 
            file.name.endsWith('.fq') || 
            file.name.endsWith('.fastq.gz') || 
            file.name.endsWith('.fq.gz')
        );
        
        if (validFiles.length !== selectedFiles.length) {
            setMessage('Please select only FASTQ files');
        } else if (validFiles.length > 0) {
            setFastqR2Files(validFiles);
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
        
        if (!selectedModel) {
            setMessage('Please select a model first');
            return;
        }

        // Check if seismic model is selected
        const isSeismicModel = selectedModel === 'seismic';

        if (inputMode === 'file') {
            if (files.length === 0) {
                setMessage('Please select a FASTA file');
                return;
            }
            
            if (shape && !isSeismicModel && shapeFiles.length === 0) {
                setMessage('Please select a SHAPE file');
                return;
            }

            if (shape && isSeismicModel) {
                if (fastqR1Files.length === 0) {
                    setMessage('Please select a FASTQ R1 file');
                    return;
                }
                if (fastqR2Files.length === 0) {
                    setMessage('Please select a FASTQ R2 file');
                    return;
                }
            }
        } else {
            if (!sequenceText.trim()) {
                setMessage('Please enter a sequence');
                return;
            }
            
            if (shape && !isSeismicModel && !shapeText.trim()) {
                setMessage('Please enter SHAPE data');
                return;
            }

            if (shape && isSeismicModel) {
                setMessage('Text input mode is not supported for seismic model. Please use file upload.');
                return;
            }
        }

        setIsLoading(true);
        const formData = new FormData();
        
        if (inputMode === 'file') {
            files.forEach((file) => {
                formData.append('files', file);
            });
            
            if (shape && !isSeismicModel && shapeFiles.length > 0) {
                formData.append('shape_files', shapeFiles[0]);
            }

            if (shape && isSeismicModel) {
                if (fastqR1Files.length > 0) {
                    formData.append('fastq_r1_files', fastqR1Files[0]);
                }
                if (fastqR2Files.length > 0) {
                    formData.append('fastq_r2_files', fastqR2Files[0]);
                }
            }
        } else {
            // Create a file from the text input
            const blob = new Blob([sequenceText], { type: 'text/plain' });
            formData.append('files', blob, 'sequence.fasta');
            
            if (shape && !isSeismicModel && shapeText.trim()) {
                const shapeBlob = new Blob([shapeText], { type: 'text/plain' });
                formData.append('shape_files', shapeBlob, 'sequence.shape');
            }
        }
        
        formData.append('model', selectedModel);
        formData.append('use_shape', shape ? 'true' : 'false');

        try {
            const response = await fetch(`${API_BASE_URL}/backend/api/upload-fasta/`, {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                },
                body: formData,
                credentials: 'include',
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
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
            setShapeFiles([]);
            setFastqR1Files([]);
            setFastqR2Files([]);
            setSequenceText('');
            setShapeText('');
            event.target.reset();
        } catch (error) {
            console.error('Upload error:', error);
            setMessage(`Error uploading and processing files: ${error.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    const generateVisualization = async () => {
        if (!currentDirs) {
            setMessage('No results to visualize');
            return;
        }

        setIsGeneratingVisualization(true);
        setMessage('Generating visualization...');

        try {
            const response = await fetch(`${API_BASE_URL}/backend/api/generate-visualization/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    output_dir: currentDirs.output_directory
                }),
            });

            const data = await response.json();

            if (response.ok) {
                setMessage('Visualization generated successfully');
                setVisualizationUrl(`${API_BASE_URL}${data.image_url}`);
                if (data.helix_image_url) {
                    setHelixVisualizationUrl(`${API_BASE_URL}${data.helix_image_url}`);
                }
            } else {
                setMessage(data.error || 'Failed to generate visualization');
            }
        } catch (error) {
            setMessage('Error generating visualization');
            console.error('Visualization error:', error);
        } finally {
            setIsGeneratingVisualization(false);
        }
    };

    // Check if seismic model is selected
    const isSeismicModel = selectedModel === 'seismic';

    return (
        <div className="flex flex-col h-full">
            {/* Title based on shape prop */}
            <div className="bg-gradient-to-r from-blue-100 to-indigo-100 p-4 rounded-lg shadow-sm mb-6 text-center">
                <h2 className="text-2xl font-bold text-gray-800">
                    {shape 
                        ? isSeismicModel 
                            ? "RNA Secondary Structure Prediction with SEISMIC"
                            : "RNA Secondary Structure Prediction with SHAPE Data" 
                        : "RNA Secondary Structure Prediction"}
                </h2>
            </div>

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
                                    flex items-center justify-center h-12
                                    ${selectedModel === model 
                                        ? 'border-blue-500 bg-blue-50 shadow-sm' 
                                        : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50/50'}`}
                            >
                                <span className="text-sm font-medium text-gray-700 text-center">
                                    {model}
                                </span>
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Add vertical spacing */}
            <div className="h-8"></div>

            {/* Input Mode Selection - Hide for seismic */}
            {(!shape || !isSeismicModel) && (
                <div className="mb-4 flex justify-center space-x-4">
                    <button
                        onClick={() => setInputMode('file')}
                        className={`px-4 py-2 rounded-lg transition-colors ${
                            inputMode === 'file'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                    >
                        Upload File
                    </button>
                    <button
                        onClick={() => setInputMode('text')}
                        className={`px-4 py-2 rounded-lg transition-colors ${
                            inputMode === 'text'
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                    >
                        Enter Sequence
                    </button>
                </div>
            )}

            {/* File Upload or Text Input Section */}
            <div className="flex-1 flex flex-col">
                <h2 className="text-2xl font-bold mb-4 text-gray-800 flex-shrink-0">
                    {inputMode === 'file' ? 'Upload Files' : 'Enter Sequence Data'}
                </h2>
                
                <form onSubmit={handleSubmit} className="flex flex-col space-y-4 flex-1">
                    {/* FASTA Input */}
                    <div className="space-y-2">
                        <label className="block text-sm font-medium text-gray-700">
                            {inputMode === 'file' ? 'FASTA File' : 'FASTA Sequence'}
                        </label>
                        
                        {inputMode === 'file' ? (
                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                                <div className="text-center">
                                    <input
                                        type="file"
                                        accept=".fasta,.txt"
                                        onChange={handleFileChange}
                                        className="hidden"
                                        id={fastaInputId}
                                    />
                                    <label
                                        htmlFor={fastaInputId}
                                        className="cursor-pointer flex flex-col items-center space-y-2"
                                    >
                                        <span className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors">
                                            Choose FASTA File
                                        </span>
                                        <span className="text-sm text-gray-500">
                                            {files.length > 0 
                                                ? files[0].name 
                                                : 'No file chosen'}
                                        </span>
                                    </label>
                                </div>
                            </div>
                        ) : (
                            <div className="border-2 border-gray-300 rounded-lg p-4 bg-gray-50">
                                <textarea
                                    value={sequenceText}
                                    onChange={(e) => setSequenceText(e.target.value)}
                                    placeholder={`Enter your FASTA sequence here...

Example:
>bpRNA_RFAM_27767
AUGCUGAAAGGUGGGGAAUCAGUGUGAAAUACAUUGGCUGUACCUGCAACCGUAAAGUCGGAGCGCCACCCAGCAUAGUCCGCUGUUGAAUGAAGGCCAGGAAAAGUCUAGUUCUACUAUUAAAAU`}
                                    className="w-full h-32 p-2 border rounded-md font-mono text-sm"
                                    rows={6}
                                />
                            </div>
                        )}
                    </div>
                    
                    {/* SHAPE Input - Only shown when shape=true and not seismic */}
                    {shape && !isSeismicModel && (
                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">
                                {inputMode === 'file' ? 'SHAPE File' : 'SHAPE Data'}
                            </label>
                            
                            {inputMode === 'file' ? (
                                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                                    <div className="text-center">
                                        <input
                                            type="file"
                                            accept=".shape,.txt"
                                            onChange={handleShapeFileChange}
                                            className="hidden"
                                            id={shapeInputId}
                                        />
                                        <label
                                            htmlFor={shapeInputId}
                                            className="cursor-pointer flex flex-col items-center space-y-2"
                                        >
                                            <span className="px-4 py-2 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors">
                                                Choose SHAPE File
                                            </span>
                                            <span className="text-sm text-gray-500">
                                                {shapeFiles.length > 0 
                                                    ? shapeFiles[0].name 
                                                    : 'No file chosen'}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            ) : (
                                <div className="border-2 border-gray-300 rounded-lg p-4 bg-gray-50">
                                    <textarea
                                        value={shapeText}
                                        onChange={(e) => setShapeText(e.target.value)}
                                        placeholder={`Enter your SHAPE data here...

Example:
1 0.1
2 0.2
3 0.3
...`}
                                        className="w-full h-32 p-2 border rounded-md font-mono text-sm"
                                        rows={6}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* FASTQ Paired-End Files - Only shown when shape=true and seismic model */}
                    {shape && isSeismicModel && (
                        <>
                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-700">
                                    FASTQ R1 File (Mate 1)
                                </label>
                                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                                    <div className="text-center">
                                        <input
                                            type="file"
                                            // accept=".fastq,.fq,.fastq.gz,.fq.gz"
                                            onChange={handleFastqR1FileChange}
                                            className="hidden"
                                            id={fastqR1InputId}
                                        />
                                        <label
                                            htmlFor={fastqR1InputId}
                                            className="cursor-pointer flex flex-col items-center space-y-2"
                                        >
                                            <span className="px-4 py-2 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors">
                                                Choose FASTQ R1 File
                                            </span>
                                            <span className="text-sm text-gray-500">
                                                {fastqR1Files.length > 0 
                                                    ? fastqR1Files[0].name 
                                                    : 'No file chosen'}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-2">
                                <label className="block text-sm font-medium text-gray-700">
                                    FASTQ R2 File (Mate 2)
                                </label>
                                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-32 flex items-center justify-center bg-gray-50">
                                    <div className="text-center">
                                        <input
                                            type="file"
                                            // accept=".fastq,.fq,.fastq.gz,.fq.gz"
                                            onChange={handleFastqR2FileChange}
                                            className="hidden"
                                            id={fastqR2InputId}
                                        />
                                        <label
                                            htmlFor={fastqR2InputId}
                                            className="cursor-pointer flex flex-col items-center space-y-2"
                                        >
                                            <span className="px-4 py-2 bg-purple-600 text-white rounded-full hover:bg-purple-700 transition-colors">
                                                Choose FASTQ R2 File
                                            </span>
                                            <span className="text-sm text-gray-500">
                                                {fastqR2Files.length > 0 
                                                    ? fastqR2Files[0].name 
                                                    : 'No file chosen'}
                                            </span>
                                        </label>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}

                    <div className="flex gap-4">
                        <button
                            type="submit"
                            disabled={isLoading || !selectedModel || 
                                (inputMode === 'file' && files.length === 0) ||
                                (inputMode === 'text' && !sequenceText.trim()) ||
                                (shape && !isSeismicModel && inputMode === 'file' && shapeFiles.length === 0) ||
                                (shape && !isSeismicModel && inputMode === 'text' && !shapeText.trim()) ||
                                (shape && isSeismicModel && fastqR1Files.length === 0) ||
                                (shape && isSeismicModel && fastqR2Files.length === 0)}
                            className={`flex-1 py-3 rounded-lg text-white font-medium
                                    ${(isLoading || !selectedModel || 
                                        (inputMode === 'file' && files.length === 0) ||
                                        (inputMode === 'text' && !sequenceText.trim()) ||
                                        (shape && !isSeismicModel && inputMode === 'file' && shapeFiles.length === 0) ||
                                        (shape && !isSeismicModel && inputMode === 'text' && !shapeText.trim()) ||
                                        (shape && isSeismicModel && fastqR1Files.length === 0) ||
                                        (shape && isSeismicModel && fastqR2Files.length === 0))
                                        ? 'bg-gray-400 cursor-not-allowed'
                                        : 'bg-blue-600 hover:bg-blue-700'}`}
                        >
                            {isLoading ? 'Processing...' : 
                             !selectedModel ? 'Please select a model first' :
                             inputMode === 'file' && files.length === 0 ? 'Please select a FASTA file' :
                             inputMode === 'text' && !sequenceText.trim() ? 'Please enter a sequence' :
                             shape && !isSeismicModel && inputMode === 'file' && shapeFiles.length === 0 ? 'Please select a SHAPE file' :
                             shape && !isSeismicModel && inputMode === 'text' && !shapeText.trim() ? 'Please enter SHAPE data' :
                             shape && isSeismicModel && fastqR1Files.length === 0 ? 'Please select a FASTQ R1 file' :
                             shape && isSeismicModel && fastqR2Files.length === 0 ? 'Please select a FASTQ R2 file' :
                             'Process Sequence'}
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

                {message && (
                    <div className={`mt-4 p-3 rounded-lg break-words whitespace-pre-wrap max-w-full overflow-x-auto
                        ${message.includes('success') || message.includes('cleared')
                            ? 'bg-green-100 text-green-700'
                            : 'bg-red-100 text-red-700'}`}
                    >
                        <div className="break-all">
                            {message}
                        </div>
                    </div>
                )}

                {currentDirs && (
                    <div className="mt-4 flex justify-center space-x-4">
                        <a
                            href={`${API_BASE_URL}/backend/api/download-result/?output_dir=${encodeURIComponent(currentDirs.output_directory)}`}
                            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                            download
                        >
                            Download Result
                        </a>
                        <button
                            onClick={generateVisualization}
                            disabled={isGeneratingVisualization}
                            className={`px-4 py-2 rounded-lg text-white
                                ${isGeneratingVisualization
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 transition-colors'}`}
                        >
                            {isGeneratingVisualization ? 'Generating...' : 'Generate Visualization'}
                        </button>
                    </div>
                )}

                {visualizationUrl && (
                    <div className="mt-6">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">RNA Structure Visualization</h3>
                        <div className="flex justify-center">
                            <img 
                                src={visualizationUrl} 
                                alt="RNA Structure Visualization" 
                                className="max-w-full h-auto border rounded-lg shadow-md"
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
                        <h3 className="text-lg font-medium text-gray-900 mb-4">
                            Double Helix Visualization
                        </h3>
                        <div className="flex justify-center">
                            <img 
                                src={helixVisualizationUrl} 
                                alt="Double Helix Visualization" 
                                className="max-w-full h-auto border rounded-lg shadow-md"
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
    );
} 