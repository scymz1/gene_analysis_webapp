'use client';

import { useState, useEffect } from 'react';
import { FaFolderClosed } from "react-icons/fa6";
import { BsFiletypeTxt, BsFiletypeCsv } from "react-icons/bs";
import { CiFileOn } from "react-icons/ci";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://scdrugmap.com';

export default function DataPage() {
    const [files, setFiles] = useState([]);
    const [currentPath, setCurrentPath] = useState('home/lxndt_filter');
    const [isLoading, setIsLoading] = useState(true);
    const [selectedFiles, setSelectedFiles] = useState(new Set());
    const [mounted, setMounted] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
    const [selectedImage, setSelectedImage] = useState(null);
    const [isImageModalOpen, setIsImageModalOpen] = useState(false);

    useEffect(() => {
        setMounted(true);
        fetchFiles(currentPath);
    }, [currentPath]);

    const fetchFiles = async (path) => {
        try {
            const truePath = path.split('/').slice(1).join('/');
            console.log('Fetching files from:', truePath); // Debug log
            const response = await fetch(`${API_BASE_URL}/backend/api/files/?path=${encodeURIComponent(truePath)}`, {
                method: 'GET',
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            setFiles(data.files);
        } catch (error) {
            console.error('Error fetching files:', error);
            setFiles([]);
        } finally {
            setIsLoading(false);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    // const formatDate = (date) => {
    //     // const d = new Date(date);
    //     // Use a fixed format that will be consistent between server and client
    //     // return d.toISOString().split('.')[0].replace('T', ' ');
    // };

    const handleBreadcrumbClick = (index) => {
        const newPath = currentPath.split('/').slice(0, index + 1).join('/');
        setCurrentPath(newPath);
    };

    const handleFolderClick = (folderName) => {
        // Use path.join equivalent in JS by handling slashes properly
        const newPath = currentPath.endsWith('/') 
            ? currentPath + folderName
            : currentPath + '/' + folderName;
        setCurrentPath(newPath);
    };

    const handleFileSelect = (fileName) => {
        setSelectedFiles((prev) => {
            const newSet = new Set(prev);
            if (newSet.has(fileName)) {
                newSet.delete(fileName);
            } else {
                newSet.add(fileName);
            }
            return newSet;
        });
    };

    const handleSelectAll = () => {
        if (selectedFiles.size === files.length) {
            setSelectedFiles(new Set());
        } else {
            setSelectedFiles(new Set(files.map(file => file.name)));
        }
    };

    const handleDownload = async (fileName) => {
        try {
            const truePath = currentPath.split('/').slice(1).join('/');
            const response = await fetch(`${API_BASE_URL}/backend/api/download/?path=${encodeURIComponent(`${truePath}/${fileName}`)}`);
            
            if (!response.ok) throw new Error('Download failed');
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Download error:', error);
        }
    };

    const handleBulkDownload = async () => {
        if (selectedFiles.size === 0) return;
        
        try {
            setIsDownloading(true);
            console.log('Starting bulk download...', selectedFiles);
            
            const response = await fetch(`${API_BASE_URL}/backend/api/bulk-download/`, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    path: currentPath.split('/').slice(1).join('/'),
                    files: Array.from(selectedFiles)
                }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'selected_files.zip';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Bulk download error:', error);
            alert('Failed to download files. Please check the console for details.');
        } finally {
            setIsDownloading(false);
        }
    };

    const getFileIcon = (file) => {
        if (file.isDirectory) {
            return <FaFolderClosed className="text-yellow-600" />;
        }
        
        const extension = file.name.split('.').pop().toLowerCase();
        switch (extension) {
            case 'txt':
                return <BsFiletypeTxt className="text-gray-600" />;
            case 'csv':
                return <BsFiletypeCsv className="text-green-600" />;
            default:
                return <CiFileOn className="text-gray-600" />;
        }
    };

    const getUmapImagePath = (fileName) => {
        if (fileName && fileName.includes('.')) {
            const baseName = fileName.split('.')[0];
            return `${API_BASE_URL}/backend/api/umap-image/?path=lxndt_filter_imgs/${baseName}.UMAP.png`;
        }
        return null;
    };

    const handleImageClick = (fileName) => {
        const imagePath = getUmapImagePath(fileName);
        if (imagePath) {
            setSelectedImage(imagePath);
            setIsImageModalOpen(true);
        }
    };

    const closeImageModal = () => {
        setIsImageModalOpen(false);
        setSelectedImage(null);
    };

    if (!mounted) {
        return null;
    }

    return (
        <div className="p-10 min-h-screen dark:bg-gray-900">
            <div className="max-w-7xl mx-auto">
                {/* Breadcrumb and Button Container */}
                <div className="flex justify-between items-center mb-6">
                    {/* Breadcrumb */}
                    <nav className="flex" aria-label="Breadcrumb">
                        <ol className="inline-flex items-center space-x-1 md:space-x-2 rtl:space-x-reverse">
                            {currentPath.split('/').map((segment, index, array) => (
                                <li key={index} className="inline-flex items-center">
                                    <a
                                        href="#"
                                        onClick={() => handleBreadcrumbClick(index)}
                                        className={`inline-flex items-center text-sm font-medium ${
                                            index === array.length - 1
                                                ? 'text-gray-500 dark:text-gray-400'
                                                : 'text-gray-700 hover:text-blue-600 dark:text-gray-400 dark:hover:text-white'
                                        }`}
                                    >
                                        {index === 0 ? (
                                            <svg
                                                className="w-3 h-3 me-2.5"
                                                aria-hidden="true"
                                                xmlns="http://www.w3.org/2000/svg"
                                                fill="currentColor"
                                                viewBox="0 0 20 20"
                                            >
                                                <path d="m19.707 9.293-2-2-7-7a1 1 0 0 0-1.414 0l-7 7-2 2a1 1 0 0 0 1.414 1.414L2 10.414V18a2 2 0 0 0 2 2h3a1 1 0 0 0 1-1v-4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v4a1 1 0 0 0 1 1h3a2 2 0 0 0 2-2v-7.586l.293.293a1 1 0 0 0 1.414-1.414Z" />
                                            </svg>
                                        ) : (
                                            <svg
                                                className="rtl:rotate-180 w-3 h-3 text-gray-400 mx-1"
                                                aria-hidden="true"
                                                xmlns="http://www.w3.org/2000/svg"
                                                fill="none"
                                                viewBox="0 0 6 10"
                                            >
                                                <path
                                                    stroke="currentColor"
                                                    strokeLinecap="round"
                                                    strokeLinejoin="round"
                                                    strokeWidth="2"
                                                    d="m1 9 4-4-4-4"
                                                />
                                            </svg>
                                        )}
                                        {segment || 'Home'}
                                    </a>
                                </li>
                            ))}
                        </ol>
                    </nav>

                    {/* Download Button - Always visible */}
                    <button
                        onClick={handleBulkDownload}
                        disabled={selectedFiles.size === 0 || isDownloading}
                        className="inline-flex items-center px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
                    >
                        {isDownloading ? (
                            <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Preparing Download...
                            </>
                        ) : (
                            <>
                                <svg className="w-4 h-4 mr-2" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 16 18">
                                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 1v11m0 0 4-4m-4 4L4 8m11 4v3a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2v-3"/>
                                </svg>
                                {selectedFiles.size > 0 ? `Download Selected (${selectedFiles.size})` : 'Download Selected'}
                            </>
                        )}
                    </button>
                </div>

                {/* File Table */}
                <div className="overflow-x-auto rounded-lg shadow">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs uppercase bg-gray-50 dark:bg-gray-800">
                            <tr>
                                <th className="px-6 py-3">
                                    <input
                                        type="checkbox"
                                        checked={selectedFiles.size === files.length}
                                        onChange={handleSelectAll}
                                    />
                                </th>
                                <th className="px-6 py-3 text-gray-700 dark:text-gray-300">Name</th>
                                <th className="px-6 py-3 text-gray-700 dark:text-gray-300">Date Modified</th>
                                <th className="px-6 py-3 text-gray-700 dark:text-gray-300">Size</th>
                                <th className="px-6 py-3 text-gray-700 dark:text-gray-300">Kind</th>
                                <th className="px-6 py-3 text-gray-700 dark:text-gray-300">UMAP</th>
                            </tr>
                        </thead>
                        <tbody>
                            {isLoading ? (
                                <tr className="bg-white dark:bg-gray-900">
                                    <td colSpan="7" className="px-6 py-4 text-center text-gray-700 dark:text-gray-300">
                                        Loading...
                                    </td>
                                </tr>
                            ) : (
                                files.map((file, index) => (
                                    <tr
                                        key={index}
                                        className="group bg-white dark:bg-gray-900 border-b dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800"
                                    >
                                        <td className="px-6 py-4">
                                            <input
                                                type="checkbox"
                                                checked={selectedFiles.has(file.name)}
                                                onChange={() => handleFileSelect(file.name)}
                                            />
                                        </td>
                                        <td
                                            className="px-6 py-4 text-gray-700 dark:text-gray-300 flex items-center gap-2 cursor-pointer relative"
                                            onClick={() => file.isDirectory && handleFolderClick(file.name)}
                                        >
                                            {getFileIcon(file)} {file.name}
                                            {!file.isDirectory && (
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleDownload(file.name);
                                                    }}
                                                    className="invisible group-hover:visible absolute right-2 hover:text-blue-600"
                                                >
                                                    <svg className="w-5 h-5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 16 18">
                                                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 1v11m0 0 4-4m-4 4L4 8m11 4v3a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2v-3"/>
                                                    </svg>
                                                </button>
                                            )}
                                        </td>
                                        <td className="px-6 py-4 text-gray-700 dark:text-gray-300">
                                            {file.modifiedTime}
                                        </td>
                                        <td className="px-6 py-4 text-gray-700 dark:text-gray-300">
                                            {formatFileSize(file.size)}
                                        </td>
                                        <td className="px-6 py-4 text-gray-700 dark:text-gray-300">
                                            {file.isDirectory ? 'Folder' : file.name.split('.').pop()}
                                        </td>
                                        <td className="px-6 py-4 text-gray-700 dark:text-gray-300">
                                            {!file.isDirectory && file.name.endsWith('.txt') ? (
                                                <div className="flex justify-center">
                                                    <img
                                                        src={getUmapImagePath(file.name)}
                                                        alt={`UMAP for ${file.name}`}
                                                        className="w-9 h-9 object-cover rounded cursor-pointer hover:opacity-80 transition-opacity"
                                                        onClick={() => handleImageClick(file.name)}
                                                        onError={(e) => {
                                                            e.target.style.display = 'none';
                                                        }}
                                                    />
                                                </div>
                                            ) : (
                                                <span className="text-gray-400">-</span>
                                            )}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Image Modal */}
            {isImageModalOpen && selectedImage && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="relative max-w-4xl max-h-full p-4">
                        <button
                            onClick={closeImageModal}
                            className="absolute top-2 right-2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-100 z-10"
                        >
                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                        <img
                            src={selectedImage}
                            alt="UMAP Visualization"
                            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
